#include <gtest/gtest.h>
#include "common/test_framework.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "backends/cpu/renderercpu.h"
#include "optimizers/mapOptimizer.h"

TEST_F(RendererTestBase, ComputeMap)
{
    const long long acceptableTimeMs = 30;
    const float errorThreshold = 0.023; // best = 0.022444;

    std::chrono::milliseconds accProcessingTime = std::chrono::milliseconds(0);
    float accError = 0;
    int framesProcessedCounter = 0;

    std::vector<float> s_pos_buff_, s_tex_buff_, s_wei_buff_;
    std::vector<unsigned int> s_idx_buff_;
    CreateScreenQuad(s_pos_buff_, s_tex_buff_, s_wei_buff_, s_idx_buff_);
    MeshCPU screen_mesh(s_pos_buff_, s_tex_buff_, s_wei_buff_, s_idx_buff_);

    DepthRendererCPU depth_renderer;
    ImageRendererCPU image_renderer;
    DIDxyRendererCPU didxy_renderer;
    ResidualRendererCPU residual_renderer;
    L2RendererCPU l2_renderer;
    NodataReducerCPU nodata_reducer;

    MapOptimizer optimizer(w_, h_, false);

    std::vector<Frame> frames;
    KeyFrame *kframe;

    TextureCPU<float> image_cpu(w_, h_, -1);
    TextureCPU<float> depth_cpu(w_, h_, -1);
    TextureCPU<Vec3> didxy_cpu(w_, h_, Vec3(0.0, 0.0, 0.0));
    TextureCPU<float> l2_cpu(w_, h_, -1);

    for (unsigned int i = 0; i < image_files_.size(); i++)
    {
        std::cout << "Frame " << i << std::endl;

        cv::Mat image_cv = ReadMat(image_files_[i]);
        cv::Mat depth_cv = ReadMat(depth_files_[i]) / depth_factor_;
        SE3 gt_pose = poses_[i];
        cv::Mat depth_mask = (depth_cv > 0.0);
        cv::Scalar depth_mean = cv::mean(depth_cv, depth_mask);
        depth_cv = depth_cv * mesh_vo::mapping_mean_depth / depth_mean[0];

        UploadMatToTexture(image_cpu, 0, image_cv);
        UploadMatToTexture(depth_cpu, 0, depth_cv);

        for (int lvl = 0; lvl < didxy_cpu.levels(); lvl++)
            didxy_renderer.Render(screen_mesh, lvl, lvl, image_cpu, didxy_cpu);

        if (i == 0)
        {
            std::vector<float> pos_buff_, tex_buff_, wei_buff_;
            std::vector<unsigned int> idx_buff_;
            CreateMesh(depth_cpu, cam_, 32, pos_buff_, tex_buff_, wei_buff_, idx_buff_);
            // CreateMesh(mesh_vo::mapping_mean_depth * 0.5, mesh_vo::mapping_mean_depth * 1.5, cam_, mesh_vo::mesh_width, pos_buff_, tex_buff_, wei_buff_, idx_buff_);
            MeshCPU mesh(pos_buff_, tex_buff_, wei_buff_, idx_buff_);
            kframe = new KeyFrame(Frame(image_cpu, didxy_cpu, 0, SE3(), gt_pose), mesh, depth_mean[0]);
            continue;
        }

        SE3 init_local_pose = kframe->globalPoseToLocal(gt_pose);

        Frame frame(image_cpu, didxy_cpu, i, init_local_pose, gt_pose);

        float minViewAngle = M_PI;
        for (int i = 0; i < frames.size(); i++)
        {
            float viewAngle = kframe->meanViewAngle(frame.local_pose(), frames[i].local_pose());
            if (viewAngle < minViewAngle)
                minViewAngle = viewAngle;
        }

        if (minViewAngle < mesh_vo::last_min_angle)
            continue;

        frames.push_back(frame);
        if (frames.size() > mesh_vo::num_frames)
        {
            frames.erase(frames.begin());
        }
        else
        {
            continue;
        }

        int kframeIndex = frames.size() / 2;
        std::vector<Frame> oframes = frames;
        oframes.erase(oframes.begin() + kframeIndex);

        std::vector<float> pos_buff_, tex_buff_, wei_buff_;
        std::vector<unsigned int> idx_buff_;
        CreateMesh(depth_cpu, cam_, 32, pos_buff_, tex_buff_, wei_buff_, idx_buff_);
        // CreateMesh(mesh_vo::mapping_mean_depth * 0.5, mesh_vo::mapping_mean_depth * 1.5, cam_, mesh_vo::mesh_width, pos_buff_, tex_buff_, wei_buff_, idx_buff_);
        MeshCPU mesh(pos_buff_, tex_buff_, wei_buff_, idx_buff_);
        kframe = new KeyFrame(frames[kframeIndex], mesh, depth_mean[0]);

        frame.local_pose() = kframe->globalPoseToLocal(frame.global_pose());

        for (int i = 0; i < frames.size(); i++)
        {
            frames[i].local_pose() = kframe->globalPoseToLocal(frames[i].global_pose());
        }

        for (int i = 0; i < oframes.size(); i++)
        {
            oframes[i].local_pose() = kframe->globalPoseToLocal(oframes[i].global_pose());
        }

        auto startTime = std::chrono::high_resolution_clock::now();
        for (int lvl = mesh_vo::mapping_ini_lvl; lvl >= mesh_vo::mapping_fin_lvl; lvl--)
        {
            optimizer.init(oframes, *kframe, cam_, lvl);
            while (!optimizer.converged())
            {
                optimizer.step(oframes, *kframe, cam_, lvl);
            }
        }
        auto endTime = std::chrono::high_resolution_clock::now();

        residual_renderer.Render(kframe->mesh(), frame.local_pose(), cam_, 1, 1, kframe->frame().image(), frame.image(), l2_cpu);
        cv::Mat l2_mat = DownloadTexture(l2_cpu, 1, CV_32FC1);
        SaveDebugImageColor(l2_mat, "l2_" + std::to_string(i) + ".png");

        depth_renderer.Render(kframe->mesh(), frame.local_pose(), cam_, 1, depth_cpu);
        cv::Mat depth_mat = DownloadTexture(depth_cpu, 1, CV_32FC1);
        SaveDebugImageColor(depth_mat, "depth_" + std::to_string(i) + ".png");

        // depth_renderer.Render(kframe->mesh(), oframes[oframes.size() - 1].local_pose(), cam_, 0, depth_cpu);
        float error = 1.0; // computeImageError(depth_cpu.get(0), gtDepthData);

        accProcessingTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        accError += error;
        framesProcessedCounter++;
    }

    auto meanDuration = accProcessingTime.count() / framesProcessedCounter;
    float meanError = accError / framesProcessedCounter;
    std::cout << "Mean processing time " << meanDuration << " ms" << std::endl;
    std::cout << "Mean error " << meanError << " ms" << std::endl;

    // The test passes if the error is below the threshold
    EXPECT_LT(meanError, errorThreshold)
        << "mean error (" << meanError
        << ") exceeds the acceptable threshold (" << errorThreshold << ").";

    // EXPECT_LE(durationMs, acceptableTimeMs)
    //     << "Pose estimation took " << durationMs << "ms, which exceeds the acceptable threshold of "
    //     << acceptableTimeMs << "ms.";
}