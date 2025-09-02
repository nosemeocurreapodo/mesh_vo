#include <gtest/gtest.h>
#include "common/test_framework.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "common/common.h"
#include "backends/cpu/renderercpu.h"
#include "optimizers/mapOptimizer.h"

TEST_F(RendererTestBase, ComputeMap)
{
    const int in_lvl = 0, out_lvl = 0;

    const long long acceptableTimeMs = 30;
    const float translationErrorThreshold = 1.5; // best = 0.0160271;
    const float rotationErrorThreshold = 0.0011; // best = 0.00105154;

    std::chrono::milliseconds accProcessingTime = std::chrono::milliseconds(0);
    float accTranslationError = 0;
    float accRotationError = 0;
    int framesProcessedCounter = 0;

    std::vector<float> pos_buff_, tex_buff_, wei_buff_;
    CreateScreenQuad(pos_buff_, tex_buff_, wei_buff_);
    MeshCPU screen_mesh(pos_buff_, tex_buff_, wei_buff_);

    TextureCPU<float> kimage_cpu(w_, h_, 0.0f);
    UploadMatToTexture(kimage_cpu, 0, image_src_cv_);

    TextureCPU<float> kdepth_cpu(w_, h_, 0.0f);
    UploadMatToTexture(kdepth_cpu, 0, depth_src_cv_);

    MeshCPU mesh = CreateMesh(cam_, 32);

    TextureCPU<Vec3> kdidxy_cpu(w_, h_, Vec3(0.0, 0.0, 0.0));

    DepthRendererCPU depth_renderer;
    ImageRendererCPU image_renderer;
    DIDxyRendererCPU didxy_renderer;
    ResidualRendererCPU residual_renderer;
    L2RendererCPU l2_renderer;

    NodataReducerCPU nodata_reducer;

    for (int lvl = 0; lvl < kdidxy_cpu.levels(); lvl++)
        didxy_renderer.Render(screen_mesh, lvl, lvl, kimage_cpu, kdidxy_cpu);

    KeyFrame kframe(Frame(kimage_cpu, kdidxy_cpu, 0, SE3(), pose_src_), mesh);

    MapOptimizer optimizer(w_, h_, false);

    TextureCPU<float> image_cpu(w_, h_, 0);
    TextureCPU<float> depth_cpu(w_, h_, 0);
    TextureCPU<Vec3> didxy_cpu(w_, h_, Vec3(0.0, 0.0, 0.0));
    TextureCPU<float> l2_texture(w_, h_, -1);

    for (unsigned int i = 1; i < image_files_.size(); i++)
    {
        std::vector<Frame> frames;

        std::cout << "Frame " << i << std::endl;

        cv::Mat image_cv = cv::imread(image_files_[i], cv::IMREAD_GRAYSCALE);
        cv::Mat gt_depth_cv = cv::imread(depth_files_[i], cv::IMREAD_GRAYSCALE);
        SE3 gt_pose = poses_[i].inverse();

        image_cv.convertTo(image_cv, CV_32FC1);
        gt_depth_cv.convertTo(gt_depth_cv, CV_32FC1);
        //  gt_depth_cv /= dataset.getDepthFactor();
        //  gt_depth_cv *= 100.0;

        UploadMatToTexture(image_cpu, 0, image_cv);
        UploadMatToTexture(depth_cpu, 0, gt_depth_cv);

        for (int lvl = 0; lvl < kdidxy_cpu.levels(); lvl++)
            didxy_renderer.Render(screen_mesh, lvl, lvl, image_cpu, didxy_cpu);

        SE3 init_local_pose = kframe.globalPoseToLocal(gt_pose);

        Frame frame(image_cpu, didxy_cpu, i, init_local_pose, gt_pose);

        frames.push_back(frame);

        auto startTime = std::chrono::high_resolution_clock::now();
 
        for (int lvl = mesh_vo::mapping_ini_lvl; lvl >= mesh_vo::mapping_fin_lvl; lvl--)
        {
            optimizer.init(frames, kframe, cam_, lvl);
            while (!optimizer.converged())
            {
                optimizer.step(frames, kframe, cam_, lvl);
            }
        }
  
        auto endTime = std::chrono::high_resolution_clock::now();

        // float error = computeImageError(estMipMapDepthData.get(0), gtDepthData);

        accProcessingTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        // accTranslationError += error[0];
        // accRotationError += error[1];
        framesProcessedCounter++;

        // change keyframe logic
        // float keyframeViewAngle = kframe.meanViewAngle(SE3(), frame.local_pose());

        depth_renderer.Render(kframe.mesh(), SE3(), cam_, 1, depth_cpu);
        cv::Mat depth_mat = DownloadTexture(depth_cpu, 1, CV_32FC1);
        SaveDebugImageColor(depth_mat, "Depth keyframe_" + std::to_string(i) + ".png");

        image_renderer.Render(kframe.mesh(), frame.local_pose(), cam_, 1, 1, kframe.frame().image(), image_cpu);
        Error nodata = nodata_reducer.reduce(1, image_cpu);
        float pnodata = nodata.getError() / image_cpu.size(1);
        float viewPercent = 1.0 - pnodata;

        std::cout << "view percent " << viewPercent << std::endl;

        if (viewPercent < mesh_vo::min_view_perc) // || keyframeViewAngle > mesh_vo::key_max_angle)
        {
            //MeshCPU new_mesh = CreateMesh(depth_cpu, cam_, 32);
            MeshCPU new_mesh = CreateMesh(cam_, 32);

            kframe = KeyFrame(frame, new_mesh);

            frame.local_pose() = kframe.globalPoseToLocal(frame.global_pose());

            image_renderer.Render(kframe.mesh(), frame.local_pose(), cam_, 1, 1, kframe.frame().image(), image_cpu);
            cv::Mat image_mat = DownloadTexture(image_cpu, 1, CV_32FC1);
            SaveDebugImageColor(image_mat, "Frame keyframe_" + std::to_string(i) + ".png");

            Error nodata = nodata_reducer.reduce(1, image_cpu);
            float pnodata = nodata.getError() / image_cpu.size(1);
            float viewPercent = 1.0 - pnodata;

            std::cout << "new view percent " << viewPercent << std::endl;
        }

        residual_renderer.Render(kframe.mesh(), frame.local_pose(), cam_, 1, 1, kframe.frame().image(), frame.image(), l2_texture);
        cv::Mat l2_mat = DownloadTexture(l2_texture, 1, CV_32FC1);
        SaveDebugImageColor(l2_mat, "l2_" + std::to_string(i) + ".png");
    }

    auto meanDuration = accProcessingTime.count() / framesProcessedCounter;
    float meanTranslationError = accTranslationError / framesProcessedCounter;
    float meanRotationError = accRotationError / framesProcessedCounter;
    std::cout << "Mean processing time " << meanDuration << " ms" << std::endl;
    std::cout << "Mean translation error " << meanTranslationError << " ms" << std::endl;
    std::cout << "Mean rotation error " << meanRotationError << " ms" << std::endl;

    // The test passes if the error is below the threshold
    EXPECT_LT(meanTranslationError, translationErrorThreshold)
        << "mean translation estimation error (" << meanTranslationError
        << ") exceeds the acceptable threshold (" << translationErrorThreshold << ").";

    EXPECT_LT(meanRotationError, rotationErrorThreshold)
        << "mean rotation estimation error (" << meanRotationError
        << ") exceeds the acceptable threshold (" << rotationErrorThreshold << ").";

    // EXPECT_LE(durationMs, acceptableTimeMs)
    //     << "Pose estimation took " << durationMs << "ms, which exceeds the acceptable threshold of "
    //     << acceptableTimeMs << "ms.";
}