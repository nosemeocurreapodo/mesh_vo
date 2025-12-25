#include <gtest/gtest.h>
#include "tests/common/test_framework.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "optimizers/poseOptimizer.h"
#include "optimizers/poseExpOptimizer.h"
#include "optimizers/poseVelExpOptimizer.h"
#include "optimizers/poseMapOptimizer.h"
#include "optimizers/poseExpMapOptimizer.h"
#include "optimizers/poseVelMapOptimizer.h"
#include "optimizers/poseVelExpMapOptimizer.h"

TEST_F(RendererTestBase, ComputePoseMap)
{
#ifdef COMPILE_GL
    InitEGL();
#endif

    const long long acceptableTimeMs = 30;
    const float errorThreshold = 0.023; // best = 0.022444;

    std::chrono::milliseconds accProcessingTime = std::chrono::milliseconds(0);
    float accError = 0;
    int framesProcessedCounter = 0;

    std::vector<float> s_ver_buff_;
    std::vector<int> s_idx_buff_;
    CreateScreenQuad(s_ver_buff_, s_idx_buff_);
    Mesh screen_mesh(s_ver_buff_, s_idx_buff_, true, true, false);

    DepthRenderer depth_renderer;
    ImageRenderer image_renderer;
    DIDxyRenderer didxy_renderer;
    ResidualRenderer residual_renderer;

    NodataReducerCPU nodata_reducer;

    PoseExpOptimizer pose_optimizer(w_, h_, true);
    PoseExpMapOptimizer posemap_optimizer(w_, h_, true);

    std::vector<Frame> frames;
    KeyFrame *kframe;

    Texture<ImageType> image_cpu(w_, h_, 0);
    Texture<float> depth_cpu(w_, h_, 0);
    Texture<float> estimated_depth_cpu(w_, h_, 0);
    Texture<Vec3f> didxy_cpu(w_, h_, Vec3f(0.0, 0.0, 0.0));
    Texture<float> l2_cpu(w_, h_, 0.0);

    cv::Mat image_cv = cv::imread(image_files_[0], cv::IMREAD_GRAYSCALE);
    cv::Mat depth_cv = cv::imread(depth_files_[0], cv::IMREAD_GRAYSCALE);
    depth_cv.convertTo(depth_cv, CV_32FC1);
    depth_cv = depth_cv / depth_factor_;
    cv::Mat depth_mask;
    cv::Scalar depth_mean;
    UploadMatToTexture(image_cpu, 0, image_cv);
    UploadMatToTexture(depth_cpu, 0, depth_cv);
    SE3f gt_pose = poses_[0];

    // This is the scale of the map and the movements, such that when normalied the map (and the movements) the mean depth is around 1.0
    float scale = 1.0;
    if (depth_files_.size() == image_files_.size())
    {
        depth_mask = (depth_cv > 0.0);
        scale = 1.0; // cv::mean(depth_cv, depth_mask)[0];
        depth_cv = depth_cv * mesh_vo::mapping_mean_depth / scale;
        // UploadMatToTexture(depth_cpu, 0, depth_cv);
    }
    else
    {
        scale = depth_factor_ / mesh_vo::mapping_mean_depth;
    }

    std::vector<float> ver_buff_;
    std::vector<int> idx_buff_;

    //CreateMesh(depth_cv, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
    CreateFlatMesh(mesh_vo::mapping_mean_depth * 0.5, mesh_vo::mapping_mean_depth * 1.5, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
    //   CreateSphereMesh(mesh_vo::mapping_mean_depth, cam_, mesh_vo::mesh_width, pos_buff_, tex_buff_, wei_buff_, idx_buff_);

    for (int lvl = 0; lvl < didxy_cpu.levels(); lvl++)
        didxy_renderer.Render(screen_mesh, lvl, lvl, image_cpu, didxy_cpu);

    Mesh mesh(ver_buff_, idx_buff_, true, true, true);
    kframe = new KeyFrame(Frame(image_cpu, didxy_cpu, 0, SE3f(), gt_pose), mesh, scale);
    SE3f tracked_global_pose = gt_pose;

    depth_renderer.Render(kframe->mesh(),
                          SE3f(),
                          cam_,
                          1,
                          depth_cpu);
    depth_cv = DownloadTextureToMat(depth_cpu, 1);
    depth_mask = (depth_cv > 0.0);
    depth_mean = cv::mean(depth_cv, depth_mask);
    std::cout << "Init Depth mean " << depth_mean[0] << std::endl;

    for (std::size_t img_id = 1; img_id < image_files_.size(); img_id++)
    {
        std::cout << "Frame " << img_id << std::endl;

        cv::Mat image_cv = cv::imread(image_files_[img_id], cv::IMREAD_GRAYSCALE);
        cv::Mat depth_cv = cv::imread(depth_files_[img_id], cv::IMREAD_GRAYSCALE);
        depth_cv.convertTo(depth_cv, CV_32FC1);
        depth_cv = depth_cv / depth_factor_;
        UploadMatToTexture(image_cpu, 0, image_cv);
        UploadMatToTexture(depth_cpu, 0, depth_cv);
        gt_pose = poses_[img_id];

        for (int lvl = 0; lvl < didxy_cpu.levels(); lvl++)
            didxy_renderer.Render(screen_mesh, lvl, lvl, image_cpu, didxy_cpu);

        // didxy_renderer.Render(screen_mesh, 0, 0, image_cpu, didxy_cpu);
        // didxy_cpu.generate_mipmaps(0);

        SE3f init_global_pose = tracked_global_pose;
        SE3f init_local_pose = kframe->globalPoseToLocal(init_global_pose);

        Frame frame(image_cpu, didxy_cpu, img_id, init_local_pose, init_global_pose);

        //        auto startTime = std::chrono::high_resolution_clock::now();
        for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_fin_lvl; lvl--)
        {
            int in_lvl = lvl;
            int out_lvl = lvl;
            pose_optimizer.init(frame, *kframe, cam_, in_lvl, out_lvl);
            while (!pose_optimizer.converged())
            {
                pose_optimizer.step(frame, *kframe, cam_, in_lvl, out_lvl);
            }
        }

        SE3f new_global_pose = kframe->localPoseToGlobal(frame.local_pose());
        SE3f new_local_pose = frame.local_pose();
        Vec6f new_local_vel = frame.local_vel();
        Vec2f new_local_exposure = frame.local_exposure();

        frame.global_pose() = new_global_pose;
        frame.local_pose() = new_local_pose;
        frame.local_vel() = new_local_vel;
        frame.local_exposure() = new_local_exposure;

        // tracked_global_movement = new_global_pose * tracked_global_pose.inverse();
        tracked_global_pose = new_global_pose;
        // tracked_global_pose = gt_pose;
        // tracked_local_exposure = new_local_exposure;

        //        auto endTime = std::chrono::high_resolution_clock::now();

        float minViewAngle = M_PI;
        for (std::size_t j = 0; j < frames.size(); j++)
        {
            float viewAngle = kframe->meanViewAngle(frame.local_pose(), frames[j].local_pose());
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

        // CreateMesh(depth_cv, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
        CreateFlatMesh(mesh_vo::mapping_mean_depth * 0.5, mesh_vo::mapping_mean_depth * 1.5, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
        //   CreateSphereMesh(mesh_vo::mapping_mean_depth, cam_, mesh_vo::mesh_width, pos_buff_, tex_buff_, wei_buff_, idx_buff_);

        Mesh mesh(ver_buff_, idx_buff_, true, true, true);
        kframe = new KeyFrame(frames[kframeIndex], mesh, scale);

        frame.local_pose() = kframe->globalPoseToLocal(frame.global_pose());

        for (std::size_t k = 0; k < frames.size(); k++)
        {
            frames[k].local_pose() = kframe->globalPoseToLocal(frames[k].global_pose());
        }

        for (std::size_t k = 0; k < oframes.size(); k++)
        {
            oframes[k].local_pose() = kframe->globalPoseToLocal(oframes[k].global_pose());
        }

        auto startTime = std::chrono::high_resolution_clock::now();

        for (int lvl = mesh_vo::mapping_ini_lvl; lvl >= mesh_vo::mapping_fin_lvl; lvl--)
        {
            int in_lvl = lvl;
            int out_lvl = lvl;

            posemap_optimizer.init(oframes, *kframe, cam_, in_lvl, out_lvl);
            while (!posemap_optimizer.converged())
            {
                posemap_optimizer.step(oframes, *kframe, cam_, in_lvl, out_lvl);
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();

        int plot_lvl = 1;
        for (std::size_t k = 0; k < oframes.size(); k++)
        {
            residual_renderer.Render(kframe->mesh(),
                                     oframes[k].local_pose(),
                                     oframes[k].local_exposure(),
                                     cam_,
                                     plot_lvl, plot_lvl,
                                     kframe->frame().image(),
                                     oframes[k].image(),
                                     l2_cpu);
            // residual_renderer.Render(kframe->mesh(), SE3f(), cam_, plot_lvl, plot_lvl, kframe->frame().image(), oframes[k].image(), l2_cpu);
            cv::Mat l2_mat = DownloadTextureToMat(l2_cpu, plot_lvl);
            SaveDebugImage(l2_mat, "l2_" + std::to_string(img_id) + "_" + std::to_string(k) + ".png");
        }

        depth_renderer.Render(kframe->mesh(),
                              SE3f(),
                              cam_,
                              plot_lvl,
                              estimated_depth_cpu);

        double error = RMSE(depth_cpu, estimated_depth_cpu, plot_lvl);
        
        depth_cv = DownloadTextureToMat(estimated_depth_cpu, plot_lvl);
        SaveDebugImage(depth_cv, "depth_" + std::to_string(img_id) + ".png");
        depth_mask = (depth_cv > 0.0);
        depth_mean = cv::mean(depth_cv, depth_mask);
        std::cout << "Depth mean " << depth_mean[0] << std::endl;
        // scale = scale * depth_mean[0];

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