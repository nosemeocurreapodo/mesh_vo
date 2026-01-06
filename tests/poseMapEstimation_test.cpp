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

    std::vector<float> s_ver_buff;
    std::vector<int> s_idx_buff;
    CreateScreenQuad(s_ver_buff, s_idx_buff);
    Mesh screen_mesh(s_ver_buff, s_idx_buff, true, true, false);

    DepthRenderer depth_renderer;
    ImageRenderer image_renderer;
    DIDxyRenderer didxy_renderer;
    ResidualRenderer residual_renderer;

    NodataReducerCPU nodata_reducer;

    PoseExpOptimizer pose_optimizer(w_, h_, true);
    PoseExpMapOptimizer posemap_optimizer(w_, h_, true);

    std::vector<Frame> frames;
    KeyFrame *kframe;

    Texture<ImageType> image(w_, h_, 0);
    Texture<float> gt_depth(w_, h_, 0);
    Texture<float> es_depth(w_, h_, 0);
    Texture<Vec3f> didxy(w_, h_, Vec3f(0.0, 0.0, 0.0));
    Texture<float> l2(w_, h_, 0.0);

    SE3f tracked_local_pose;
    SE3f tracked_local_movement;
    Vec2f tracked_local_exposure(0.0, 0.0);

    bool initial_keyframe = true;

    for (std::size_t img_id = 0; img_id < image_files_.size(); img_id++)
    {
        std::cout << "Frame " << img_id << std::endl;

        cv::Mat image_cv = cv::imread(image_files_[img_id], cv::IMREAD_GRAYSCALE);
        cv::Mat gt_depth_cv = cv::imread(depth_files_[img_id], cv::IMREAD_GRAYSCALE);
        gt_depth_cv.convertTo(gt_depth_cv, CV_32FC1);
        gt_depth_cv = gt_depth_cv / depth_factor_;
        UploadMatToTexture(image, 0, image_cv);
        UploadMatToTexture(gt_depth, 0, gt_depth_cv);
        SE3f gt_global_pose = poses_[img_id];

        for (int lvl = 0; lvl < didxy.levels(); lvl++)
            didxy_renderer.Render(screen_mesh, lvl, lvl, image, didxy);

        // didxy_renderer.Render(screen_mesh, 0, 0, image_cpu, didxy_cpu);
        // didxy_cpu.generate_mipmaps(0);

        if (img_id == 0)
        {
            std::vector<float> ver_buff;
            std::vector<int> idx_buff;

            double gt_depth_mean = cv::mean(gt_depth_cv)[0];

            // CreateMesh(gt_depth, cam_, mesh_vo::mesh_width, ver_buff, idx_buff, true, true, true);
            CreateFlatMesh(gt_depth_mean * 0.5, gt_depth_mean * 1.5, cam_, mesh_vo::mesh_width, ver_buff, idx_buff, true, true, true);
            //     CreateSphereMesh(mesh_vo::mapping_mean_depth, cam_, mesh_vo::mesh_width, pos_buff_, tex_buff_, wei_buff_, idx_buff_);

            Mesh mesh(ver_buff, idx_buff, true, true, true);
            kframe = new KeyFrame(image, didxy, gt_global_pose, mesh, 1.0);

            float meanDepth = kframe->meanDepth();
            kframe->scaleMesh(meanDepth / mesh_vo::mapping_mean_depth);

            continue;
        }

        SE3f init_local_pose = tracked_local_movement * tracked_local_pose;
        Vec2f init_local_exposure = tracked_local_exposure;

        Frame frame(image, didxy, img_id, init_local_pose, init_local_exposure);

        // auto startTime = std::chrono::high_resolution_clock::now();
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
        // auto endTime = std::chrono::high_resolution_clock::now();

        SE3f new_local_pose = frame.local_pose();
        Vec2f new_local_exposure = frame.local_exposure();

        tracked_local_movement = new_local_pose * tracked_local_pose.inverse();
        tracked_local_pose = new_local_pose;
        tracked_local_exposure = new_local_exposure;

        float minViewAngle = M_PI;
        for (std::size_t j = 0; j < frames.size(); j++)
        {
            float viewAngle = kframe->meanViewAngle(frame.local_pose(), frames[j].local_pose());
            if (viewAngle < minViewAngle)
                minViewAngle = viewAngle;
        }

        if (minViewAngle < mesh_vo::last_min_angle && !initial_keyframe)
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

        image_renderer.Render(kframe->mesh(),
                              frame.local_pose(),
                              frame.local_exposure(),
                              cam_,
                              1, 1,
                              kframe->image(), image);

        Error nodata = nodata_reducer.reduce(1, image);
        float pnodata = nodata.getError() / (image.width(1) * image.height(1));
        float viewPercent = 1.0 - pnodata;

        std::cout << "view percent " << viewPercent << std::endl;

        if (viewPercent > mesh_vo::min_view_perc && !initial_keyframe) // || keyframeViewAngle > mesh_vo::key_max_angle)
            continue;

        initial_keyframe = false;

        int kframeIndex = frames.size() / 2;

        // CreateMesh(depth_cv, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
        // CreateFlatMesh(mesh_vo::mapping_mean_depth * 0.5, mesh_vo::mapping_mean_depth * 1.5, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
        //   CreateSphereMesh(mesh_vo::mapping_mean_depth, cam_, mesh_vo::mesh_width, pos_buff_, tex_buff_, wei_buff_, idx_buff_);

        // Mesh mesh(ver_buff_, idx_buff_, true, true, true);
        // kframe = new KeyFrame(frames[kframeIndex], mesh, scale);

        kframe->changeFrame(frames[kframeIndex].image(),
                            frames[kframeIndex].didxy(),
                            frames[kframeIndex].local_pose(),
                            cam_);

        SE3f reference_pose = frames[kframeIndex].local_pose().inverse();

        tracked_local_pose = tracked_local_pose * reference_pose;
        tracked_local_movement = tracked_local_movement * reference_pose;
        tracked_local_exposure = Vec2f(0.0, 0.0);

        frame.local_pose() = frame.local_pose() * reference_pose;

        for (std::size_t k = 0; k < frames.size(); k++)
        {
            frames[k].local_pose() = frames[k].local_pose() * reference_pose;
        }

        std::vector<Frame> oframes = frames;
        oframes.erase(oframes.begin() + kframeIndex);

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

        for (std::size_t k = 0; k < oframes.size(); k++)
        {
            if (frame.id() == oframes[k].id())
            {
                frame.local_pose() = oframes[k].local_pose();
            }

            for (int j = 0; j < frames.size(); j++)
            {
                if (frames[j].id() == oframes[k].id())
                {
                    frames[j].local_pose() = oframes[k].local_pose();
                }
            }
        }

        float meanDepth = kframe->meanDepth();
        kframe->scaleMesh(meanDepth / mesh_vo::mapping_mean_depth);

        tracked_local_pose.translation() /= (meanDepth / mesh_vo::mapping_mean_depth);
        tracked_local_movement.translation() /= (meanDepth / mesh_vo::mapping_mean_depth);

        frame.scalePose(meanDepth / mesh_vo::mapping_mean_depth);

        for (int j = 0; j < frames.size(); j++)
        {
            frames[j].scalePose(meanDepth / mesh_vo::mapping_mean_depth);
        }

        int plot_lvl = 1;
        for (std::size_t k = 0; k < frames.size(); k++)
        {
            residual_renderer.Render(kframe->mesh(),
                                     frames[k].local_pose(),
                                     frames[k].local_exposure(),
                                     cam_,
                                     plot_lvl, plot_lvl,
                                     kframe->image(),
                                     frames[k].image(),
                                     l2);
            // residual_renderer.Render(kframe->mesh(), SE3f(), cam_, plot_lvl, plot_lvl, kframe->frame().image(), oframes[k].image(), l2_cpu);
            cv::Mat l2_mat = DownloadTextureToMat(l2, plot_lvl);
            SaveDebugImage(l2_mat, "l2_" + std::to_string(img_id) + "_" + std::to_string(k) + ".png");
        }

        depth_renderer.Render(kframe->mesh(),
                              SE3f(),
                              cam_,
                              plot_lvl,
                              es_depth);

        double error = 1.0; // RMSE(depth_cpu, estimated_depth_cpu, plot_lvl);

        cv::Mat es_depth_cv = DownloadTextureToMat(es_depth, plot_lvl);
        SaveDebugImage(es_depth_cv, "depth_" + std::to_string(img_id) + ".png");

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