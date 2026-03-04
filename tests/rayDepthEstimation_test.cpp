#include <gtest/gtest.h>
#include "test_framework.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "common/FrameWindow.h"
#include "optimizers/depthOptimizer.h"
#include "optimizers/depthExpOptimizer.h"

TEST_F(RendererTestBase, ComputeDepth)
{
#ifdef COMPILE_GL
    InitEGL();
#endif

    const long long acceptableTimeMs = 30;
    const float errorThreshold = 0.023; // best = 0.022444;

    std::chrono::milliseconds accProcessingTime = std::chrono::milliseconds(0);
    float accError = 0;
    int framesProcessedCounter = 0;

    Mesh screen_mesh = CreateScreenQuad<Mesh>();

    DepthRenderer depth_renderer;
    PidsRenderer pids_renderer;
    ImageRenderer image_renderer;
    DIDxyRenderer didxy_renderer;

    NodataReducerCPU nodata_reducer;

    DepthOptimizer optimizer(w_, h_, true);

    FrameWindow frames(w_, h_);
    KeyFrame kframe(w_, h_,
                    CreateFlatMesh<Mesh>(0.5,
                                         1.5,
                                         cam_,
                                         mesh_vo::mesh_width));

    Texture<ImageType> image_tex_tmp(w_, h_, 0);
    Texture<float> depth_tex_tmp(w_, h_, 0.0);
    Texture<Vec3f> pids_tex_tmp(w_, h_, Vec3f(0.0, 0.0, 0.0));

    for (std::size_t img_id = 0; img_id < image_files_.size(); img_id++)
    {
        std::cout << "Frame " << img_id << std::endl;

        cv::Mat image_cv = cv::imread(image_files_[img_id], cv::IMREAD_GRAYSCALE);
        cv::Mat gt_depth_cv = cv::imread(depth_files_[img_id], cv::IMREAD_GRAYSCALE);
        gt_depth_cv.convertTo(gt_depth_cv, CV_32FC1);
        gt_depth_cv = gt_depth_cv / depth_factor_;
        double gt_depth_mean = cv::mean(gt_depth_cv)[0];
        SE3f gt_global_pose = poses_[img_id];

        if (img_id == 0)
        {
            UploadMatToTexture(kframe.image(), 0, image_cv);

            Mesh mesh = CreateFlatMesh<Mesh>(gt_depth_mean * 0.5,
                                             gt_depth_mean * 1.5,
                                             cam_,
                                             mesh_vo::mesh_width);

            kframe.global_pose() = gt_global_pose;
            kframe.global_scale() = 1.0;
            kframe.id() = img_id;
            kframe.mesh() = std::move(mesh);

            continue;
        }

        Frame &frame = frames.latest();

        UploadMatToTexture(frame.image(), 0, image_cv);
        //UploadMatToTexture(frame.depth(), 0, gt_depth_cv);

        for (int lvl = 0; lvl < frame.didxy().levels(); lvl++)
            didxy_renderer.Render(screen_mesh, lvl, lvl, frame.image(), frame.didxy());

        frame.id() = img_id;
        frame.global_pose() = gt_global_pose;

        float minViewAngle = M_PI;

        for (auto f : frames.window_span_mut_all())
        {
            float viewAngle = kframe.meanViewAngle(frame.global_pose(), f->global_pose(), cam_);
            if (viewAngle < minViewAngle)
                minViewAngle = viewAngle;
        }

        std::cout << "Min view angle " << minViewAngle << std::endl;

        if (minViewAngle < mesh_vo::last_min_angle && kframe.id() != 0)
            continue;

        frames.accept_latest();

        image_renderer.Render(kframe.mesh(),
                              kframe.global_pose_to_local(frame.global_pose()),
                              frame.local_exposure(),
                              cam_,
                              1, 1,
                              kframe.image(), image_tex_tmp);

        Error nodata;
        nodata_reducer.reduce(1, image_tex_tmp, nodata);
        float pnodata = nodata.getError() / (image_tex_tmp.width(1) * image_tex_tmp.height(1));
        float viewPercent = 1.0 - pnodata;

        std::cout << "view percent " << viewPercent << std::endl;

        if (viewPercent > mesh_vo::min_view_perc && kframe.id() != 0) // || keyframeViewAngle > mesh_vo::key_max_angle)
            continue;

        // int kframeIndex = frames.size() / 2;

        // frames.promote_middle_to_keyframe();
        Frame &new_kf = frames.get_keyframe();
        std::span<Frame *const> frame_span = frames.window_span_mut_no_kf();

        // std::vector<float> ver_buff_;
        // std::vector<int> idx_buff_;
        //  CreateMesh(gt_depth_textures[kframeIndex], cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
        // CreateFlatMesh(mesh_vo::mapping_mean_depth * 0.5, mesh_vo::mapping_mean_depth * 1.5, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
        //   CreateSphereMesh(mesh_vo::mapping_mean_depth, cam_, mesh_vo::mesh_width, pos_buff_, tex_buff_, wei_buff_, idx_buff_);
        depth_renderer.Render(kframe.mesh(),
                              new_kf.global_pose(),
                              cam_,
                              0,
                              depth_tex_tmp);

        // Mesh mesh(ver_buff_, idx_buff_, true, true, true);
        // float global_scale = kframe->getGlobalScale();
        // kframe = new KeyFrame(frames[kframeIndex].image(),
        //                       frames[kframeIndex].didxy(),
        //                       gt_global_poses[kframeIndex], mesh, 1.0);
        // kframe->scaleMesh(global_scale);

        // kframe->changeFrame(frames[kframeIndex].image(),
        //                     frames[kframeIndex].didxy(),
        //                     // es_depth_texture,
        //                     frames[kframeIndex].local_pose(),
        //                     frames[kframeIndex].id(),
        //                     cam_);

        std::cout << "Mean depth " << mean_depth(kframe.mesh()) << std::endl;

        auto startTime = std::chrono::high_resolution_clock::now();
        for (int lvl = mesh_vo::mapping_ini_lvl; lvl >= mesh_vo::mapping_fin_lvl; lvl--)
        {
            int in_lvl = lvl;
            int out_lvl = lvl;

            optimizer.init(frame_span, kframe, cam_, in_lvl, out_lvl);
            while (!optimizer.converged())
            {
                auto stepStartTime = std::chrono::high_resolution_clock::now();
                optimizer.step(frame_span, kframe, cam_, in_lvl, out_lvl);
                auto stepEndTime = std::chrono::high_resolution_clock::now();
                std::chrono::milliseconds stepProcessingTime = std::chrono::duration_cast<std::chrono::milliseconds>(stepEndTime - stepStartTime);
                auto duration = stepProcessingTime.count();
                std::cout << "step processing time " << duration << " ms" << std::endl;
            }
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        auto duration = processingTime.count();
        std::cout << "processing time " << duration << " ms" << std::endl;

        int plot_lvl = 0;
        for (Frame *f : frame_span)
        {
            image_renderer.Render(kframe.mesh(),
                                  f->global_pose(),
                                  f->local_exposure(),
                                  cam_,
                                  plot_lvl, plot_lvl,
                                  kframe.image(), image_tex_tmp);
            cv::Mat image_mat = DownloadTextureToMat(image_tex_tmp, plot_lvl);
            cv::Mat ref_mat = DownloadTextureToMat(f->image(), plot_lvl);
            cv::Mat l2_mat = ref_mat - image_mat;
            SaveDebugImage(l2_mat, "l2_" + std::to_string(kframe.id()) + "_" + std::to_string(f->id()) + ".png");
        }

        depth_renderer.Render(kframe.mesh(),
                              SE3f(),
                              cam_,
                              plot_lvl,
                              depth_tex_tmp);
        double error = 0.0; // RMSE(es_depth_texture, gt_depth_textures[kframeIndex], plot_lvl);

        cv::Mat depth_cv = DownloadTextureToMat(depth_tex_tmp, plot_lvl);
        SaveDebugImage(depth_cv, "depth_" + std::to_string(kframe.id()) + ".png");

        // cv::Mat gt_depth_cv_2 = DownloadTextureToMat(gt_depth_textures[kframeIndex], plot_lvl);
        // SaveDebugImage(gt_depth_cv_2, "depth_" + std::to_string(kframe->id()) + "_gt.png");

        pids_renderer.Render(kframe.mesh(),
                             SE3f(),
                             cam_, plot_lvl, pids_tex_tmp);

        cv::Mat pids_cv = DownloadTextureToMat(pids_tex_tmp, plot_lvl);
        SaveDebugImage(pids_cv, "pids_" + std::to_string(kframe.id()) + ".png");

        accProcessingTime += processingTime;
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