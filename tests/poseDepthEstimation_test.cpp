#include <gtest/gtest.h>
#include "test_framework.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/FrameWindow.h"
#include "common/keyframe.h"
#include "poseEstimator.h"
#include "poseDepthEstimator.h"

TEST_F(RendererTestBase, ComputePoseDepth)
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
    ImageRenderer image_renderer;
    DIDxyRenderer didxy_renderer;
    PidsRenderer pids_renderer;

    NodataReducerCPU nodata_reducer;

    PoseEstimator pose_estimator(w_, h_, false);
    PoseDepthEstimator posedepth_estimator(w_, h_, true);

    FrameWindow frames(w_, h_);
    KeyFrame kframe(w_, h_,
                    CreateFlatMesh<Mesh>(0.5,
                                         1.5,
                                         cam_,
                                         mesh_vo::mesh_width));

    Texture<ImageType> image_tex_tmp(w_, h_, 0);
    Texture<float> depth_tex_tmp(w_, h_, 0);
    Texture<Vec3f> pids_tex_tmp(w_, h_, Vec3f(0.0, 0.0, 0.0));

    cv::Mat image_mat;
    cv::Mat ref_mat;
    cv::Mat l2_mat;
    cv::Mat depth_mat;

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

            // Mesh mesh = CreateFlatMesh<Mesh>(gt_depth_mean * 0.5,
            //                                  gt_depth_mean * 1.5,
            //                                  cam_,
            //                                  mesh_vo::mesh_width);

            Mesh mesh = CreateFlatMesh<Mesh>(0.5,
                                             1.5,
                                             cam_,
                                             mesh_vo::mesh_width);

            kframe.mesh() = std::move(mesh);
            kframe.id() = img_id;
            // kframe.global_pose() = gt_global_pose;
            kframe.global_pose() = SE3f();
            kframe.global_scale() = 1.0;

            float md = mean_depth(kframe.mesh());
            kframe.scale_mesh(md / mesh_vo::mapping_mean_depth);

            // pose_estimator.init_global_pose(gt_global_pose);
            pose_estimator.init_global_pose(kframe.global_pose());

            continue;
        }

        Frame &frame = frames.latest();
        UploadMatToTexture(frame.image(), 0, image_cv);
        for (int lvl = 0; lvl < frame.didxy().levels(); lvl++)
            didxy_renderer.Render(screen_mesh, lvl, lvl, frame.image(), frame.didxy());

        frame.id() = img_id;
        // frame.global_pose() = gt_global_pose;
        pose_estimator.guess(frame, kframe);

        int plot_lvl = 0;

        ///////////// Debug //////////////////
        image_renderer.Render(kframe.mesh(),
                              kframe.global_pose_to_local(frame.global_pose()),
                              frame.local_exposure(),
                              cam_,
                              plot_lvl, plot_lvl,
                              kframe.image(),
                              image_tex_tmp);
        // residual_renderer.Render(kframe->mesh(), SE3f(), cam_, plot_lvl, plot_lvl, kframe->frame().image(), oframes[k].image(), l2_cpu);
        image_mat = DownloadTextureToMat(image_tex_tmp, plot_lvl);
        ref_mat = DownloadTextureToMat(frame.image(), plot_lvl);
        l2_mat = ref_mat - image_mat;
        SaveDebugImage(l2_mat, "loc_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + "_l2_ini.png");
        //////////////////////////////////////

        // auto startTime = std::chrono::high_resolution_clock::now();
        pose_estimator.estimate(frame, kframe, cam_);
        // auto endTime = std::chrono::high_resolution_clock::now();

        ///////////// Debug //////////////////
        image_renderer.Render(kframe.mesh(),
                              kframe.global_pose_to_local(frame.global_pose()),
                              frame.local_exposure(),
                              cam_,
                              plot_lvl, plot_lvl,
                              kframe.image(),
                              image_tex_tmp);

        // residual_renderer.Render(kframe->mesh(), SE3f(), cam_, plot_lvl, plot_lvl, kframe->frame().image(), oframes[k].image(), l2_cpu);
        image_mat = DownloadTextureToMat(image_tex_tmp, plot_lvl);
        ref_mat = DownloadTextureToMat(frame.image(), plot_lvl);
        l2_mat = ref_mat - image_mat;
        SaveDebugImage(l2_mat, "loc_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + "_l2_opt.png");
        //////////////////////////////////////

        // image_renderer.Render(kframe.mesh(),
        //                       kframe.global_pose_to_local(frame.global_pose()),
        //                       frame.local_exposure(),
        //                       cam_,
        //                       1, 1,
        //                       kframe.image(), image_tex_tmp);

        // Error nodata;
        // nodata_reducer.reduce(1, image_tex_tmp, nodata);
        // float pnodata = nodata.getError() / (image_tex_tmp.width(1) * image_tex_tmp.height(1));
        // float viewPercent = 1.0 - pnodata;

        // std::cout << "view percent " << viewPercent << std::endl;

        float minViewAngle = M_PI;
        for (Frame *f : frames.window_span_mut_all())
        {
            float viewAngle = kframe.meanViewAngle(frame.global_pose(), f->global_pose(), cam_);
            if (viewAngle < minViewAngle)
                minViewAngle = viewAngle;
        }

        if (minViewAngle < mesh_vo::last_min_angle && kframe.id() != 0)
            continue;

        frames.accept_latest();

        if (!frames.full())
        {
            continue;
        }

        // if (viewPercent > mesh_vo::min_view_perc && kframe.id() != 0) // || keyframeViewAngle > mesh_vo::key_max_angle)
        //     continue;

        Frame &new_kf = frames.get_keyframe();
        std::span<Frame *const> frame_span = frames.window_span_mut_no_kf();

        depth_renderer.Render(kframe.mesh(),
                              kframe.global_pose_to_local(new_kf.global_pose()),
                              cam_,
                              plot_lvl,
                              depth_tex_tmp);

        posedepth_estimator.update_keyframe(new_kf, depth_tex_tmp, kframe, cam_);

        /////////////////// Debug /////////////////

        depth_renderer.Render(kframe.mesh(),
                              SE3f(),
                              cam_,
                              plot_lvl,
                              depth_tex_tmp);

        depth_mat = DownloadTextureToMat(depth_tex_tmp, plot_lvl);
        SaveDebugImage(depth_mat, "depth_" + std::to_string(kframe.id()) + "_ini.png");

        for (auto &f : frame_span)
        {
            image_renderer.Render(kframe.mesh(),
                                  kframe.global_pose_to_local(f->global_pose()),
                                  f->local_exposure(),
                                  cam_,
                                  plot_lvl, plot_lvl,
                                  kframe.image(),
                                  image_tex_tmp);
            // residual_renderer.Render(kframe->mesh(), SE3f(), cam_, plot_lvl, plot_lvl, kframe->frame().image(), oframes[k].image(), l2_cpu);
            cv::Mat image_mat = DownloadTextureToMat(image_tex_tmp, plot_lvl);
            cv::Mat ref_mat = DownloadTextureToMat(f->image(), plot_lvl);
            cv::Mat l2_mat = ref_mat - image_mat;
            SaveDebugImage(l2_mat, "l2_" + std::to_string(kframe.id()) + "_" + std::to_string(f->id()) + "_ini.png");
        }
        ////////////////////

        auto startTime = std::chrono::high_resolution_clock::now();
        posedepth_estimator.estimate(frame_span, kframe, cam_);
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        auto duration = processingTime.count();
        std::cout << "processing time " << duration << " ms" << std::endl;

        // frames[kframeIndex].local_pose() = SE3f();

        ////// Debug //////////////
        for (auto &f : frame_span)
        {
            image_renderer.Render(kframe.mesh(),
                                  kframe.global_pose_to_local(f->global_pose()),
                                  f->local_exposure(),
                                  cam_,
                                  plot_lvl, plot_lvl,
                                  kframe.image(),
                                  image_tex_tmp);
            // residual_renderer.Render(kframe->mesh(), SE3f(), cam_, plot_lvl, plot_lvl, kframe->frame().image(), oframes[k].image(), l2_cpu);
            cv::Mat image_mat = DownloadTextureToMat(image_tex_tmp, plot_lvl);
            cv::Mat ref_mat = DownloadTextureToMat(f->image(), plot_lvl);
            cv::Mat l2_mat = ref_mat - image_mat;
            SaveDebugImage(l2_mat, "l2_" + std::to_string(kframe.id()) + "_" + std::to_string(f->id()) + "_opt.png");
        }

        depth_renderer.Render(kframe.mesh(),
                              SE3f(),
                              cam_,
                              plot_lvl,
                              depth_tex_tmp);

        depth_mat = DownloadTextureToMat(depth_tex_tmp, plot_lvl);
        SaveDebugImage(depth_mat, "depth_" + std::to_string(kframe.id()) + "_opt.png");

        double error = 1.0; // RMSE(depth_cpu, estimated_depth_cpu, plot_lvl);

        pids_renderer.Render(kframe.mesh(),
                             SE3f(),
                             cam_, plot_lvl, pids_tex_tmp);

        cv::Mat pids_cv = DownloadTextureToMat(pids_tex_tmp, plot_lvl);
        SaveDebugImage(pids_cv, "pids_" + std::to_string(kframe.id()) + ".png");

        accProcessingTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        accError += error;
        framesProcessedCounter++;
        //////////////////////////
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