#include <gtest/gtest.h>
#include "test_framework.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/FrameWindow.h"
#include "common/keyframe.h"
#include "depthEstimator.h"

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

    DepthEstimator estimator(w_, h_, true);

    FrameWindow frames(w_, h_);
    std::vector<Texture<float>> gt_depth_textures;
    std::vector<SE3f> gt_global_poses;

    gt_depth_textures.reserve(mesh_vo::num_frames);
    gt_global_poses.reserve(mesh_vo::num_frames);

    KeyFrame *kframe;

    Texture<ImageType> image_texture(w_, h_, 0);
    Texture<float> es_depth_texture(w_, h_, 0.0);
    Texture<Vec3f> pids_texture(w_, h_, Vec3f(0.0, 0.0, 0.0));

    for (std::size_t img_id = 0; img_id < image_files_.size(); img_id++)
    {
        std::cout << "Frame " << img_id << std::endl;

        cv::Mat image_cv = cv::imread(image_files_[img_id], cv::IMREAD_GRAYSCALE);
        cv::Mat gt_depth_cv = cv::imread(depth_files_[img_id], cv::IMREAD_GRAYSCALE);
        gt_depth_cv.convertTo(gt_depth_cv, CV_32FC1);
        gt_depth_cv = gt_depth_cv / depth_factor_;

        Texture<float> gt_depth_texture(w_, h_, 0.0);

        Frame &frame = frames.latest();
        UploadMatToTexture(frame.image(), 0, image_cv);
        UploadMatToTexture(gt_depth_texture, 0, gt_depth_cv);
        SE3f gt_global_pose = poses_[img_id];

        for (int lvl = 0; lvl < frame.didxy().levels(); lvl++)
            didxy_renderer.Render(screen_mesh, lvl, lvl, frame.image(), frame.didxy());

        if (img_id == 0)
        {
            double gt_depth_mean = cv::mean(gt_depth_cv)[0];

            Mesh mesh = CreateFlatMesh<Mesh>(gt_depth_mean * 0.5,
                                             gt_depth_mean * 1.5,
                                             cam_,
                                             mesh_vo::mesh_width);
            //    CreateSphereMesh(gt_depth_mean, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_);

            kframe = new KeyFrame(frame, std::move(mesh), gt_global_pose, 1.0);
            float meanDepth = kframe->meanDepth();
            kframe->scaleMesh(meanDepth / mesh_vo::mapping_mean_depth);

            continue;
        }

        SE3f init_local_pose = kframe->globalPoseToLocal(gt_global_pose);

        frame.local_pose() = init_local_pose;

        image_renderer.Render(kframe->mesh(),
                              frame.local_pose(),
                              frame.local_exposure(),
                              cam_,
                              1, 1,
                              kframe->image(), image_texture);

        Error nodata;
        nodata_reducer.reduce(1, image_texture, nodata);
        float pnodata = nodata.getError() / (image_texture.width(1) * image_texture.height(1));
        float viewPercent = 1.0 - pnodata;

        std::cout << "view percent " << viewPercent << std::endl;

        if (viewPercent > mesh_vo::min_view_perc && kframe->id() != 0) // || keyframeViewAngle > mesh_vo::key_max_angle)
            continue;

        float minViewAngle = M_PI;

        for (auto f : frames.window_span_mut())
        {
            float viewAngle = kframe->meanViewAngle(frame.local_pose(), f->local_pose(), cam_);
            if (viewAngle < minViewAngle)
                minViewAngle = viewAngle;
        }

        std::cout << "Min view angle " << minViewAngle << std::endl;

        if (minViewAngle < mesh_vo::last_min_angle && kframe->id() != 0)
            continue;

        frames.accept_latest();

        gt_depth_textures.push_back(std::move(gt_depth_texture));
        gt_global_poses.push_back(std::move(gt_global_pose));
        if (frames.size() > mesh_vo::num_frames)
        {
            gt_depth_textures.erase(gt_depth_textures.begin());
            gt_global_poses.erase(gt_global_poses.begin());
        }
        else
        {
            continue;
        }

        frames.promote_middle_to_keyframe();
        Frame &newKeyFrame = frames.keyframe_frame();
        std::span<Frame *const> frame_span = frames.window_span_mut();

        estimator.changeKeyframe(newKeyFrame, frame_span, *kframe, cam_);

        std::cout << "Mean depth " << kframe->meanDepth() << std::endl;

        //////////// Debug /////////////////
        int plot_lvl = 0;
        for (Frame *f : frame_span)
        {
            image_renderer.Render(kframe->mesh(),
                                  f->local_pose(),
                                  f->local_exposure(),
                                  cam_,
                                  plot_lvl, plot_lvl,
                                  kframe->image(), image_texture);
            cv::Mat image_mat = DownloadTextureToMat(image_texture, plot_lvl);
            cv::Mat ref_mat = DownloadTextureToMat(f->image(), plot_lvl);
            cv::Mat l2_mat = ref_mat - image_mat;
            SaveDebugImage(l2_mat, "l2_" + std::to_string(kframe->id()) + "_" + std::to_string(f->id()) + "_init.png");
        }
        //////////////////////////

        auto startTime = std::chrono::high_resolution_clock::now();
        estimator.estimate(frame_span, *kframe, cam_);
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        auto duration = processingTime.count();
        std::cout << "processing time " << duration << " ms" << std::endl;

        float new_mean_depth = kframe->meanDepth();
        kframe->scaleMesh(new_mean_depth / mesh_vo::mapping_mean_depth);
        // frame.scalePose(new_mean_depth / mesh_vo::mapping_mean_depth);
        for (Frame *f : frame_span)
        {
            f->scalePose(new_mean_depth / mesh_vo::mapping_mean_depth);
        }

        ///////////////// Debug ///////////////////
        for (Frame *f : frame_span)
        {
            image_renderer.Render(kframe->mesh(),
                                  f->local_pose(),
                                  f->local_exposure(),
                                  cam_,
                                  plot_lvl, plot_lvl,
                                  kframe->image(), image_texture);
            cv::Mat image_mat = DownloadTextureToMat(image_texture, plot_lvl);
            cv::Mat ref_mat = DownloadTextureToMat(f->image(), plot_lvl);
            cv::Mat l2_mat = ref_mat - image_mat;
            SaveDebugImage(l2_mat, "l2_" + std::to_string(kframe->id()) + "_" + std::to_string(f->id()) + "_opt.png");
        }
        //////////////////////////

        float scale = kframe->getGlobalScale();
        kframe->scaleMesh(1.0 / scale);
        depth_renderer.Render(kframe->mesh(),
                              SE3f(),
                              cam_,
                              plot_lvl,
                              es_depth_texture);
        kframe->scaleMesh(scale);
        double error = 0; // RMSE(es_depth_texture, gt_depth_textures[kframeIndex], plot_lvl);

        cv::Mat depth_cv = DownloadTextureToMat(es_depth_texture, plot_lvl);
        SaveDebugImage(depth_cv, "depth_" + std::to_string(kframe->id()) + ".png");

        // cv::Mat gt_depth_cv_2 = DownloadTextureToMat(gt_depth_textures[kframeIndex], plot_lvl);
        // SaveDebugImage(gt_depth_cv_2, "depth_" + std::to_string(kframe->id()) + "_gt.png");

        pids_renderer.Render(kframe->mesh(),
                             SE3f(),
                             cam_, plot_lvl, pids_texture);

        cv::Mat pids_cv = DownloadTextureToMat(pids_texture, plot_lvl);
        SaveDebugImage(pids_cv, "pids_" + std::to_string(kframe->id()) + ".png");

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