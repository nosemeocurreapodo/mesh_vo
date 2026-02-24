#include <gtest/gtest.h>
#include "test_framework.h"
#include "common/types.h"
#include "common/frame.h"
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

    Mesh screen_mesh;
    CreateScreenQuad(screen_mesh);

    DepthRenderer depth_renderer;
    ImageRenderer image_renderer;
    DIDxyRenderer didxy_renderer;
    PidsRenderer pids_renderer;

    NodataReducerCPU nodata_reducer;

    PoseEstimator pose_estimator(w_, h_, true);
    PoseDepthEstimator posedepth_estimator(w_, h_, true);

    std::vector<Frame> frames;
    KeyFrame *kframe;

    Texture<ImageType> image_texture(w_, h_, 0);
    Texture<float> gt_depth_texture(w_, h_, 0);
    Texture<float> es_depth_texture(w_, h_, 0);
    Texture<Vec3f> didxy_texture(w_, h_, Vec3f(0.0, 0.0, 0.0));
    Texture<Vec3f> pids_texture(w_, h_, Vec3f(0.0, 0.0, 0.0));

    cv::Mat image_mat;
    cv::Mat ref_mat;
    cv::Mat l2_mat;

    for (std::size_t img_id = 0; img_id < image_files_.size(); img_id++)
    {
        std::cout << "Frame " << img_id << std::endl;

        cv::Mat image_cv = cv::imread(image_files_[img_id], cv::IMREAD_GRAYSCALE);
        cv::Mat gt_depth_cv = cv::imread(depth_files_[img_id], cv::IMREAD_GRAYSCALE);
        gt_depth_cv.convertTo(gt_depth_cv, CV_32FC1);
        gt_depth_cv = gt_depth_cv / depth_factor_;
        UploadMatToTexture(image_texture, 0, image_cv);
        UploadMatToTexture(gt_depth_texture, 0, gt_depth_cv);
        SE3f gt_global_pose = poses_[img_id];

        for (int lvl = 0; lvl < didxy_texture.levels(); lvl++)
            didxy_renderer.Render(screen_mesh, lvl, lvl, image_texture, didxy_texture);

        // didxy_renderer.Render(screen_mesh, 0, 0, image_cpu, didxy_cpu);
        // didxy_cpu.generate_mipmaps(0);

        if (img_id == 0)
        {
            double gt_depth_mean = cv::mean(gt_depth_cv)[0];

            Mesh mesh;
            // CreateMesh(gt_depth, cam_, mesh_vo::mesh_width, ver_buff, idx_buff, true, true, true);
            CreateFlatMesh(gt_depth_mean * 0.5,
                           gt_depth_mean * 1.5,
                           cam_,
                           mesh_vo::mesh_width,
                           mesh);
            //     CreateSphereMesh(mesh_vo::mapping_mean_depth, cam_, mesh_vo::mesh_width, pos_buff_, tex_buff_, wei_buff_, idx_buff_);

            kframe = new KeyFrame(image_texture, didxy_texture, gt_global_pose, mesh, 1.0, 0);

            float meanDepth = kframe->meanDepth();
            kframe->scaleMesh(meanDepth / mesh_vo::mapping_mean_depth);

            continue;
        }

        Frame frame(image_texture, didxy_texture, img_id, kframe->id());

        pose_estimator.guess(frame);

        ///////////// Debug //////////////////
        int plot_lvl = 1;
        image_renderer.Render(kframe->mesh(),
                              frame.local_pose(),
                              frame.local_exposure(),
                              cam_,
                              plot_lvl, plot_lvl,
                              kframe->image(),
                              image_texture);
        // residual_renderer.Render(kframe->mesh(), SE3f(), cam_, plot_lvl, plot_lvl, kframe->frame().image(), oframes[k].image(), l2_cpu);
        image_mat = DownloadTextureToMat(image_texture, plot_lvl);
        ref_mat = DownloadTextureToMat(frame.image(), plot_lvl);
        l2_mat = ref_mat - image_mat;
        SaveDebugImage(l2_mat, "loc_" + std::to_string(kframe->id()) + "_" + std::to_string(frame.id()) + "_l2_ini.png");
        //////////////////////////////////////

        // auto startTime = std::chrono::high_resolution_clock::now();
        pose_estimator.estimate(frame, *kframe, cam_);
        // auto endTime = std::chrono::high_resolution_clock::now();

        ///////////// Debug //////////////////
        image_renderer.Render(kframe->mesh(),
                              frame.local_pose(),
                              frame.local_exposure(),
                              cam_,
                              plot_lvl, plot_lvl,
                              kframe->image(),
                              image_texture);
        // residual_renderer.Render(kframe->mesh(), SE3f(), cam_, plot_lvl, plot_lvl, kframe->frame().image(), oframes[k].image(), l2_cpu);
        image_mat = DownloadTextureToMat(image_texture, plot_lvl);
        ref_mat = DownloadTextureToMat(frame.image(), plot_lvl);
        l2_mat = ref_mat - image_mat;
        SaveDebugImage(l2_mat, "loc_" + std::to_string(kframe->id()) + "_" + std::to_string(frame.id()) + "_l2_opt.png");
        //////////////////////////////////////

        float minViewAngle = M_PI;
        for (std::size_t j = 0; j < frames.size(); j++)
        {
            float viewAngle = kframe->meanViewAngle(frame.local_pose(), frames[j].local_pose(), cam_);
            if (viewAngle < minViewAngle)
                minViewAngle = viewAngle;
        }

        if (minViewAngle < mesh_vo::last_min_angle && kframe->id() != 0)
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
                              kframe->image(), image_texture);

        Error nodata;
        nodata_reducer.reduce(1, image_texture, nodata);
        float pnodata = nodata.getError() / (image_texture.width(1) * image_texture.height(1));
        float viewPercent = 1.0 - pnodata;

        std::cout << "view percent " << viewPercent << std::endl;

        if (viewPercent > mesh_vo::min_view_perc && kframe->id() != 0) // || keyframeViewAngle > mesh_vo::key_max_angle)
            continue;

        int kframeIndex = frames.size() / 2;
        Frame newkframe = frames[kframeIndex];

        posedepth_estimator.changeKeyframe(newkframe, frames, *kframe, cam_);

        /////////////////// Debug /////////////////
        for (std::size_t k = 0; k < frames.size(); k++)
        {
            image_renderer.Render(kframe->mesh(),
                                  frames[k].local_pose(),
                                  frames[k].local_exposure(),
                                  cam_,
                                  plot_lvl, plot_lvl,
                                  kframe->image(),
                                  image_texture);
            // residual_renderer.Render(kframe->mesh(), SE3f(), cam_, plot_lvl, plot_lvl, kframe->frame().image(), oframes[k].image(), l2_cpu);
            cv::Mat image_mat = DownloadTextureToMat(image_texture, plot_lvl);
            cv::Mat ref_mat = DownloadTextureToMat(frames[k].image(), plot_lvl);
            cv::Mat l2_mat = ref_mat - image_mat;
            SaveDebugImage(l2_mat, "map_" + std::to_string(kframe->id()) + "_" + std::to_string(frames[k].id()) + "_l2_ini.png");
        }
        ////////////////////

        std::vector<Frame> oframes = frames;
        oframes.erase(oframes.begin() + kframeIndex);

        auto startTime = std::chrono::high_resolution_clock::now();
        posedepth_estimator.estimate(oframes, *kframe, cam_);
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        auto duration = processingTime.count();
        std::cout << "processing time " << duration << " ms" << std::endl;

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

        //tracked_local_pose.translation() /= (meanDepth / mesh_vo::mapping_mean_depth);
        //tracked_local_movement.translation() /= (meanDepth / mesh_vo::mapping_mean_depth);

        frame.scalePose(meanDepth / mesh_vo::mapping_mean_depth);

        for (int j = 0; j < frames.size(); j++)
        {
            frames[j].scalePose(meanDepth / mesh_vo::mapping_mean_depth);
        }

        frames[kframeIndex].local_pose() = SE3f();

        for (std::size_t k = 0; k < frames.size(); k++)
        {
            image_renderer.Render(kframe->mesh(),
                                  frames[k].local_pose(),
                                  frames[k].local_exposure(),
                                  cam_,
                                  plot_lvl, plot_lvl,
                                  kframe->image(),
                                  image_texture);
            // residual_renderer.Render(kframe->mesh(), SE3f(), cam_, plot_lvl, plot_lvl, kframe->frame().image(), oframes[k].image(), l2_cpu);
            cv::Mat image_mat = DownloadTextureToMat(image_texture, plot_lvl);
            cv::Mat ref_mat = DownloadTextureToMat(frames[k].image(), plot_lvl);
            cv::Mat l2_mat = ref_mat - image_mat;
            SaveDebugImage(l2_mat, "map_" + std::to_string(kframe->id()) + "_" + std::to_string(frames[k].id()) + "_l2_opt.png");
        }

        depth_renderer.Render(kframe->mesh(),
                              SE3f(),
                              cam_,
                              plot_lvl,
                              es_depth_texture);

        double error = 1.0; // RMSE(depth_cpu, estimated_depth_cpu, plot_lvl);

        cv::Mat es_depth_cv = DownloadTextureToMat(es_depth_texture, plot_lvl);
        SaveDebugImage(es_depth_cv, "depth_" + std::to_string(kframe->id()) + ".png");

        pids_renderer.Render(kframe->mesh(),
                             SE3f(),
                             cam_, plot_lvl, pids_texture);

        cv::Mat pids_cv = DownloadTextureToMat(pids_texture, plot_lvl);
        SaveDebugImage(pids_cv, "pids_" + std::to_string(kframe->id()) + ".png");

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