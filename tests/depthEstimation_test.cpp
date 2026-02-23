#include <gtest/gtest.h>
#include "test_framework.h"
#include "common/types.h"
#include "common/frame.h"
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

    Mesh screen_mesh;
    CreateScreenQuad(screen_mesh);

    DepthRenderer depth_renderer;
    PidsRenderer pids_renderer;
    ImageRenderer image_renderer;
    DIDxyRenderer didxy_renderer;

    NodataReducerCPU nodata_reducer;

    DepthEstimator estimator(w_, h_, true);

    std::vector<Frame> frames;
    std::vector<Texture<float>> gt_depth_textures;
    std::vector<SE3f> gt_global_poses;
    KeyFrame *kframe;

    Texture<ImageType> image_texture(w_, h_, 0);
    Texture<Vec3f> didxy_texture(w_, h_, Vec3f(0.0, 0.0, 0.0));
    Texture<float> gt_depth_texture(w_, h_, 0.0);
    Texture<float> es_depth_texture(w_, h_, 0.0);
    Texture<Vec3f> pids_texture(w_, h_, Vec3f(0.0, 0.0, 0.0));

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

        // didxy_renderer.Render(screen_mesh, 0, 0, image_texture, didxy_texture);
        // didxy_texture.generate_mipmaps(0);

        // cv::Mat didxy_cv = DownloadTextureToMat(didxy_texture, 1);
        // SaveDebugImage(didxy_cv, "didxy_" + std::to_string(img_id) + ".png");

        if (img_id == 0)
        {
            std::vector<float> ver_buff_;
            std::vector<int> idx_buff_;

            double gt_depth_mean = cv::mean(gt_depth_cv)[0];

            Mesh mesh;
            // CreateMesh(gt_depth_texture, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
            CreateFlatMesh(gt_depth_mean * 0.5, 
                gt_depth_mean * 1.5, 
                cam_, 
                mesh_vo::mesh_width, 
                mesh);
            //    CreateSphereMesh(gt_depth_mean, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_);

            kframe = new KeyFrame(image_texture, didxy_texture, gt_global_pose, mesh, 1.0, 0);
            float meanDepth = kframe->meanDepth();
            kframe->scaleMesh(meanDepth / mesh_vo::mapping_mean_depth);

            continue;
        }

        SE3f init_local_pose = kframe->globalPoseToLocal(gt_global_pose);

        Frame frame(image_texture, didxy_texture, img_id, kframe->id(), init_local_pose);

        float minViewAngle = M_PI;
        for (std::size_t j = 0; j < frames.size(); j++)
        {
            float viewAngle = kframe->meanViewAngle(frame.local_pose(), frames[j].local_pose(), cam_);
            if (viewAngle < minViewAngle)
                minViewAngle = viewAngle;
        }

        std::cout << "Min view angle " << minViewAngle << std::endl;

        if (minViewAngle < mesh_vo::last_min_angle && kframe->id() != 0)
            continue;

        frames.push_back(frame);
        gt_depth_textures.push_back(gt_depth_texture);
        gt_global_poses.push_back(gt_global_pose);
        if (frames.size() > mesh_vo::num_frames)
        {
            frames.erase(frames.begin());
            gt_depth_textures.erase(gt_depth_textures.begin());
            gt_global_poses.erase(gt_global_poses.begin());
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
        Frame newKeyFrame = frames[kframeIndex];

        estimator.changeKeyframe(newKeyFrame, frames, kframe, cam_);

        std::cout << "Mean depth " << kframe->meanDepth() << std::endl;

        //////////// Debug /////////////////
        int plot_lvl = 0;
        for (std::size_t k = 0; k < frames.size(); k++)
        {
            image_renderer.Render(kframe->mesh(),
                                  frames[k].local_pose(),
                                  frames[k].local_exposure(),
                                  cam_,
                                  plot_lvl, plot_lvl,
                                  kframe->image(), image_texture);
            cv::Mat image_mat = DownloadTextureToMat(image_texture, plot_lvl);
            cv::Mat ref_mat = DownloadTextureToMat(frames[k].image(), plot_lvl);
            cv::Mat l2_mat = ref_mat - image_mat;
            SaveDebugImage(l2_mat, "l2_" + std::to_string(kframe->id()) + "_" + std::to_string(frames[k].id()) + "_init.png");
        }
        //////////////////////////

        std::vector<Frame> oframes = frames;
        oframes.erase(oframes.begin() + kframeIndex);

        auto startTime = std::chrono::high_resolution_clock::now();
        estimator.estimate(oframes, *kframe, cam_);
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        auto duration = processingTime.count();
        std::cout << "processing time " << duration << " ms" << std::endl;

        float new_mean_depth = kframe->meanDepth();
        kframe->scaleMesh(new_mean_depth / mesh_vo::mapping_mean_depth);
        frame.scalePose(new_mean_depth / mesh_vo::mapping_mean_depth);
        for (std::size_t k = 0; k < frames.size(); k++)
        {
            frames[k].scalePose(new_mean_depth / mesh_vo::mapping_mean_depth);
        }

        ///////////////// Debug ///////////////////
        for (std::size_t k = 0; k < frames.size(); k++)
        {
            image_renderer.Render(kframe->mesh(),
                                  frames[k].local_pose(),
                                  frames[k].local_exposure(),
                                  cam_,
                                  plot_lvl, plot_lvl,
                                  kframe->image(), image_texture);
            cv::Mat image_mat = DownloadTextureToMat(image_texture, plot_lvl);
            cv::Mat ref_mat = DownloadTextureToMat(frames[k].image(), plot_lvl);
            cv::Mat l2_mat = ref_mat - image_mat;
            SaveDebugImage(l2_mat, "l2_" + std::to_string(kframe->id()) + "_" + std::to_string(frames[k].id()) + "_opt.png");
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
        double error = RMSE(es_depth_texture, gt_depth_textures[kframeIndex], plot_lvl);

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