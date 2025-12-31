#include <gtest/gtest.h>
#include "tests/common/test_framework.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "optimizers/mapOptimizer.h"
#include "optimizers/mapExpOptimizer.h"

TEST_F(RendererTestBase, ComputeMap)
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
    PidsRenderer pids_renderer;
    ImageRenderer image_renderer;
    DIDxyRenderer didxy_renderer;
    ResidualRenderer residual_renderer;

    NodataReducerCPU nodata_reducer;

    MapExpOptimizer optimizer(w_, h_, true);

    std::vector<Frame> frames;
    std::vector<Texture<float>> gt_depths;

    Texture<ImageType> image(w_, h_, 0);
    Texture<Vec3f> didxy(w_, h_, Vec3f(0.0, 0.0, 0.0));
    Texture<float> gt_depth(w_, h_, 0.0);
    Texture<float> depth(w_, h_, 0.0);
    Texture<float> l2(w_, h_, 0.0);
    Texture<Vec3f> pids_texture(w_, h_, Vec3f(0.0, 0.0, 0.0));

    cv::Mat image_cv = cv::imread(image_files_[0], cv::IMREAD_GRAYSCALE);
    cv::Mat gt_depth_cv = cv::imread(depth_files_[0], cv::IMREAD_GRAYSCALE);
    gt_depth_cv.convertTo(gt_depth_cv, CV_32FC1);
    gt_depth_cv = gt_depth_cv / depth_factor_;
    double gt_depth_min;
    double gt_depth_max;
    double gt_depth_mean;
    cv::minMaxLoc(gt_depth_cv, &gt_depth_min, &gt_depth_max);
    gt_depth_mean = cv::mean(gt_depth_cv)[0];
    UploadMatToTexture(image, 0, image_cv);
    UploadMatToTexture(gt_depth, 0, gt_depth_cv);
    SE3f gt_pose = poses_[0];

    std::vector<float> ver_buff_;
    std::vector<int> idx_buff_;

    // CreateMesh(gt_depth, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
    CreateFlatMesh(gt_depth_mean * 0.5, gt_depth_mean * 1.5, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
    //     CreateSphereMesh(mesh_vo::mapping_mean_depth, cam_, mesh_vo::mesh_width, pos_buff_, tex_buff_, wei_buff_, idx_buff_);

    Mesh mesh(ver_buff_, idx_buff_, true, true, true);
    KeyFrame kframe(Frame(image, didxy, 0, SE3f(), gt_pose), mesh, 1.0);
    float meanDepth = kframe.meanDepth();
    kframe.scaleMesh(meanDepth / mesh_vo::mapping_mean_depth);

    std::cout << "Initial mean depth " << kframe.meanDepth() << std::endl;

    for (std::size_t img_id = 1; img_id < image_files_.size(); img_id++)
    {
        std::cout << "Frame " << img_id << std::endl;

        cv::Mat image_cv = cv::imread(image_files_[img_id], cv::IMREAD_GRAYSCALE);
        cv::Mat gt_depth_cv = cv::imread(depth_files_[img_id], cv::IMREAD_GRAYSCALE);
        gt_depth_cv.convertTo(gt_depth_cv, CV_32FC1);
        gt_depth_cv = gt_depth_cv / depth_factor_;
        UploadMatToTexture(image, 0, image_cv);
        UploadMatToTexture(gt_depth, 0, gt_depth_cv);
        gt_pose = poses_[img_id];

        for (int lvl = 0; lvl < didxy.levels(); lvl++)
            didxy_renderer.Render(screen_mesh, lvl, lvl, image, didxy);

        // didxy_renderer.Render(screen_mesh, 0, 0, image_cpu, didxy_cpu);
        // didxy_cpu.generate_mipmaps(0);

        SE3f init_local_pose = kframe.globalPoseToLocal(gt_pose);

        Frame frame(image, didxy, img_id, init_local_pose, gt_pose);

        float minViewAngle = M_PI;
        for (std::size_t j = 0; j < frames.size(); j++)
        {
            float viewAngle = kframe.meanViewAngle(frame.local_pose(), frames[j].local_pose());
            if (viewAngle < minViewAngle)
                minViewAngle = viewAngle;
        }

        std::cout << "Min view angle " << minViewAngle << std::endl;

        if (minViewAngle < mesh_vo::last_min_angle && kframe.frame().id() != 0)
            continue;

        frames.push_back(frame);
        gt_depths.push_back(gt_depth);
        if (frames.size() > mesh_vo::num_frames)
        {
            frames.erase(frames.begin());
            gt_depths.erase(gt_depths.begin());
        }
        else
        {
            continue;
        }

        image_renderer.Render(kframe.mesh(),
                              frame.local_pose(),
                              frame.local_exposure(),
                              cam_,
                              1, 1,
                              kframe.frame().image(), image);

        Error nodata = nodata_reducer.reduce(1, image);
        float pnodata = nodata.getError() / (image.width(1) * image.height(1));
        float viewPercent = 1.0 - pnodata;

        std::cout << "view percent " << viewPercent << std::endl;

        if (viewPercent > mesh_vo::min_view_perc && kframe.frame().id() != 0) // || keyframeViewAngle > mesh_vo::key_max_angle)
            continue;

        int kframeIndex = frames.size() / 2;
        std::vector<Frame> oframes = frames;
        oframes.erase(oframes.begin() + kframeIndex);

        /*
        // CreateMesh(gt_depths[kframeIndex], cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
        // CreateFlatMesh(mesh_vo::mapping_mean_depth * 0.5, mesh_vo::mapping_mean_depth * 1.5, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
        // CreateSphereMesh(mesh_vo::mapping_mean_depth, cam_, mesh_vo::mesh_width, pos_buff_, tex_buff_, wei_buff_, idx_buff_);
        depth_renderer.Render(kframe.mesh(),
                              frames[kframeIndex].local_pose(),
                              cam_,
                              0,
                              depth);
        CreateMesh(depth, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);

        Mesh mesh(ver_buff_, idx_buff_, true, true, true);
        float global_scale = kframe.getGlobalScale();
        kframe = KeyFrame(frames[kframeIndex], mesh, global_scale);
        // kframe.scaleMesh(global_scale);
        */

        kframe.changeFrame(frames[kframeIndex], cam_);

        std::cout << "Mean depth " << kframe.meanDepth() << std::endl;

        frame.local_pose() = kframe.globalPoseToLocal(frame.global_pose());

        for (std::size_t k = 0; k < oframes.size(); k++)
        {
            oframes[k].local_pose() = kframe.globalPoseToLocal(oframes[k].global_pose());
        }

        auto startTime = std::chrono::high_resolution_clock::now();

        for (int lvl = mesh_vo::mapping_ini_lvl; lvl >= mesh_vo::mapping_fin_lvl; lvl--)
        {
            int in_lvl = lvl;
            int out_lvl = lvl;

            optimizer.init(oframes, kframe, cam_, in_lvl, out_lvl);
            while (!optimizer.converged())
            {
                optimizer.step(oframes, kframe, cam_, in_lvl, out_lvl);
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();

        int plot_lvl = 1;
        for (std::size_t k = 0; k < oframes.size(); k++)
        {
            residual_renderer.Render(kframe.mesh(),
                                     oframes[k].local_pose(),
                                     oframes[k].local_exposure(),
                                     cam_,
                                     plot_lvl, plot_lvl,
                                     kframe.frame().image(), oframes[k].image(), l2);
            // residual_renderer.Render(kframe->mesh(), SE3f(), cam_, plot_lvl, plot_lvl, kframe->frame().image(), oframes[k].image(), l2_cpu);
            cv::Mat l2_mat = DownloadTextureToMat(l2, plot_lvl);
            SaveDebugImage(l2_mat, "l2_" + std::to_string(img_id) + "_" + std::to_string(k) + ".png");
        }

        float new_mean_depth = kframe.meanDepth();
        kframe.scaleMesh(new_mean_depth / mesh_vo::mapping_mean_depth);

        for (std::size_t k = 0; k < frames.size(); k++)
        {
            frames[k].local_pose() = kframe.globalPoseToLocal(frames[k].global_pose());
        }

        float scale = kframe.getGlobalScale();
        kframe.scaleMesh(1.0 / scale);
        depth_renderer.Render(kframe.mesh(),
                              SE3f(),
                              cam_,
                              plot_lvl,
                              depth);
        kframe.scaleMesh(scale);
        double error = RMSE(depth, gt_depths[kframeIndex], plot_lvl);

        cv::Mat depth_cv = DownloadTextureToMat(depth, plot_lvl);
        SaveDebugImage(depth_cv, "depth_" + std::to_string(img_id) + ".png");

        cv::Mat gt_depth_cv_2 = DownloadTextureToMat(gt_depths[kframeIndex], plot_lvl);
        SaveDebugImage(gt_depth_cv_2, "depth_" + std::to_string(img_id) + "_gt.png");

        pids_renderer.Render(kframe.mesh(),
                             frames[kframeIndex].local_pose(),
                             cam_, plot_lvl, pids_texture);

        cv::Mat pids_cv = DownloadTextureToMat(pids_texture, plot_lvl);
        SaveDebugImage(pids_cv, "pids_" + std::to_string(img_id) + ".png");

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