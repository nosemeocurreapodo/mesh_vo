#include <gtest/gtest.h>
#include "tests/common/test_framework.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "optimizers/poseOptimizer.h"
#include "optimizers/poseExpOptimizer.h"
// #include "optimizers/poseVelOptimizer.h"
// #include "optimizers/poseVelExpOptimizer.h"

// Function to compute error between two SE3 poses
std::array<double, 2> ComputeSE3Error(const SE3f &pose_est, const SE3f &pose_gt)
{
    // Compute the relative transformation: error transformation T_error
    SE3f T_error = pose_est.inverse() * pose_gt;

    double translation_error = T_error.translation().norm();
    double rotation_error = 0.0; // T_error.so3().log().norm();

    std::array<double, 2> error = {translation_error, rotation_error};

    return error;

    // Convert T_error to a 6D vector (Lie algebra) representing the error
    // vec6f error_vector = T_error.log();

    // Return the norm of the error vector
    // return error_vector.norm();
}

TEST_F(RendererTestBase, ComputePose)
{

#ifdef COMPILE_GL
    InitEGL();
#endif

    const int in_lvl = 0, out_lvl = 0;

    const long long acceptableTimeMs = 30;
    const float translationErrorThreshold = 0.02; // best = 0.0160271;
    const float rotationErrorThreshold = 0.0011;  // best = 0.00105154;

    std::chrono::milliseconds accProcessingTime = std::chrono::milliseconds(0);
    float accTranslationError = 0;
    float accRotationError = 0;
    int framesProcessedCounter = 0;

    std::vector<float> s_ver_buff_;
    std::vector<int> s_idx_buff_;
    CreateScreenQuad(s_ver_buff_, s_idx_buff_);
    Mesh screen_mesh(s_ver_buff_, s_idx_buff_, false, true, false);

    Texture<ImageType> image_texture(w_, h_, 0);
    Texture<float> gt_depth_texture(w_, h_, 0);
    Texture<float> es_depth_texture(w_, h_, 0);
    Texture<Vec3f> didxy_texture(w_, h_, Vec3f(0.0, 0.0, 0.0));

    DepthRenderer depth_renderer;
    ImageRenderer image_renderer;
    DIDxyRenderer didxy_renderer;
    NodataReducerCPU nodata_reducer;

    PoseOptimizer optimizer(w_, h_, true);

    KeyFrame *kframe;

    SE3f tracked_local_pose;
    SE3f tracked_local_movement;
    Vec2f tracked_local_exposure(0, 0);

    for (unsigned int img_id = 0; img_id < image_files_.size(); img_id++)
    {
        std::cout << "Frame " << img_id << std::endl;

        cv::Mat image_cv = cv::imread(image_files_[img_id], cv::IMREAD_GRAYSCALE);
        cv::Mat gt_depth_cv = cv::imread(depth_files_[img_id], cv::IMREAD_GRAYSCALE);
        gt_depth_cv.convertTo(gt_depth_cv, CV_32FC1);
        gt_depth_cv = gt_depth_cv / depth_factor_;
        SE3f gt_global_pose = poses_[img_id];

        UploadMatToTexture(image_texture, 0, image_cv);
        UploadMatToTexture(gt_depth_texture, 0, gt_depth_cv);

        for (int lvl = 0; lvl < didxy_texture.levels(); lvl++)
            didxy_renderer.Render(screen_mesh, lvl, lvl, image_texture, didxy_texture);

        // didxy_renderer.Render(screen_mesh, 0, 0, image_cpu, didxy_cpu);
        // didxy_cpu.generate_mipmaps(0);

        if (img_id == 0)
        {
            std::vector<float> ver_buff_;
            std::vector<int> idx_buff_;
            CreateMesh(gt_depth_texture, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, false, false);
            Mesh mesh(ver_buff_, idx_buff_, true, false, false);

            kframe = new KeyFrame(image_texture, didxy_texture, gt_global_pose, mesh, 1.0, 0);
            float meanDepth = kframe->meanDepth();
            kframe->scaleMesh(meanDepth / mesh_vo::mapping_mean_depth);

            continue;
        }

        SE3f ini_local_pose = tracked_local_movement * tracked_local_pose;
        Vec2f ini_local_exposure = tracked_local_exposure;

        Frame frame(image_texture, didxy_texture, img_id, kframe->id(), ini_local_pose, ini_local_exposure);

        auto startTime = std::chrono::high_resolution_clock::now();
        for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_fin_lvl; lvl--)
        {
            int in_lvl = lvl;
            int out_lvl = lvl;
            optimizer.init(frame, *kframe, cam_, in_lvl, out_lvl);
            while (!optimizer.converged())
            {
                optimizer.step(frame, *kframe, cam_, in_lvl, out_lvl);
            }
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        auto duration = processingTime.count();
        std::cout << "processing time " << duration << " ms" << std::endl;

        SE3f new_local_pose = frame.local_pose();
        Vec6f new_local_vel = frame.local_vel();
        Vec2f new_local_exposure = frame.local_exposure();

        tracked_local_movement = new_local_pose * tracked_local_pose.inverse();
        tracked_local_pose = new_local_pose;
        tracked_local_exposure = new_local_exposure;

        SE3f es_global_pose = kframe->localPoseToGlobal(frame.local_pose());

        std::array<double, 2> error = ComputeSE3Error(es_global_pose, gt_global_pose);

        accProcessingTime += processingTime;
        accTranslationError += error[0];
        accRotationError += error[1];
        framesProcessedCounter++;

        image_renderer.Render(kframe->mesh(),
                              frame.local_pose(),
                              frame.local_exposure(),
                              cam_,
                              1, 1,
                              kframe->image(),
                              image_texture);
        cv::Mat image_mat = DownloadTextureToMat(image_texture, 1);
        cv::Mat ref_mat = DownloadTextureToMat(frame.image(), 1);
        cv::Mat l2_mat = ref_mat - image_mat;
        SaveDebugImage(l2_mat, "l2_" + std::to_string(img_id) + ".png");

        Error nodata;
        nodata_reducer.reduce(1, image_texture, nodata);
        float pnodata = nodata.getError() / (image_texture.width(1) * image_texture.height(1));
        float viewPercent = 1.0 - pnodata;

        std::cout << "view percent " << viewPercent << std::endl;

        if (viewPercent > mesh_vo::min_view_perc) // || keyframeViewAngle > mesh_vo::key_max_angle)
            continue;

        std::vector<float> ver_buff_;
        std::vector<int> idx_buff_;
        CreateMesh(gt_depth_texture, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
        Mesh new_mesh(ver_buff_, idx_buff_, true, true, true);

        kframe = new KeyFrame(frame.image(), frame.didxy(), es_global_pose, new_mesh, 1.0, frame.id());
        frame.local_pose() = SE3f();
        frame.local_exposure() = Vec2f(0.0, 0.0);
        frame.local_vel() = Vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        tracked_local_pose = SE3f();
        tracked_local_movement = SE3f();
        tracked_local_exposure = Vec2f(0.0, 0.0);

        float meanDepth = kframe->meanDepth();
        kframe->scaleMesh(meanDepth / mesh_vo::mapping_mean_depth);
        frame.scalePose(meanDepth / mesh_vo::mapping_mean_depth);

        depth_renderer.Render(kframe->mesh(),
                              SE3f(),
                              cam_,
                              1,
                              es_depth_texture);
        cv::Mat depth_mat = DownloadTextureToMat(es_depth_texture, 1);
        SaveDebugImage(depth_mat, "Depth keyframe_" + std::to_string(img_id) + ".png");

        image_renderer.Render(kframe->mesh(),
                              SE3f(),
                              Vec2f(0.0, 0.0),
                              cam_,
                              1, 1,
                              kframe->image(),
                              image_texture);
        cv::Mat image_mat_ = DownloadTextureToMat(image_texture, 1);
        SaveDebugImage(image_mat_, "Frame keyframe_" + std::to_string(img_id) + ".png");
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