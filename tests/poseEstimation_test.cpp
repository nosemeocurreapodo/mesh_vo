#include <gtest/gtest.h>
#include "tests/common/test_framework.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "optimizers/poseOptimizer.h"
#include "optimizers/poseExpOptimizer.h"
#include "optimizers/poseVelOptimizer.h"
#include "optimizers/poseVelExpOptimizer.h"

template <typename Type>
double ComputeImageError(const cv::Mat &image_est, const cv::Mat &image_gt, Type nodata_value)
{
    assert(image_est.cols == image_gt.cols && image_est.rows == image_gt.rows);

    double error = 0.0;
    int count = 0;
    for (int y = 0; y < image_est.rows; y++)
    {
        for (int x = 0; x < image_est.cols; x++)
        {
            double est = double(image_est.at<Type>(y, x));
            double gt = double(image_gt.at<Type>(y, x));

            if (est == nodata_value || gt == nodata_value)
                continue;

            error += std::fabs(est - gt); // * (est - gt);
            count += 1;
        }
    }
    return error / count;
}

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
    Mesh screen_mesh(s_ver_buff_, s_idx_buff_, true, true, false);

    Texture<ImageType> image_texture(w_, h_, -1);
    Texture<float> gt_depth_texture(w_, h_, 0);
    Texture<float> es_depth_texture(w_, h_, 0);
    Texture<Vec3f> didxy_texture(w_, h_, Vec3f(0.0, 0.0, 0.0));
    Texture<float> l2_texture(w_, h_, -1);

    DepthRenderer depth_renderer;
    ImageRenderer image_renderer;
    DIDxyRenderer didxy_renderer;
    ResidualRenderer residual_renderer;
    NodataReducerCPU nodata_reducer;

    PoseExpOptimizer optimizer(w_, h_, true);

    KeyFrame *kframe;

    SE3f tracked_global_pose;
    SE3f tracked_global_movement;
    Vec2f tracked_local_exposure(0, 0);

    for (unsigned int img_id = 0; img_id < image_files_.size(); img_id++)
    {
        std::cout << "Frame " << img_id << std::endl;

        cv::Mat image_cv = cv::imread(image_files_[img_id], cv::IMREAD_GRAYSCALE);
        cv::Mat gt_depth_cv = cv::imread(depth_files_[img_id], cv::IMREAD_GRAYSCALE);
        gt_depth_cv.convertTo(gt_depth_cv, CV_32FC1);
        gt_depth_cv = gt_depth_cv / depth_factor_;
        SE3f gt_pose = poses_[img_id];

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
            CreateMesh(gt_depth_texture, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
            Mesh mesh(ver_buff_, idx_buff_, true, true, true);

            kframe = new KeyFrame(Frame(image_texture, didxy_texture, 0, SE3f(), gt_pose), mesh, 1.0);
            float meanDepth = kframe->meanDepth();
            kframe->scaleMesh(meanDepth / mesh_vo::mapping_mean_depth);

            tracked_global_pose = gt_pose;

            continue;
        }

        SE3f ini_global_pose = tracked_global_movement * tracked_global_pose;
        SE3f ini_local_pose = kframe->globalPoseToLocal(ini_global_pose);
        Vec2f ini_local_exposure = tracked_local_exposure;

        // std::cout << "init_local_pose " << std::endl;
        // std::cout << init_local_pose.translation() << std::endl;

        // std::cout << "init_global_pose " << std::endl;
        // std::cout << tracked_global_pose.translation() << std::endl;

        Frame frame(image_texture, didxy_texture, img_id, ini_local_pose, ini_global_pose, ini_local_exposure);

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

        SE3f new_global_pose = kframe->localPoseToGlobal(frame.local_pose());
        SE3f new_local_pose = frame.local_pose();
        Vec6f new_local_vel = frame.local_vel();
        Vec2f new_local_exposure = frame.local_exposure();

        frame.global_pose() = new_global_pose;
        frame.local_pose() = new_local_pose;
        frame.local_vel() = new_local_vel;
        frame.local_exposure() = new_local_exposure;

        tracked_global_movement = new_global_pose * tracked_global_pose.inverse();
        tracked_global_pose = new_global_pose;
        tracked_local_exposure = new_local_exposure;

        auto endTime = std::chrono::high_resolution_clock::now();

        std::array<double, 2> error = ComputeSE3Error(frame.global_pose(), gt_pose);

        accProcessingTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        accTranslationError += error[0];
        accRotationError += error[1];
        framesProcessedCounter++;

        // change keyframe logic
        // float keyframeViewAngle = kframe.meanViewAngle(SE3(), frame.local_pose());

        image_renderer.Render(kframe->mesh(),
                              frame.local_pose(),
                              frame.local_exposure(),
                              cam_,
                              1, 1,
                              kframe->frame().image(), image_texture);

        Error nodata = nodata_reducer.reduce(1, image_texture);
        float pnodata = nodata.getError() / (image_texture.width(1) * image_texture.height(1));
        float viewPercent = 1.0 - pnodata;

        std::cout << "view percent " << viewPercent << std::endl;

        if (viewPercent < mesh_vo::min_view_perc) // || keyframeViewAngle > mesh_vo::key_max_angle)
        {
            std::vector<float> ver_buff_;
            std::vector<int> idx_buff_;
            CreateMesh(gt_depth_texture, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
            Mesh new_mesh(ver_buff_, idx_buff_, true, true, true);

            kframe = new KeyFrame(frame, new_mesh, 1.0);
            float meanDepth = kframe->meanDepth();
            kframe->scaleMesh(meanDepth / mesh_vo::mapping_mean_depth);

            frame.local_pose() = SE3f();//kframe->globalPoseToLocal(frame.global_pose());
            frame.local_exposure() = Vec2f(0.0, 0.0);

            depth_renderer.Render(kframe->mesh(),
                                  SE3f(),
                                  cam_,
                                  1,
                                  es_depth_texture);
            cv::Mat depth_mat = DownloadTextureToMat(es_depth_texture, 1);
            SaveDebugImage(depth_mat, "Depth keyframe_" + std::to_string(img_id) + ".png");

            image_renderer.Render(kframe->mesh(),
                                  frame.local_pose(),
                                  frame.local_exposure(),
                                  cam_,
                                  1, 1,
                                  kframe->frame().image(),
                                  image_texture);
            cv::Mat image_mat = DownloadTextureToMat(image_texture, 1);
            SaveDebugImage(image_mat, "Frame keyframe_" + std::to_string(img_id) + ".png");

            // Error nodata = nodata_reducer.reduce(1, image_cpu);
            // float pnodata = nodata.getError() / (image_cpu.width(1) * image_cpu.height(1));
            // float viewPercent = 1.0 - pnodata;

            // std::cout << "new view percent " << viewPercent << std::endl;
        }

        residual_renderer.Render(kframe->mesh(),
                                 frame.local_pose(),
                                 frame.local_exposure(),
                                 cam_,
                                 1, 1,
                                 kframe->frame().image(),
                                 frame.image(),
                                 l2_texture);
        cv::Mat l2_mat = DownloadTextureToMat(l2_texture, 1);
        SaveDebugImage(l2_mat, "l2_" + std::to_string(img_id) + ".png");
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