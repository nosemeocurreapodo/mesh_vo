#include <gtest/gtest.h>
#include "tests/common/test_framework.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "optimizers/poseOptimizer.h"

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
    std::vector<unsigned int> s_idx_buff_;
    CreateScreenQuad(s_ver_buff_, s_idx_buff_);
    Mesh screen_mesh(s_ver_buff_, s_idx_buff_, true, true, false);

    cv::Mat kimage_cv = ReadMat(image_files_[0], false);
    cv::Mat kdepth_cv = ReadMat(depth_files_[0], true) / depth_factor_;
    SE3f kpose = poses_[0];
    cv::Mat kdepth_mask = (kdepth_cv > 0.0);
    cv::Scalar kdepth_mean = cv::mean(kdepth_cv, kdepth_mask);
    kdepth_cv = kdepth_cv * mesh_vo::mapping_mean_depth / kdepth_mean[0];

    Texture<unsigned char> kimage_cpu(w_, h_, 0);
    // Texture<float> kdepth_cpu(w_, h_, 0.0f);

    UploadMatToTexture(kimage_cpu, 0, kimage_cv);
    // UploadMatToTexture(kdepth_cpu, 0, kdepth_cv);

    std::vector<float> ver_buff_;
    std::vector<unsigned int> idx_buff_;
    CreateMesh(kdepth_cv, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
    Mesh mesh(ver_buff_, idx_buff_, true, true, true);

    Texture<Vec3f> kdidxy_cpu(w_, h_, Vec3f(0.0, 0.0, 0.0));

    DepthRenderer depth_renderer;
    ImageRenderer image_renderer;
    DIDxyRenderer didxy_renderer;
    ResidualRenderer residual_renderer;

    NodataReducerCPU nodata_reducer;

    for (int lvl = 0; lvl < kdidxy_cpu.levels(); lvl++)
        didxy_renderer.Render(screen_mesh, lvl, lvl, kimage_cpu, kdidxy_cpu);

    KeyFrame kframe(Frame(kimage_cpu, kdidxy_cpu, 0, SE3f(), kpose), mesh, kdepth_mean[0]);

    PoseOptimizer optimizer(w_, h_, true);

    SE3f tracked_global_pose = kframe.frame().global_pose();

    Texture<unsigned char> image_cpu(w_, h_, 0);
    Texture<float> depth_cpu(w_, h_, -1);
    Texture<Vec3f> didxy_cpu(w_, h_, Vec3f(0.0, 0.0, 0.0));
    Texture<float> l2_texture(w_, h_, 0);

    for (unsigned int img_id = 1; img_id < image_files_.size(); img_id++)
    {
        std::cout << "Frame " << img_id << std::endl;

        cv::Mat image_cv = ReadMat(image_files_[img_id], false);
        cv::Mat depth_cv = ReadMat(depth_files_[img_id], true) / depth_factor_;
        SE3f gt_pose = poses_[img_id];
        cv::Mat depth_mask = (depth_cv > 0.0);
        cv::Scalar depth_mean = cv::mean(depth_cv, depth_mask);
        depth_cv = depth_cv * mesh_vo::mapping_mean_depth / depth_mean[0];

        UploadMatToTexture(image_cpu, 0, image_cv);
        // UploadMatToTexture(depth_cpu, 0, depth_cv);

        for (int lvl = 0; lvl < kdidxy_cpu.levels(); lvl++)
            didxy_renderer.Render(screen_mesh, lvl, lvl, image_cpu, didxy_cpu);

        SE3f init_local_pose = kframe.globalPoseToLocal(tracked_global_pose);

        // std::cout << "init_local_pose " << std::endl;
        // std::cout << init_local_pose.translation() << std::endl;

        // std::cout << "init_global_pose " << std::endl;
        // std::cout << tracked_global_pose.translation() << std::endl;

        Frame frame(image_cpu, didxy_cpu, img_id, init_local_pose, tracked_global_pose);

        auto startTime = std::chrono::high_resolution_clock::now();
        for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_fin_lvl; lvl--)
        {
            optimizer.init(frame, kframe, cam_, lvl);
            while (!optimizer.converged())
            {
                optimizer.step(frame, kframe, cam_, lvl);
            }
        }

        frame.global_pose() = kframe.localPoseToGlobal(frame.local_pose());

        tracked_global_pose = frame.global_pose();
        auto endTime = std::chrono::high_resolution_clock::now();

        std::array<double, 2> error = ComputeSE3Error(frame.global_pose(), gt_pose);

        accProcessingTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        accTranslationError += error[0];
        accRotationError += error[1];
        framesProcessedCounter++;

        // change keyframe logic
        // float keyframeViewAngle = kframe.meanViewAngle(SE3(), frame.local_pose());

        image_renderer.Render(kframe.mesh(), frame.local_pose(), cam_, 1, 1, kframe.frame().image(), image_cpu);
        Error nodata = nodata_reducer.reduce(1, image_cpu);
        float pnodata = nodata.getError() / (image_cpu.width(1) * image_cpu.height(1));
        float viewPercent = 1.0 - pnodata;

        std::cout << "view percent " << viewPercent << std::endl;

        if (viewPercent < mesh_vo::min_view_perc) // || keyframeViewAngle > mesh_vo::key_max_angle)
        {
            CreateMesh(depth_cv, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
            Mesh new_mesh(ver_buff_, idx_buff_, true, true, true);

            kframe = KeyFrame(frame, new_mesh, depth_mean[0]);

            frame.local_pose() = kframe.globalPoseToLocal(frame.global_pose());

            depth_renderer.Render(kframe.mesh(), SE3f(), cam_, 1, depth_cpu);
            cv::Mat depth_mat = DownloadTextureToMat(depth_cpu, 1, CV_32FC1);
            SaveDebugImageColor(depth_mat, "Depth keyframe_" + std::to_string(img_id) + ".png");

            image_renderer.Render(kframe.mesh(), frame.local_pose(), cam_, 1, 1, kframe.frame().image(), image_cpu);
            cv::Mat image_mat = DownloadTextureToMat(image_cpu, 1, CV_8UC1);
            SaveDebugImageColor(image_mat, "Frame keyframe_" + std::to_string(img_id) + ".png");

            // Error nodata = nodata_reducer.reduce(1, image_cpu);
            // float pnodata = nodata.getError() / (image_cpu.width(1) * image_cpu.height(1));
            // float viewPercent = 1.0 - pnodata;

            // std::cout << "new view percent " << viewPercent << std::endl;
        }

        residual_renderer.Render(kframe.mesh(), frame.local_pose(), cam_, 1, 1, kframe.frame().image(), frame.image(), l2_texture);
        cv::Mat l2_mat = DownloadTextureToMat(l2_texture, 1, CV_32FC1);
        SaveDebugImageColor(l2_mat, "l2_" + std::to_string(img_id) + ".png");
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