#include <gtest/gtest.h>
#include "common/test_framework.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "backends/cpu/renderercpu.h"
#include "optimizers/poseOptimizer.h"

template <typename Type>
inline double ComputeImageError(const cv::Mat &image_est, const cv::Mat &image_gt, Type nodata_value)
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
inline std::array<double, 2> ComputeSE3Error(const SE3 &pose_est, const SE3 &pose_gt)
{
    // Compute the relative transformation: error transformation T_error
    SE3 T_error = pose_est.inverse() * pose_gt;

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
    const int in_lvl = 0, out_lvl = 0;

    const long long acceptableTimeMs = 30;
    const float translationErrorThreshold = 0.017; // best = 0.0160271;
    const float rotationErrorThreshold = 0.0011;   // best = 0.00105154;

    std::chrono::milliseconds accProcessingTime = std::chrono::milliseconds(0);
    float accTranslationError = 0;
    float accRotationError = 0;
    int framesProcessedCounter = 0;

    std::vector<float> pos_buff_, tex_buff_, wei_buff_;
    CreateScreenQuad(pos_buff_, tex_buff_, wei_buff_);
    MeshCPU screen_mesh(pos_buff_, tex_buff_, wei_buff_);
    MeshCPU mesh(vertices_, texcoords_, weights_);

    TextureCPU<float> kimage_cpu(w_, h_, 0.0f);
    UploadMatToTexture(kimage_cpu, 0, image_src_cv_);

    TextureCPU<Vec3> kdidxy_cpu(w_, h_, Vec3(0.0, 0.0, 0.0));

    ImageRendererCPU image_renderer;
    DIDxyRendererCPU didxy_renderer;

    for (int lvl = 0; lvl < kdidxy_cpu.levels(); lvl++)
        didxy_renderer.Render(screen_mesh, SE3(), cam_, kimage_cpu, kdidxy_cpu, lvl, lvl);

    KeyFrame kframe(Frame(kimage_cpu, kdidxy_cpu, 0, SE3(), pose_src_), mesh);

    PoseOptimizer optimizer(w_, h_, true);

    SE3 tracked_global_pose = kframe.frame().global_pose();

    for (unsigned int i = 1; i < image_files_.size(); i++)
    {
        cv::Mat image_cv = cv::imread(image_files_[i], cv::IMREAD_GRAYSCALE);
        cv::Mat gt_depth_cv = cv::imread(depth_files_[i], cv::IMREAD_GRAYSCALE);
        SE3 gt_pose = poses_[i].inverse();

        image_cv.convertTo(image_cv, CV_32FC1);
        // gt_depth_cv.convertTo(gt_depth_cv, CV_32FC1);
        //  gt_depth_cv /= dataset.getDepthFactor();
        //  gt_depth_cv *= 100.0;

        TextureCPU<float> image_cpu(w_, h_, 0);
        UploadMatToTexture(image_cpu, 0, image_cv);

        TextureCPU<Vec3> didxy_cpu(w_, h_, Vec3(0.0, 0.0, 0.0));
        for (int lvl = 0; lvl < kdidxy_cpu.levels(); lvl++)
            didxy_renderer.Render(screen_mesh, SE3(), cam_, image_cpu, didxy_cpu, lvl, lvl);

        SE3 init_local_pose = kframe.globalPoseToLocal(tracked_global_pose);

        Frame frame(image_cpu, didxy_cpu, i, init_local_pose, tracked_global_pose);

        auto startTime = std::chrono::high_resolution_clock::now();
        for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_fin_lvl; lvl--)
        {
            optimizer.init(frame, kframe, cam_, lvl);
            while (!optimizer.converged())
            {
                optimizer.step(frame, kframe, cam_, lvl);
            }
        }
        tracked_global_pose = frame.global_pose();
        auto endTime = std::chrono::high_resolution_clock::now();

        std::array<double, 2> error = ComputeSE3Error(frame.global_pose(), gt_pose);

        accProcessingTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        accTranslationError += error[0];
        accRotationError += error[1];
        framesProcessedCounter++;

        // change keyframe logic
        //float keyframeViewAngle = kframe.meanViewAngle(SE3(), frame.local_pose());

        TextureCPU<float> image_buffer(w_, h_, -1);
        image_renderer.Render(kframe.mesh(), frame.local_pose(), cam_, kframe.frame().image(), image_buffer, 1, 1);

        int count = 0;
        auto mm = image_buffer.MapRead(1);
        for(int i = 0; i < image_buffer.size(1); i++)
        {
            if(mm[i] == image_buffer.nodata())
                count++;
        }

        float pnodata = float(count)/image_buffer.size(1);
        float viewPercent = 1.0 - pnodata;

        if (viewPercent < mesh_vo::min_view_perc)// || keyframeViewAngle > mesh_vo::key_max_angle)
        {
            kframe = KeyFrame(imageData, vec2f(0.0, 0.0), gtPose, 1.0);
            kframe.initGeometryFromDepth(gtDepthData, dataCPU<float>(w, h, 1.0 / mesh_vo::mapping_param_initial_var), cam);
        }
        */
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