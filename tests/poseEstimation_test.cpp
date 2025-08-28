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
    const float translationErrorThreshold = 0.75; // best = 0.0160271;
    const float rotationErrorThreshold = 0.0011;   // best = 0.00105154;

    std::chrono::milliseconds accProcessingTime = std::chrono::milliseconds(0);
    float accTranslationError = 0;
    float accRotationError = 0;
    int framesProcessedCounter = 0;

    std::vector<float> pos_buff_, tex_buff_, wei_buff_;
    CreateScreenQuad(pos_buff_, tex_buff_, wei_buff_);
    MeshCPU screen_mesh(pos_buff_, tex_buff_, wei_buff_);

    TextureCPU<float> kimage_cpu(w_, h_, 0.0f);
    UploadMatToTexture(kimage_cpu, 0, image_src_cv_);

    TextureCPU<float> kdepth_cpu(w_, h_, 0.0f);
    UploadMatToTexture(kdepth_cpu, 0, depth_src_cv_);

    MeshCPU mesh = CreateMesh(kdepth_cpu, cam_, 32);

    TextureCPU<Vec3> kdidxy_cpu(w_, h_, Vec3(0.0, 0.0, 0.0));

    DepthRendererCPU depth_renderer;
    ImageRendererCPU image_renderer;
    DIDxyRendererCPU didxy_renderer;
    ResidualRendererCPU residual_renderer;
    L2RendererCPU l2_renderer;

    NodataReducerCPU nodata_reducer;

    for (int lvl = 0; lvl < kdidxy_cpu.levels(); lvl++)
        didxy_renderer.Render(screen_mesh, SE3(), cam_, lvl, lvl, kimage_cpu, kdidxy_cpu);

    KeyFrame kframe(Frame(kimage_cpu, kdidxy_cpu, 0, SE3(), pose_src_), mesh);

    PoseOptimizer optimizer(w_, h_, false);

    SE3 tracked_global_pose = kframe.frame().global_pose();

    TextureCPU<float> image_cpu(w_, h_, 0);
    TextureCPU<float> depth_cpu(w_, h_, 0);
    TextureCPU<Vec3> didxy_cpu(w_, h_, Vec3(0.0, 0.0, 0.0));
    TextureCPU<float> l2_texture(w_, h_, -1);

    for (unsigned int i = 1; i < image_files_.size(); i++)
    {
        std::cout << "Frame " << i << std::endl;

        cv::Mat image_cv = cv::imread(image_files_[i], cv::IMREAD_GRAYSCALE);
        cv::Mat gt_depth_cv = cv::imread(depth_files_[i], cv::IMREAD_GRAYSCALE);
        SE3 gt_pose = poses_[i].inverse();

        image_cv.convertTo(image_cv, CV_32FC1);
        gt_depth_cv.convertTo(gt_depth_cv, CV_32FC1);
        //  gt_depth_cv /= dataset.getDepthFactor();
        //  gt_depth_cv *= 100.0;

        UploadMatToTexture(image_cpu, 0, image_cv);
        UploadMatToTexture(depth_cpu, 0, gt_depth_cv);

        for (int lvl = 0; lvl < kdidxy_cpu.levels(); lvl++)
            didxy_renderer.Render(screen_mesh, SE3(), cam_, lvl, lvl, image_cpu, didxy_cpu);

        SE3 init_local_pose = kframe.globalPoseToLocal(tracked_global_pose);

        std::cout << "init_local_pose " << std::endl;
        std::cout << init_local_pose.translation() << std::endl;

        std::cout << "init_global_pose " << std::endl;
        std::cout << tracked_global_pose.translation() << std::endl;

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
        float pnodata = nodata.getError() / image_cpu.size(1);
        float viewPercent = 1.0 - pnodata;

        std::cout << "view percent " << viewPercent << std::endl;

        if (viewPercent < mesh_vo::min_view_perc) // || keyframeViewAngle > mesh_vo::key_max_angle)
        {
            MeshCPU new_mesh = CreateMesh(depth_cpu, cam_, 32);
            kframe = KeyFrame(frame, new_mesh);

            frame.local_pose() = kframe.globalPoseToLocal(frame.global_pose());

            depth_renderer.Render(kframe.mesh(), SE3(), cam_, 1, 1, depth_cpu);
            cv::Mat depth_mat = DownloadTexture(depth_cpu, 1, CV_32FC1);
            SaveDebugImageColor(depth_mat, "Depth keyframe_" + std::to_string(i) + ".png");

            image_renderer.Render(kframe.mesh(), frame.local_pose(), cam_, 1, 1, kframe.frame().image(), image_cpu);
            cv::Mat image_mat = DownloadTexture(image_cpu, 1, CV_32FC1);
            SaveDebugImageColor(image_mat, "Frame keyframe_" + std::to_string(i) + ".png");

            Error nodata = nodata_reducer.reduce(1, image_cpu);
            float pnodata = nodata.getError() / image_cpu.size(1);
            float viewPercent = 1.0 - pnodata;

            std::cout << "new view percent " << viewPercent << std::endl;
        }

        residual_renderer.Render(kframe.mesh(), frame.local_pose(), cam_, 1, 1, kframe.frame().image(), frame.image(), l2_texture);
        cv::Mat l2_mat = DownloadTexture(l2_texture, 1, CV_32FC1);
        SaveDebugImageColor(l2_mat, "l2_" + std::to_string(i) + ".png");
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