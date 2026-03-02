#include <gtest/gtest.h>
#include "test_framework.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "poseEstimator.h"

// Function to compute error between two SE3 poses
std::array<double, 2> ComputeSE3Error(const SE3f &pose_est, const SE3f &pose_gt)
{
    // Compute the relative transformation: error transformation T_error
    SE3f T_error = pose_est.inverse() * pose_gt;

    double translation_error = T_error.translation().norm();
    double rotation_error = T_error.so3().log().norm();

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

    const long long acceptableTimeMs = 30;
    const float translationErrorThreshold = 0.02; // best = 0.0160271;
    const float rotationErrorThreshold = 0.0011;  // best = 0.00105154;

    std::chrono::milliseconds accProcessingTime = std::chrono::milliseconds(0);
    float accTranslationError = 0;
    float accRotationError = 0;
    int framesProcessedCounter = 0;

    Mesh screen_mesh = CreateScreenQuad<Mesh>();

    Texture<ImageType> image_tex_tmp(w_, h_, 0);
    Texture<float> depth_tex_tmp(w_, h_, 0);

    DepthRenderer depth_renderer;
    ImageRenderer image_renderer;
    DIDxyRenderer didxy_renderer;
    NodataReducerCPU nodata_reducer;

    PoseEstimator estimator(w_, h_, true);
    Frame frame(w_, h_);
    KeyFrame kframe(w_, h_,
                    CreateFlatMesh<Mesh>(0.5,
                                         1.5,
                                         cam_,
                                         mesh_vo::mesh_width));

    for (unsigned int img_id = 0; img_id < image_files_.size(); img_id++)
    {
        std::cout << "Frame " << img_id << std::endl;

        cv::Mat image_cv = cv::imread(image_files_[img_id], cv::IMREAD_GRAYSCALE);
        cv::Mat gt_depth_cv = cv::imread(depth_files_[img_id], cv::IMREAD_GRAYSCALE);
        gt_depth_cv.convertTo(gt_depth_cv, CV_32FC1);
        gt_depth_cv = gt_depth_cv / depth_factor_;
        double gt_depth_mean = cv::mean(gt_depth_cv)[0];
        SE3f gt_global_pose = poses_[img_id];
        UploadMatToTexture(depth_tex_tmp, 0, gt_depth_cv);

        if (img_id == 0)
        {
            UploadMatToTexture(kframe.image(), 0, image_cv);

            Mesh mesh = CreateMesh<Mesh>(depth_tex_tmp.MapRead(0).data(),
                                         cam_,
                                         depth_tex_tmp.width(0),
                                         depth_tex_tmp.height(0),
                                         mesh_vo::mesh_width,
                                         gt_depth_mean);

            kframe.global_pose() = gt_global_pose;
            kframe.id() = img_id;
            kframe.mesh() = std::move(mesh);
            kframe.global_scale() = 1.0;

            estimator.init_global_pose(gt_global_pose);

            continue;
        }

        UploadMatToTexture(frame.image(), 0, image_cv);

        for (int lvl = 0; lvl < frame.didxy().levels(); lvl++)
            didxy_renderer.Render(screen_mesh, lvl, lvl, frame.image(), frame.didxy());
        
        //didxy_renderer.Render(screen_mesh, 0, 0, image_cpu, didxy_cpu);
        //didxy_cpu.generate_mipmaps(0);

        // frame.global_pose() = gt_global_pose;
        frame.id() = img_id;

        estimator.guess(frame, kframe);

        auto startTime = std::chrono::high_resolution_clock::now();
        estimator.estimate(frame, kframe, cam_);
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        auto duration = processingTime.count();
        std::cout << "processing time " << duration << " ms" << std::endl;

        std::array<double, 2> error = ComputeSE3Error(frame.global_pose(), gt_global_pose);
        std::cout << "translation error " << error[0] << " rotation error " << error[1] << std::endl;

        accProcessingTime += processingTime;
        accTranslationError += error[0];
        accRotationError += error[1];
        framesProcessedCounter++;

        image_renderer.Render(kframe.mesh(),
                              kframe.global_pose_to_local(frame.global_pose()),
                              frame.local_exposure(),
                              cam_,
                              1, 1,
                              kframe.image(),
                              image_tex_tmp);

        cv::Mat image_mat = DownloadTextureToMat(image_tex_tmp, 1);
        cv::Mat ref_mat = DownloadTextureToMat(frame.image(), 1);
        cv::Mat l2_mat = ref_mat - image_mat;
        SaveDebugImage(image_mat, "pose_est_" + std::to_string(img_id) + "_image.png");
        SaveDebugImage(ref_mat, "pose_est_" + std::to_string(img_id) + "_ref.png");
        SaveDebugImage(l2_mat, "pose_est_" + std::to_string(img_id) + "_l2.png");

        Error nodata;
        nodata_reducer.reduce(1, image_tex_tmp, nodata);
        float pnodata = nodata.getError() / (image_tex_tmp.width(1) * image_tex_tmp.height(1));
        float viewPercent = 1.0 - pnodata;

        std::cout << "view percent " << viewPercent << std::endl;

        if (viewPercent > mesh_vo::min_view_perc) // || keyframeViewAngle > mesh_vo::key_max_angle)
            continue;

        Mesh new_mesh = CreateMesh<Mesh>(depth_tex_tmp.MapRead(0).data(),
                                         cam_,
                                         depth_tex_tmp.width(0),
                                         depth_tex_tmp.height(0),
                                         mesh_vo::mesh_width,
                                         gt_depth_mean);

        kframe.image() = frame.image();
        kframe.global_pose() = frame.global_pose();
        kframe.global_scale() = 1.0;
        kframe.id() = frame.id();
        kframe.mesh() = std::move(new_mesh);

        depth_renderer.Render(kframe.mesh(),
                              SE3f(),
                              cam_,
                              1,
                              depth_tex_tmp);
        cv::Mat depth_mat = DownloadTextureToMat(depth_tex_tmp, 1);
        SaveDebugImage(depth_mat, "Depth keyframe_" + std::to_string(img_id) + ".png");

        image_renderer.Render(kframe.mesh(),
                              SE3f(),
                              Vec2f(0.0, 0.0),
                              cam_,
                              1, 1,
                              kframe.image(),
                              image_tex_tmp);
        cv::Mat image_mat_ = DownloadTextureToMat(image_tex_tmp, 1);
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