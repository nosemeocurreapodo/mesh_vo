#include <gtest/gtest.h>

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <iostream>
#include <chrono>

#include "opencv2/opencv.hpp"

#include "common.h"
#include "optimizers/poseOptimizerCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/OpenCVDebug.h"

// Test to ensure PoseEstimator correctly computes the pose
TEST(PoseEstimatorTest, ComputePose)
{
    const long long acceptableTimeMs = 30;
    const float translationErrorThreshold = 0.017; // best = 0.0160271;
    const float rotationErrorThreshold = 0.0011;   // best = 0.00105154;

    // Validate that the pose is within expected bounds
    // EXPECT_NEAR(pose.translation.x, 0.0, 0.001);
    // EXPECT_NEAR(pose.translation.y, 0.0, 0.001);
    // EXPECT_NEAR(pose.translation.z, 0.0, 0.001);

    // load_dataset_tum_rgbd dataset;
    load_dataset_icl_nuim dataset;

    std::vector<std::string> image_files = dataset.getImageFiles();
    std::vector<std::string> depth_files = dataset.getDepthFiles();
    std::vector<SE3f> poses = dataset.getPoses();
    std::vector<double> timestamps = dataset.getTimestamps();
    cameraType cam = dataset.getCamera();
    int w = dataset.getWidth();
    int h = dataset.getHeight();

    std::chrono::milliseconds accProcessingTime = std::chrono::milliseconds(0);
    float accTranslationError = 0;
    float accRotationError = 0;
    int framesProcessedCounter = 0;

    poseOptimizerCPU optimizer(w, h, false);
    renderCPU renderer(w, h);
    keyFrameCPU kframe;
    SE3f lastEstimatedGlobalPose;

    for (unsigned int i = 0; i < image_files.size(); i++)
    {
        cv::Mat image = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
        cv::Mat gtDepth = cv::imread(depth_files[i], cv::IMREAD_GRAYSCALE);
        SE3f gtPose = poses[i].inverse();

        if (std::is_same<imageType, uchar>::value)
            image.convertTo(image, CV_8UC1);
        else if (std::is_same<imageType, int>::value)
            image.convertTo(image, CV_32SC1);
        else if (std::is_same<imageType, float>::value)
            image.convertTo(image, CV_32FC1);

        gtDepth.convertTo(gtDepth, CV_32FC1);
        gtDepth /= dataset.getDepthFactor();
        gtDepth *= 100.0;

        dataCPU<imageType> imageData(w, h, 0);
        imageData.set((imageType *)image.data);

        dataCPU<float> gtDepthData(w, h, 0);
        gtDepthData.set((float *)gtDepth.data);

        if (i == 0)
        {
            lastEstimatedGlobalPose = gtPose;
            kframe = keyFrameCPU(imageData, vec2f(0.0, 0.0), gtPose, 1.0);
            kframe.initGeometryFromDepth(gtDepthData, dataCPU<float>(w, h, 1.0 / mesh_vo::mapping_param_initial_var), cam);
            continue;
        }

        frameCPU frame(imageData, i);
        frame.setGlobalPose(lastEstimatedGlobalPose);
        frame.setLocalPose(kframe.globalPoseToLocal(lastEstimatedGlobalPose));

        auto startTime = std::chrono::high_resolution_clock::now();
        for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_fin_lvl; lvl--)
        {
            optimizer.init(frame, kframe, cam, lvl);
            while (!optimizer.converged())
            {
                optimizer.step(frame, kframe, cam, lvl);
            }
        }
        auto endTime = std::chrono::high_resolution_clock::now();

        /*
         dataMipMapCPU<float> error_buffer(w, h, -1.0);
         renderer.renderResidualParallel(kframe, frame, error_buffer, cam, 1);
         show(error_buffer.get(1), "Error");

         dataMipMapCPU<float> depth_buffer(w, h, -1.0);
         renderer.renderDepthParallel(kframe, frame.getLocalPose(), depth_buffer, cam, 1);
         dataCPU d = depth_buffer.get(1);
         d.invert();
         show(d, "Depth");
        */

        lastEstimatedGlobalPose = kframe.localPoseToGlobal(frame.getLocalPose());

        SE3f gtLocalPose = kframe.globalPoseToLocal(gtPose);
        std::array<float, 2> error = computeSE3Error(frame.getLocalPose(), gtLocalPose);

        accProcessingTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        accTranslationError += error[0];
        accRotationError += error[1];
        framesProcessedCounter++;

        // change keyframe logic
        float keyframeViewAngle = kframe.meanViewAngle(SE3f(), frame.getLocalPose());

        dataMipMapCPU<float> image_buffer(w, h, -1);
        image_buffer.setToNoData(1);
        renderer.renderImageParallel(kframe, frame.getLocalPose(), image_buffer, cam, 1);
        float pnodata = image_buffer.get(1).getPercentNoData();
        float viewPercent = 1.0 - pnodata;

        if (viewPercent < mesh_vo::min_view_perc || keyframeViewAngle > mesh_vo::key_max_angle)
        {
            kframe = keyFrameCPU(imageData, vec2f(0.0, 0.0), gtPose, 1.0);
            kframe.initGeometryFromDepth(gtDepthData, dataCPU<float>(w, h, 1.0 / mesh_vo::mapping_param_initial_var), cam);
        }
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