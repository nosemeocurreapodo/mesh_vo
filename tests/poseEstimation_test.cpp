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
    const float translationErrorThreshold = 0.06;
    const float rotationErrorThreshold = 0.006;

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

    poseOptimizerCPU optimizer(w, h, false);
    renderCPU renderer(w, h);

    keyFrameCPU kframe;
    frameCPU frame;

    SE3f lastEstimatedGlobalPose;

    std::chrono::milliseconds accProcessingTime = std::chrono::milliseconds(0);
    float accTranslationError = 0;
    float accRotationError = 0;
    int framesProcessedCounter = 0;

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

        frame = frameCPU(imageData, i);

        if (i == 0)
        {
            lastEstimatedGlobalPose = gtPose;
        }

        if (i % 20 == 0)
        {
            kframe = keyFrameCPU(imageData, vec2f(0.0, 0.0), gtPose, 1.0);
            kframe.initGeometryFromDepth(gtDepthData, dataCPU<float>(w, h, 1.0 / mesh_vo::mapping_param_initial_var), cam);
        }
        else
        {
            SE3f gtLocalPose = kframe.globalPoseToLocal(gtPose);

            frame.setGlobalPose(lastEstimatedGlobalPose);
            frame.setLocalPose(kframe.globalPoseToLocal(lastEstimatedGlobalPose));

            auto startTime = std::chrono::high_resolution_clock::now();
            for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_fin_lvl; lvl--)
            {
                optimizer.init(frame, kframe, cam, lvl);
                while (!optimizer.converged())
                {
                    optimizer.step(frame, kframe, cam, lvl);

                    // dataMipMapCPU<float> error(w, h, -1.0);
                    // renderer.renderResidualParallel(kframe, frame, error, cam, lvl);
                    // show(error.get(lvl), "Error");
                }
            }
            auto endTime = std::chrono::high_resolution_clock::now();

            lastEstimatedGlobalPose = kframe.localPoseToGlobal(frame.getLocalPose());

            std::array<float, 2> error = computeSE3Error(frame.getLocalPose(), gtLocalPose);

            accProcessingTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            accTranslationError += error[0];
            accRotationError += error[1];
            framesProcessedCounter++;

            // std::cout << "translation error " << error[0] << " rotation error " << error[1] << std::endl;

            // The test passes if the error is below the threshold
            EXPECT_LT(error[0], translationErrorThreshold)
                << "Translation estimation error (" << error[0]
                << ") exceeds the acceptable threshold (" << translationErrorThreshold << ").";

            EXPECT_LT(error[1], rotationErrorThreshold)
                << "Translation estimation error (" << error[1]
                << ") exceeds the acceptable threshold (" << rotationErrorThreshold << ").";
        }
    }

    auto meanDuration = accProcessingTime.count() / framesProcessedCounter;
    float meanTranslationError = accTranslationError / framesProcessedCounter;
    float meanRotationError = accRotationError / framesProcessedCounter;
    std::cout << "Mean processing time " << meanDuration << " ms" << std::endl;
    std::cout << "Mean translation error " << meanTranslationError << " ms" << std::endl;
    std::cout << "Mean rotation error " << meanRotationError << " ms" << std::endl;

    // EXPECT_LE(durationMs, acceptableTimeMs)
    //     << "Pose estimation took " << durationMs << "ms, which exceeds the acceptable threshold of "
    //     << acceptableTimeMs << "ms.";
}