#include <gtest/gtest.h>

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <iostream>
#include <chrono>

#include "opencv2/opencv.hpp"

#include "optimizers/poseOptimizerCPU.h"
#include "test/common.h"

// Test to ensure PoseEstimator correctly computes the pose
TEST(PoseEstimatorTest, ComputePose)
{
    const long long acceptableTimeMs = 50;
    const float errorThreshold = 0.1;

    // PoseEstimator estimator;
    //  Simulated input data
    // std::vector<cv::Point2f> imagePoints = {/* simulated data */};
    // std::vector<cv::Point3f> worldPoints = {/* corresponding world points */};

    // auto pose = estimator.computePose(imagePoints, worldPoints);

    // Validate that the pose is within expected bounds
    // EXPECT_NEAR(pose.translation.x, 0.0, 0.001);
    // EXPECT_NEAR(pose.translation.y, 0.0, 0.001);
    // EXPECT_NEAR(pose.translation.z, 0.0, 0.001);

    int w = 640;
    int h = 480;
    float fx = 525.0;
    float fy = 525.0;
    float cx = 319.5;
    float cy = 239.5;

    // open image files: first try to open as file.
    std::string images_path = std::string(TEST_DATA_DIR) + "/traj3n_frei_png_part/rgb";
    std::vector<std::string> image_files;

    if (getdir(images_path, image_files) >= 0)
    {
        printf("found %d image files in folder %s!\n", (int)image_files.size(), images_path.c_str());
    }
    else
    {
        printf("could not load file list! wrong path / file?\n");
    }

    std::vector<std::string> depth_files;
    std::string depths_path = std::string(TEST_DATA_DIR) + "/traj3n_frei_png_part/depth";

    if (getdir(depths_path, depth_files) >= 0)
    {
        printf("found %d image files in folder %s!\n", (int)depth_files.size(), depths_path.c_str());
    }
    else
    {
        printf("could not load file list! wrong path / file?\n");
    }

    std::string poses_path = std::string(TEST_DATA_DIR) + "/traj3n_frei_png_part/traj3n.gt.freiburg";
    std::vector<SE3f> poses = getPoses(poses_path);

    cameraType cam(fx, fy, cx, cy, w, h);

    poseOptimizerCPU optimizer(w, h, false);

    keyFrameCPU kframe;
    frameCPU frame;

    for (unsigned int i = 0; i < image_files.size(); i++)
    {
        cv::Mat image = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
        cv::Mat depth = cv::imread(depth_files[i]); //, cv::IMREAD_GRAYSCALE);
        SE3f pose = poses[i];

        if (std::is_same<imageType, uchar>::value)
            image.convertTo(image, CV_8UC1);
        else if (std::is_same<imageType, int>::value)
            image.convertTo(image, CV_32SC1);
        else if (std::is_same<imageType, float>::value)
            image.convertTo(image, CV_32FC1);

        depth.convertTo(depth, CV_32FC1);
        depth /= 5000.0;

        dataCPU<imageType> imageData(w, h, 0);
        imageData.set((imageType *)image.data);

        dataCPU<float> depthData(w, h, 0);
        depthData.set((float *)depth.data);

        frame = frameCPU(imageData, i);

        if (i == 0)
        {
            kframe = keyFrameCPU(imageData, vec2f(0.0, 0.0), pose, 1.0);
            kframe.initGeometryFromDepth(depthData, dataCPU<float>(w, h, 1.0 / mesh_vo::mapping_param_initial_var), cam);
        }
        else
        {
            SE3f gtLocalPose = pose * kframe.getGlobalPose().inverse();

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
            auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

            EXPECT_LE(durationMs, acceptableTimeMs)
                << "Pose estimation took " << durationMs << "ms, which exceeds the acceptable threshold of "
                << acceptableTimeMs << "ms.";

            float error = computeSE3Error(frame.getLocalPose(), gtLocalPose);

            // The test passes if the error is below the threshold
            EXPECT_LT(error, errorThreshold)
                << "Pose estimation error (" << error
                << ") exceeds the acceptable threshold (" << errorThreshold << ").";
        }
    }
}