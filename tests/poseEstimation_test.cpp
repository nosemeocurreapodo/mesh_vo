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
    const float translationErrorThreshold = 0.02;
    const float rotationErrorThreshold = 0.002;

    // Validate that the pose is within expected bounds
    // EXPECT_NEAR(pose.translation.x, 0.0, 0.001);
    // EXPECT_NEAR(pose.translation.y, 0.0, 0.001);
    // EXPECT_NEAR(pose.translation.z, 0.0, 0.001);

    //load_dataset_tum_rgbd dataset;
    load_dataset_icl_nuim dataset;

    std::vector<std::string> image_files = dataset.getImageFiles();
    std::vector<std::string> depth_files = dataset.getDepthFiles();
    std::vector<SE3f> poses = dataset.getPoses();
    std::vector<double> timestamps = dataset.getTimestamps();
    cameraType cam = dataset.getCamera();
    int w = dataset.getWidth();
    int h = dataset.getHeight();

    poseOptimizerCPU optimizer(w, h, true);
    renderCPU renderer(w, h);

    keyFrameCPU kframe;
    frameCPU frame;

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
            kframe = keyFrameCPU(imageData, vec2f(0.0, 0.0), gtPose, 1.0);
            kframe.initGeometryFromDepth(gtDepthData, dataCPU<float>(w, h, 1.0 / mesh_vo::mapping_param_initial_var), cam);
        }
        else
        {
            SE3f gtLocalPose = kframe.globalPoseToLocal(gtPose);
            //SE3f gtLocalPose = gtPose * kframe.getGlobalPose().inverse();
            //SE3f gtLocalPose = gtPose.inverse() * kframe.getGlobalPose();
            //SE3f gtLocalPose = kframe.getGlobalPose() * gtPose.inverse();
            //SE3f gtLocalPose = kframe.getGlobalPose().inverse() * gtPose;

            // frame.setLocalPose(gtLocalPose);

            auto startTime = std::chrono::high_resolution_clock::now();
            for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_fin_lvl; lvl--)
            {
                optimizer.init(frame, kframe, cam, lvl);
                while (!optimizer.converged())
                {
                    optimizer.step(frame, kframe, cam, lvl);

                    //dataMipMapCPU<float> error(w, h, -1.0);
                    //renderer.renderResidualParallel(kframe, frame, error, cam, lvl);
                    //show(error.get(lvl), "Error");
                }
            }
            auto endTime = std::chrono::high_resolution_clock::now();
            auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

            //EXPECT_LE(durationMs, acceptableTimeMs)
            //    << "Pose estimation took " << durationMs << "ms, which exceeds the acceptable threshold of "
            //    << acceptableTimeMs << "ms.";

            std::array<float, 2> error = computeSE3Error(frame.getLocalPose(), gtLocalPose);

            std::cout << "translation error " << error[0] << " rotation error " << error[1] << std::endl;

            // The test passes if the error is below the threshold
            EXPECT_LT(error[0], translationErrorThreshold)
                << "Translation estimation error (" << error[0]
                << ") exceeds the acceptable threshold (" << translationErrorThreshold << ").";

            EXPECT_LT(error[1], rotationErrorThreshold)
                << "Translation estimation error (" << error[1]
                << ") exceeds the acceptable threshold (" << rotationErrorThreshold << ").";
        }
    }
}