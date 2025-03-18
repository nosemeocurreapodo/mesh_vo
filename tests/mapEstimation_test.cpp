#include <gtest/gtest.h>

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <iostream>
#include <chrono>

#include "opencv2/opencv.hpp"

#include "common.h"
#include "optimizers/mapOptimizerCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/OpenCVDebug.h"

// Test to ensure PoseEstimator correctly computes the pose
TEST(MapEstimatorTest, ComputeMapFromInitial)
{
    const long long acceptableTimeMs = 30;
    const float errorThreshold = 0.023; // best = 0.022444;

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

    std::chrono::milliseconds accProcessingTime = std::chrono::milliseconds(0);
    float accError = 0;
    int framesProcessedCounter = 0;

    mapOptimizerCPU optimizer(w, h, false);
    renderCPU renderer(w, h);
    keyFrameCPU kframe;
    std::vector<frameCPU> frames;

    for (unsigned int i = 0; i < image_files.size(); i ++)
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

        if(i == 0)
        {
            kframe = keyFrameCPU(imageData, vec2f(0.0, 0.0), gtPose, 1.0);
            kframe.initGeometryVerticallySmooth(cam);
            continue;
        }

        frameCPU frame(imageData, i);
        frame.setGlobalPose(gtPose);
        frame.setLocalPose(kframe.globalPoseToLocal(gtPose));

        float minViewAngle = M_PI;
        for(int i = 0; i < frames.size(); i++)
        {
            float viewAngle = kframe.meanViewAngle(frame.getLocalPose(), frames[i].getLocalPose());
            if(viewAngle < minViewAngle)
                minViewAngle = viewAngle;
        }

        if(minViewAngle < mesh_vo::last_min_angle)
            continue;

        frames.push_back(frame);
        if (frames.size() > mesh_vo::num_frames)
        {
            frames.erase(frames.begin());
        }
        else
        {
            continue;
        }

        int kframeIndex = frames.size() / 2;
        std::vector<frameCPU> oframes = frames;
        oframes.erase(oframes.begin() + kframeIndex);

        kframe = keyFrameCPU(frames[kframeIndex].getRawImage(0), vec2f(0.0, 0.0), frames[kframeIndex].getGlobalPose(), 1.0);
        kframe.initGeometryVerticallySmooth(cam);

        for (int i = 0; i < oframes.size(); i++)
        {
            oframes[i].setLocalPose(kframe.globalPoseToLocal(oframes[i].getGlobalPose()));
        }

        auto startTime = std::chrono::high_resolution_clock::now();
        for (int lvl = mesh_vo::mapping_ini_lvl; lvl >= mesh_vo::mapping_fin_lvl; lvl--)
        {
            optimizer.init(oframes, kframe, cam, lvl);
            while (!optimizer.converged())
            {
                optimizer.step(oframes, kframe, cam, lvl);
            }
        }
        auto endTime = std::chrono::high_resolution_clock::now();

        /*
        dataMipMapCPU<float> error_buffer(w, h, -1.0);
        renderer.renderResidualParallel(kframe, oframes[oframes.size() - 1], error_buffer, cam, 1);
        show(error_buffer.get(1), "Error");

        dataMipMapCPU<float> depth_buffer(w, h, -1.0);
        renderer.renderDepthParallel(kframe, oframes[oframes.size() - 1].getLocalPose(), depth_buffer, cam, 1);
        dataCPU d = depth_buffer.get(1);
        d.invert();
        show(d, "Depth");
        */

        dataMipMapCPU<float> estMipMapDepthData(w, h, -1);
        renderer.renderDepthParallel(kframe, oframes[oframes.size() - 1].getLocalPose(), estMipMapDepthData, cam, 0);
        float error = computeImageError(estMipMapDepthData.get(0), gtDepthData);

        accProcessingTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        accError += error;
        framesProcessedCounter++;
    }

    auto meanDuration = accProcessingTime.count() / framesProcessedCounter;
    float meanError = accError / framesProcessedCounter;
    std::cout << "Mean processing time " << meanDuration << " ms" << std::endl;
    std::cout << "Mean error " << meanError << " ms" << std::endl;

    // The test passes if the error is below the threshold
    EXPECT_LT(meanError, errorThreshold)
        << "mean error (" << meanError
        << ") exceeds the acceptable threshold (" << errorThreshold << ").";

    // EXPECT_LE(durationMs, acceptableTimeMs)
    //     << "Pose estimation took " << durationMs << "ms, which exceeds the acceptable threshold of "
    //     << acceptableTimeMs << "ms.";
}