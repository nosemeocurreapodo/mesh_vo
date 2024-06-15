#pragma once

#include <opencv2/core.hpp>

#include <iostream>
#include <fstream>

// #include "Common/se3.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "sophus/se3.hpp"

#include "utils/tictoc.h"

#include "scene/rayDepthMeshScene.h"
#include "common/camera.h"
#include "cpu/frame_cpu.h"

#include "cpu/platform_cpu.h"

#include "common/HGPose.h"

class meshVO
{
public:
    meshVO(float _fx, float _fy, float _cx, float _cy, int _width, int _height);

    void initScene(cv::Mat frame, Sophus::SE3f pose);
    void initScene(cv::Mat frame, cv::Mat idepth, Sophus::SE3f pose);

    void visualOdometry(cv::Mat frame);
    void localization(cv::Mat frame);
    void mapping(cv::Mat _frame, Sophus::SE3f pose);

private:

    frameCpu keyframe;
    frameCpu lastframe;
    rayDepthMeshScene scene;

    // compute platforms
    platformCpu cpu;

    // functions
    void optPose(frameCpu &frame);
    void optMap();
    // void optMapVertex();
    void optPoseMap();

    // mesh regularization
    float errorMeshRegu(rayDepthMeshScene &scene);
    void HGMeshRegu(HGMap &hgmap, rayDepthMeshScene &scene);
};
