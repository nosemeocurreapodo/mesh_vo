#pragma once

#include <opencv2/core.hpp>

#include <iostream>
#include <fstream>

//#include "Common/se3.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "sophus/se3.hpp"

#include "utils/tictoc.h"

#include "scene/scene_mesh.h"
#include "common/camera.h"
#include "cpu/frame_cpu.h"

#include "cpu/platform_cpu.h"

#include "common/HGPose.h"

class meshVO
{
public:
    meshVO(float _fx, float _fy, float _cx, float _cy, int _width, int _height);

    void visualOdometry(cv::Mat frame);
    void localization(cv::Mat frame);
    void mapping(cv::Mat _frame, Sophus::SE3f _globalPose);

private:

    camera cam;
    frameCpu keyframe;
    frameCpu lastframe;
    sceneMesh scene;

    //compute platforms
    platformCpu cpu;

    //optimization data
    Eigen::SparseMatrix<float> H_depth;
    Eigen::VectorXf J_depth;
    Eigen::VectorXf inc_depth;
    Eigen::VectorXi count_depth;

    Eigen::MatrixXf H_joint;
    Eigen::VectorXf J_joint;
    Eigen::VectorXf inc_joint;
    Eigen::VectorXi count_joint;

    //functions

    void optPose(frameCpu &frame);
    void optMap();
    //void optMapVertex();
    void optPoseMap();


    //mesh regularization
    float errorMesh();
    void addHGMesh();
};
