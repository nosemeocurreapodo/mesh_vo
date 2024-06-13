#pragma once

#include <opencv2/core.hpp>

#include <iostream>
#include <fstream>

//#include "Common/se3.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "sophus/se3.hpp"

#include "utils/tictoc.h"

#include "cpu/data_cpu.h"
#include "cpu/frame_cpu.h"
#include "cpu/HGPose_cpu.h"

class mesh_vo
{
public:
    mesh_vo(float _fx, float _fy, float _cx, float _cy, int _width, int _height);

    void initWithRandomIdepth(cv::Mat _keyFrame, Sophus::SE3f _pose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));
    void initWithIdepth(cv::Mat _keyFrame, cv::Mat _idepth, Sophus::SE3f _pose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));

    void visual_odometry(cv::Mat _frame);
    void localization(cv::Mat _frame);
    void mapping(cv::Mat _frame, Sophus::SE3f _globalPose);

private:

    //frames
    frame keyframeData;
    frame lastframeData;
    frame frameDataStack[MAX_FRAMES];

    //data
    data vertexIdData;
    data debugData;
    data view3DData;



    Eigen::SparseMatrix<float> H_depth;
    Eigen::VectorXf J_depth;
    Eigen::VectorXf inc_depth;
    Eigen::VectorXi count_depth;

    Eigen::MatrixXf H_joint;
    Eigen::VectorXf J_joint;
    Eigen::VectorXf inc_joint;
    Eigen::VectorXi count_joint;

    void changeKeyframe(frame &newkeyFrame, int lvl, float min_occupancy);
    void addFrameToStack(frame &_frame);
    void setTriangles();

    void optPose(frame &_frame);
    void optPose2(frame &_frame);
    void optMapJoint();
    //void optMapVertex();
    void optPoseMapJoint();



};
