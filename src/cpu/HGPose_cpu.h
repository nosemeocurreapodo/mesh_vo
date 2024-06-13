#pragma once

#include "sophus/se3.hpp"

//struct HJPose{
//    float J[6];
//    float H[21];
//    float error;
//    int cout;
//};

class HGPose_cpu
{
public:
    HGPose_cpu()
    {
        H_pose.setZero();
        G_pose.setZero();
        error = 0.0;
        count = 0;
    }

    HGPose_cpu operator+(HGPose_cpu _pose)
    {
        HGPose_cpu _p;
        _p.H_pose = H_pose + _pose.H_pose;
        _p.G_pose = G_pose + _pose.G_pose;
        _p.error = error + _pose.error;
        _p.count = count + _pose.count;

        return _p;
    }

    void operator+=(HGPose_cpu _pose)
    {
        H_pose += _pose.H_pose;
        G_pose += _pose.G_pose;
        error += _pose.error;
        count += _pose.count;
    }

    void operator=(HGPose_cpu _pose)
    {
        H_pose = _pose.H_pose;
        G_pose = _pose.G_pose;
        error = _pose.error;
        count = _pose.count;
    }

private:
    Eigen::Matrix<float, 6, 6> H_pose;
    Eigen::Matrix<float, 6, 1> G_pose;

    float error;
    float count;
};
