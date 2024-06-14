#pragma once

#include <Eigen/Core>

//struct HJPose{
//    float J[6];
//    float H[21];
//    float error;
//    int cout;
//};

class HGPose
{
public:
    HGPose()
    {
        H_pose.setZero();
        G_pose.setZero();
        error = 0.0;
        count = 0;
    }

    HGPose operator+(HGPose _pose)
    {
        HGPose _p;
        _p.H_pose = H_pose + _pose.H_pose;
        _p.G_pose = G_pose + _pose.G_pose;
        _p.error = error + _pose.error;
        _p.count = count + _pose.count;

        return _p;
    }

    void operator+=(HGPose _pose)
    {
        H_pose += _pose.H_pose;
        G_pose += _pose.G_pose;
        error += _pose.error;
        count += _pose.count;
    }

    void operator=(HGPose _pose)
    {
        H_pose = _pose.H_pose;
        G_pose = _pose.G_pose;
        error = _pose.error;
        count = _pose.count;
    }

    Eigen::Matrix<float, 6, 6> H_pose;
    Eigen::Matrix<float, 6, 1> G_pose;

    float error;
    float count;

private:

};
