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
        H.setZero();
        G.setZero();
        error = 0.0;
        count = 0;
    }

    HGPose operator+(HGPose _pose)
    {
        HGPose _p;
        _p.H = H + _pose.H;
        _p.G = G + _pose.G;
        _p.error = error + _pose.error;
        _p.count = count + _pose.count;

        return _p;
    }

    void operator+=(HGPose _pose)
    {
        H += _pose.H;
        G += _pose.G;
        error += _pose.error;
        count += _pose.count;
    }

    void operator=(HGPose _pose)
    {
        H = _pose.H;
        G = _pose.G;
        error = _pose.error;
        count = _pose.count;
    }

    Eigen::Matrix<float, 6, 6> H;
    Eigen::Matrix<float, 6, 1> G;

    float error;
    float count;

private:

};
