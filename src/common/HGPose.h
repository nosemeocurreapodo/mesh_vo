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
    }

    HGPose operator+(HGPose _pose)
    {
        HGPose _p;
        _p.H = H + _pose.H;
        _p.G = G + _pose.G;

        return _p;
    }

    void operator+=(HGPose _pose)
    {
        H += _pose.H;
        G += _pose.G;
    }

    /*
    void operator=(HGPose _pose)
    {
        H = _pose.H;
        G = _pose.G;
    }
    */

    Eigen::Matrix<float, 6, 6> H;
    Eigen::Matrix<float, 6, 1> G;

private:

};
