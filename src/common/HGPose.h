#pragma once

#include <Eigen/Core>

class HGPose
{
public:
    HGPose()
    {
        H.setZero();
        G.setZero();
        count = 0;
    }

    void setZero()
    {
        H.setZero();
        G.setZero();
        count = 0;
    }

    HGPose operator+(HGPose a)
    {
        HGPose sum;
        sum.H = H + a.H;
        sum.G = G + a.G;
        sum.count = count + a.count;
        return sum;
    }

    void operator+=(HGPose a)
    {
        H += a.H;
        G += a.G;
        count += a.count;
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
    int count;

private:

};
