#pragma once

#include "sophus/se3.hpp"

class HJPose
{
public:

    HJPose()
    {
        H_pose.setZero();
        J_pose.setZero();
        error = 0.0;
        count = 0;
    }

     HJPose operator+(HJPose _pose)
     {
         HJPose _p;
         _p.H_pose = H_pose + _pose.H_pose;
         _p.J_pose = J_pose + _pose.J_pose;
         _p.error  = error + _pose.error;
         _p.count  = count + _pose.count;

         return _p;
     }

     void operator+=(HJPose _pose)
     {
         H_pose += _pose.H_pose;
         J_pose += _pose.J_pose;
         error += _pose.error;
         count += _pose.count;
     }

     void operator=(HJPose _pose)
     {
         H_pose = _pose.H_pose;
         J_pose = _pose.J_pose;
         error = _pose.error;
         count = _pose.count;
     }

     Eigen::Matrix<float, 6, 6> H_pose;
     Eigen::Matrix<float, 6, 1> J_pose;

    float error;
    float count;

private:

};
