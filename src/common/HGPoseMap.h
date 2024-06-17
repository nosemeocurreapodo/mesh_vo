#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "params.h"

//struct HJPose{
//    float J[6];
//    float H[21];
//    float error;
//    int cout;
//};

class HGPoseMap
{
public:
    HGPoseMap()
    {
        //H_depth.setZero();
        //G_depth.setZero();
        //count_depth.setZero();

        H = Eigen::SparseMatrix<float>(MAX_FRAMES*6 + VERTEX_HEIGH*VERTEX_WIDTH, MAX_FRAMES*6 + VERTEX_HEIGH*VERTEX_WIDTH);
        G = Eigen::VectorXf::Zero(MAX_FRAMES*6 + VERTEX_HEIGH*VERTEX_WIDTH);
        count = Eigen::VectorXf::Zero(MAX_FRAMES*6 + VERTEX_HEIGH*VERTEX_WIDTH);

        //inc_depth = Eigen::VectorXf::Zero(VERTEX_HEIGH*VERTEX_WIDTH);
        error = 0.0;
        count = 0;
    }

    Eigen::SparseMatrix<float> H;
    Eigen::VectorXf G;
    Eigen::VectorXf count;

    float error;
    int count;

private:

};
