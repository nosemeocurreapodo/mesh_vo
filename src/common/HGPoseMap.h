#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "params.h"

class HGPoseMap
{
public:
    HGPoseMap(int frames)
    {
        num_frames = frames;

        H = Eigen::SparseMatrix<float>(num_frames * 6 + VERTEX_HEIGH * VERTEX_WIDTH, num_frames * 6 + VERTEX_HEIGH * VERTEX_WIDTH);
        G = Eigen::VectorXf::Zero(num_frames * 6 + VERTEX_HEIGH * VERTEX_WIDTH);
        count = Eigen::VectorXf::Zero(num_frames * 6 + VERTEX_HEIGH * VERTEX_WIDTH);
    }

    Eigen::SparseMatrix<float> H;
    Eigen::VectorXf G;
    Eigen::VectorXf count;

    int num_frames;

private:
};
