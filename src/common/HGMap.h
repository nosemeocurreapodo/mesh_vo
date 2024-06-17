#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "params.h"

class HGMap
{
public:
    HGMap()
    {
        // H_depth.setZero();
        // G_depth.setZero();
        // count_depth.setZero();

        H = Eigen::SparseMatrix<float>(VERTEX_HEIGH * VERTEX_WIDTH, VERTEX_HEIGH * VERTEX_WIDTH);
        G = Eigen::VectorXf::Zero(VERTEX_HEIGH * VERTEX_WIDTH);
        count = Eigen::VectorXf::Zero(VERTEX_HEIGH * VERTEX_WIDTH);
    }

    Eigen::SparseMatrix<float> H;
    Eigen::VectorXf G;
    Eigen::VectorXf count;

private:
};
