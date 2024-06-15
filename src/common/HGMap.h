#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

//struct HJPose{
//    float J[6];
//    float H[21];
//    float error;
//    int cout;
//};

class HGMap
{
public:
    HGMap()
    {
        //H_depth.setZero();
        //G_depth.setZero();
        //count_depth.setZero();

        H_depth = Eigen::SparseMatrix<float>(VERTEX_HEIGH*VERTEX_WIDTH, VERTEX_HEIGH*VERTEX_WIDTH);
        G_depth = Eigen::VectorXf::Zero(VERTEX_HEIGH*VERTEX_WIDTH);
        G_count = Eigen::VectorXf::Zero(VERTEX_HEIGH*VERTEX_WIDTH);

        //inc_depth = Eigen::VectorXf::Zero(VERTEX_HEIGH*VERTEX_WIDTH);
        error = 0.0;
        count = 0;
    }

    Eigen::SparseMatrix<float> H_depth;
    Eigen::VectorXf G_depth;
    Eigen::VectorXf G_count;

    float error;
    int count;

private:

};
