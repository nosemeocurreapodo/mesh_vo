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
        H_depth.setZero();
        G_depth.setZero();
        count_depth.setZero();
    }

    Eigen::SparseMatrix<float> H_depth;
    Eigen::VectorXf G_depth;
    Eigen::VectorXi count_depth;

private:

};
