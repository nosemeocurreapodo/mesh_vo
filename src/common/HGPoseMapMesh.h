#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "params.h"

#define VERTEX_WIDTH 32
#define VERTEX_HEIGHT 32

class HGPoseMapMesh
{
public:
    HGPoseMapMesh()
    {
        H = Eigen::SparseMatrix<float>(NUM_FRAMES * 6 + VERTEX_HEIGHT * VERTEX_WIDTH, NUM_FRAMES * 6 + VERTEX_HEIGHT * VERTEX_WIDTH);
        G = Eigen::VectorXf::Zero(NUM_FRAMES * 6 + VERTEX_HEIGHT * VERTEX_WIDTH);
        G_count = Eigen::VectorXf::Zero(NUM_FRAMES * 6 + VERTEX_HEIGHT * VERTEX_WIDTH);
        //H.setZero();
        //G.setZero();
        //G_count.setZero();
        count = 0.0;
    }

    void setZero()
    {
        H.setZero();
        G.setZero();
        G_count.setZero();
        count = 0;
    }

    HGPoseMapMesh operator+(HGPoseMapMesh a)
    {
        HGPoseMapMesh sum;
        sum.H = H + a.H;
        sum.G = G + a.G;
        sum.G_count = G + a.G_count;
        sum.count = count + a.count;
        return sum;
    }

    void operator+=(HGPoseMapMesh a)
    {
        H += a.H;
        G += a.G;
        G_count += a.G_count;
        count += a.count;
    }

    Eigen::SparseMatrix<float> H;
    Eigen::VectorXf G;
    Eigen::VectorXf G_count;

    //Eigen::SparseMatrix<float, NUM_FRAMES * 6 + VERTEX_HEIGHT * VERTEX_WIDTH, NUM_FRAMES * 6 + VERTEX_HEIGHT * VERTEX_WIDTH> H;
    //Eigen::Matrix<float, NUM_FRAMES * 6 + VERTEX_HEIGHT * VERTEX_WIDTH, 1> G;
    //Eigen::Matrix<float, NUM_FRAMES * 6 + VERTEX_HEIGHT * VERTEX_WIDTH, 1> G_count;

    int count;

private:
};