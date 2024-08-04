#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

class HGEigen
{
public:
    HGEigen(int size)
    {
        H = Eigen::SparseMatrix<float>(size, size);
        G = Eigen::VectorXf::Zero(size);
        count = 0;
    }

    /*
    void setZero()
    {
        H.setZero();
        G.setZero();
        count = 0;
    }
    */
    /*
    HGEigen operator+(HGEigen a)
    {
        HGEigen sum;
        sum.H = H + a.H;
        sum.G = G + a.G;
        sum.count = count + a.count;
        return sum;
    }

    void operator+=(HGMap &a)
    {
        H += a.H;
        G += a.G;
        count += a.count;
    }
    */

    /*
    void operator=(HGPose _pose)
    {
        H = _pose.H;
        G = _pose.G;
    }
    */

    Eigen::SparseMatrix<float> H;
    Eigen::VectorXf G;
    int count;

private:
};
