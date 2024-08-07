#pragma once

#include <Eigen/Core>

class HGEigenDense
{
public:
    HGEigenDense(int size = 6)
    {
        H = Eigen::MatrixXf::Zero(size, size);
        G = Eigen::VectorXf::Zero(size);

        count = 0;
    }

    void setZero()
    {
        H.setZero();
        G.setZero();
        count = 0;
    }

    template <typename type>
    void add(type J, float error)
    {
        count++;

        //G += J * error;
        //H += J * J.transpose();

        for (int i = 0; i < J.size(); i++)
        {
            G(i) += J(i)*error;
            for (int j = 0; j < J.size(); j++)
            {
                float jj = J(i) * J(j);
                H(i, j) += jj;
            }
        }
    }

    HGEigenDense operator+(HGEigenDense &a)
    {
        HGEigenDense sum(G.size());
        sum.H = H + a.H;
        sum.G = G + a.G;
        sum.count = count + a.count;
        return sum;
    }

    void operator+=(HGEigenDense &a)
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

    Eigen::MatrixXf getH()
    {
        return H / count;
    }

    Eigen::VectorXf getG()
    {
        return G / count;
    }

private:
    Eigen::MatrixXf H;
    Eigen::VectorXf G;

    // Eigen::Matrix<float, 6, 6> H;
    // Eigen::Matrix<float, 6, 1> G;
    int count;
};
