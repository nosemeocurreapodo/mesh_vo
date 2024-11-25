#pragma once

#include <Eigen/Core>

class HGEigenDense
{
public:
    HGEigenDense(int size)
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
    void add(type J, float error, float weight)
    {
        count++;

        // G += J * error;
        // H += J * J.transpose();

        for (int i = 0; i < type::size(); i++)
        {
            G(i) += J(i) * error * weight;
            H(i, i) += J(i) * J(i) * weight;

            for (int j = i + 1; j < type::size(); j++)
            {
                float jj = J(i) * J(j) * weight;
                H(i, j) += jj;
                H(j, i) += jj;
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
