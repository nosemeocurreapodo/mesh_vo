#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/camera.h"
#include "common/common.h"

class triangle
{
public:
    triangle(std::vector<std::array<float, 3>> &vertices, std::array<unsigned int, 3> &indices, bool isRayIdepth = true)
    {
        for (int i = 0; i < 3; i++)
        {
            vertex[i](0) = vertices[indices[i]][0];
            vertex[i](1) = vertices[indices[i]][1];
            vertex[i](2) = vertices[indices[i]][2];

            if (isRayIdepth)
                vertex[i] = fromRayIdepthToVertex(vertex[i]);
        }
    };

    void transform(Sophus::SE3f &pose)
    {
        for (int v_id = 0; v_id < 3; v_id++)
        {
            vertex[v_id] = pose * vertex[v_id];
        }
    };

    void project(camera &cam, int lvl)
    {
        for (int v_id = 0; v_id < 3; v_id++)
        {
            pix[v_id] = cam.project(vertex[v_id], lvl);
        }
    };

    void computeTinv()
    {
        Eigen::Matrix2f T;
        T(0, 0) = pix[0](0) - pix[2](0);
        T(0, 1) = pix[1](0) - pix[2](0);
        T(1, 0) = pix[0](1) - pix[2](1);
        T(1, 1) = pix[1](1) - pix[2](1);
        T_inv = T.inverse();
    };

    void computeBarycentric(Eigen::Vector2f p)
    {
        Eigen::Vector2f lambda = T_inv * (p - pix[2]);
        barycentric = Eigen::Vector3f(lambda(0), lambda(1), 1 - lambda(0) - lambda(1));
    };

    bool isBarycentricOk()
    {
        if (barycentric(0) < 0.0 || barycentric(1) < 0.0 || barycentric(2) < 0.0)
            return false;
        return true;
    };

    template <typename Type>
    Type interpolate(std::array<Type, 3> data)
    {
        return barycentric(0) * data[0] + barycentric(1) * data[1] + barycentric(2) * data[2];
    };

    std::array<Eigen::Vector2f, 2> getMinMax()
    {
        std::array<Eigen::Vector2f, 2> minmax;
        minmax[0](0) = std::min(std::min(pix[0](0), pix[1](0)), pix[2](0));
        minmax[0](1) = std::min(std::min(pix[0](1), pix[1](1)), pix[2](1));
        minmax[1](0) = std::max(std::max(pix[0](0), pix[1](0)), pix[2](0));
        minmax[1](1) = std::max(std::max(pix[0](1), pix[1](1)), pix[2](1));

        return minmax;
    };

    std::array<Eigen::Vector3f, 3> vertex;
    std::array<Eigen::Vector2f, 3> pix;
    Eigen::Matrix2f T_inv;
    Eigen::Vector3f barycentric;

private:
};