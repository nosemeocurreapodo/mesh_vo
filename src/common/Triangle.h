#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/Vertex.h"
#include "common/camera.h"
#include "common/common.h"

class Triangle
{
public:

    Triangle(Vertex &vert1, Vertex &vert2, Vertex &vert3, unsigned int i)
    {
        vertices[0] = &vert1;
        vertices[1] = &vert2;
        vertices[2] = &vert3;
        id = i;
    };

    bool isBackFace()
    {
        Eigen::Vector3f f_tri_nor = (vertices[0]->position - vertices[2]->position).cross(vertices[0]->position - vertices[1]->position);
        // back-face culling
        float point_dot_normal = vertices[0]->position.dot(f_tri_nor);
        if (point_dot_normal <= 0.0)
            return true;
        return false;
    }

    void computeTinv()
    {
        Eigen::Matrix2f T;
        T(0, 0) = vertices[0]->texcoord(0) - vertices[2]->texcoord(0);
        T(0, 1) = vertices[1]->texcoord(0) - vertices[2]->texcoord(0);
        T(1, 0) = vertices[0]->texcoord(1) - vertices[2]->texcoord(1);
        T(1, 1) = vertices[1]->texcoord(1) - vertices[2]->texcoord(1);
        T_inv = T.inverse();
    };

    void computeBarycentric(Eigen::Vector2f p)
    {
        Eigen::Vector2f lambda = T_inv * (p - vertices[2]->texcoord);
        barycentric = Eigen::Vector3f(lambda(0), lambda(1), 1 - lambda(0) - lambda(1));
    };

    bool isBarycentricOk()
    {
        if (barycentric(0) < 0.0 || barycentric(1) < 0.0 || barycentric(2) < 0.0)
            return false;
        return true;
    };

    template <typename Type>
    Type interpolate(Type &d1, Type &d2, Type &d3)
    {
        return barycentric(0) * d1 + barycentric(1) * d2 + barycentric(2) * d2;
    };

    std::array<Eigen::Vector2f, 2> getMinMax()
    {
        std::array<Eigen::Vector2f, 2> minmax;
        minmax[0](0) = std::min(std::min(vertices[0]->texcoord(0), vertices[1]->texcoord(0)), vertices[2]->texcoord(0));
        minmax[0](1) = std::min(std::min(vertices[0]->texcoord(1), vertices[1]->texcoord(1)), vertices[2]->texcoord(1));
        minmax[1](0) = std::max(std::max(vertices[0]->texcoord(0), vertices[1]->texcoord(0)), vertices[2]->texcoord(0));
        minmax[1](1) = std::max(std::max(vertices[0]->texcoord(1), vertices[1]->texcoord(1)), vertices[2]->texcoord(1));

        return minmax;
    };

    unsigned int id;
    std::array<Vertex *, 3> vertices;

    Eigen::Matrix2f T_inv;
    Eigen::Vector3f barycentric;

private:
};