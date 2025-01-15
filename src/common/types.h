#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include "sophus/se3.hpp"

using vec2f = Eigen::Vector2f;
using vec3f = Eigen::Vector3f;
using vec6f = Eigen::Matrix<float, 6, 1>;
using vecxf = Eigen::VectorXf;

using mat3f = Eigen::Matrix3f;
using mat6f = Eigen::Matrix<float, 6, 6>;
using matxf = Eigen::MatrixXf;

using vec2i = Eigen::Vector2i;
using vec3i = Eigen::Vector3i;
using vecxi = Eigen::VectorXi;

using SE3f = Sophus::SE3f;

using imageType = float;
using jmapType = vec3f;
using idsType = vec3i;

struct vertex
{
    vertex()
    {
        used = false;
    }

    vertex(vec3f v, vec3f r, vec2f p)
    {
        ver = v;
        ray = r;
        pix = p;
        // weight = w;
        used = true;
    }

    vec3f ver;
    vec3f ray;
    vec2f pix;
    // float weight;
    bool used;
};

struct triangle
{
    triangle()
    {
        used = false;
    }

    triangle(vec3i i)
    {
        vertexIds = i;
        used = true;
    }

    vec3i vertexIds;
    bool used;
};