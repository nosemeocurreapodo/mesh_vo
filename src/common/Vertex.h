#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/camera.h"
#include "common/common.h"
#include "common/Triangle.h"

class Vertex
{
public:
    Vertex(Eigen::Vector3f &d, unsigned int ind)
    {
        data = d;
        index = ind;
    };

    void operator=(const Eigen::Vector3f a)
    {
        data = a;
    }

    void operator=(const Vertex a)
    {
        data = a.data;
        index = a.index;
        triangle_indices = a.triangle_indices;
    }

    void addTriangle(unsigned int t_id)
    {
        triangles_indices.push_back(t_id);
    }

private:
    Eigen::Vector3f data;
    unsigned int index;

    std::vector<unsigned int> triangles_indices;
};

Eigen::Vector3f operator*(Sophus::SE3f const &pose, Vertex &vert)
{
    return pose * vert.data;
}