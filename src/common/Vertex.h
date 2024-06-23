#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/camera.h"
#include "common/common.h"
#include "common/Triangle.h"

class Vertex
{
public:
    Vertex(Eigen::Vector3f &pos, Eigen::Vector2f &tc, unsigned int i)
    {
        position = pos;
        texcoord = tc;
        id = i;
    };

    Eigen::Vector3f position;
    Eigen::Vector2f texcoord;
    unsigned int id;
    std::vector<Triangle*> triangles;

private:

};
