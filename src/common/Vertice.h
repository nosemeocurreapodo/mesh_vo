#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/camera.h"
#include "common/common.h"
//#include "common/Triangle.h"

class Triangle;

class Vertice
{
public:
    Vertice(Eigen::Vector3f &pos, Eigen::Vector2f &tc, unsigned int i)
    {
        position = pos;
        texcoord = tc;
        id = i;
    };

    //can be either rayidepth or a 3d position
    Eigen::Vector3f position;
    //normalized texcoords
    Eigen::Vector2f texcoord;
    unsigned int id;

private:
};
