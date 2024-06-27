#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/camera.h"
#include "common/common.h"

class VerticeCPU
{
public:
    VerticeCPU()
    {

    };

    VerticeCPU(Eigen::Vector3f &pos, Eigen::Vector2f &tc)
    {
        position = pos;
        texcoord = tc;
    };

    // can be either rayidepth or a 3d position
    Eigen::Vector3f position;
    // normalized texcoords
    Eigen::Vector2f texcoord;

private:
};
