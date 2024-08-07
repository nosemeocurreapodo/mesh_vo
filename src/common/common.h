#pragma once

#include "common/types.h"

enum DepthJacobianMethod
{
    depthJacobian,
    idepthJacobian,
    logDepthJacobian,
    logIdepthJacobian
};

bool isTriangleEqual(vec3<int> tri_indices_1, vec3<int> tri_indices_2);
bool isEdgeEqual(vec2<int> edge_indices_1, vec2<int> edge_indices_2);



