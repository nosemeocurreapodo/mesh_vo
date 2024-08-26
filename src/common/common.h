#pragma once

#include "common/types.h"

float fromParamToDepth(float param);
float fromDepthToParam(float depth);
float d_depth_d_param(float depth);
float initialIvar();

bool isTriangleEqual(vec3<int> tri_indices_1, vec3<int> tri_indices_2);
bool isEdgeEqual(vec2<int> edge_indices_1, vec2<int> edge_indices_2);



