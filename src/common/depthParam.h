#pragma once

#include <cmath>

static float fromParamToDepth(float param)
{
    // depth
    //return param;
    // idepth
    //return 1.0 / param;
    // logdepth
    return std::exp(param);
    // logidepth
    // return 1.0 / std::exp(param);
}

static float fromDepthToParam(float depth)
{
    //  depth
    //return depth;
    //  idepth
    //return 1.0 / depth;
    //  logdepth
    return std::log(depth);
    // logidepth
    // return -std::log(depth);
}

static float d_depth_d_param(float depth)
{
    // depth
    //return 1.0;
    // idepth
    //return -(depth * depth);
    // logdepth
    return depth;
    // logidepth
    // return -depth;
}
