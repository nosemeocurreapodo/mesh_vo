#pragma once

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGMap.h"
#include "cpu/frameCPU.h"

// #include "cpu/IndexThreadReduce.h"

class sceneCPU
{
public:
    sceneCPU(float fx, float fy, float cx, float cy, int width, int height);
    void init(frameCPU &f);
    float computeError(frameCPU &frame, int lvl);
    dataCPU<float> computeIdepth(frameCPU &frame, int lvl);
    HGPose computeHGPose(frameCPU &frame, int lvl);
    HGMap computeHGMap(frameCPU &frame, int lvl);

private:

    HGPose errorPerIndex(frameCPU &frame, int lvl, int ymin, int ymax);
    void idepthPerIndex(frameCPU &frame, frameCPU &frameIdepth, int lvl);
    HGPose HGPosePerIndex(frameCPU &frame, int lvl, int ymin, int ymax);

    frameCPU keyframe;
    dataCPU<float> keyframeIdepth;
    camera cam;
}
