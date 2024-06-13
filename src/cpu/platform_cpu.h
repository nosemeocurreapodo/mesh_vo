#pragma once

#include "common/camera.h"
#include "common/HGPose.h"
#include "cpu/frame_cpu.h"
#include "scene/scene_mesh.h"

// #include "cpu/IndexThreadReduce.h"

class platformCpu
{
public:
    platformCpu(){

    };

    void computeFrameDerivative(frameCpu &frame, camera &cam, int lvl);
    void computeFrameIdepth(frameCpu &frame, camera &cam, sceneMesh &scene, int lvl);
    float computeError(frameCpu &frame, frameCpu &keyframe, camera &cam, int lvl);
    HGPose computeHGPose(frameCpu &frame, frameCpu &keyframe, camera &cam, int lvl);

private:
    HGPose errorPerIndex(frameCpu &frame, frameCpu &keyframe, camera &cam, int lvl, int ymin, int ymax);
    HGPose HGPosePerIndex(frameCpu &frame, frameCpu &keyframe, camera &cam, int lvl, int ymin, int ymax);
};
