#pragma once

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGMap.h"
#include "cpu/frame_cpu.h"
#include "scene/rayDepthMeshScene.h"

// #include "cpu/IndexThreadReduce.h"

class platformCpu
{
public:
    platformCpu(){

    };

    void computeFrameIdepth(frameCpu &frame, rayDepthMeshScene &scene, int lvl);
    float computeError(frameCpu &frame, frameCpu &keyframe, rayDepthMeshScene &scene, int lvl);
    HGPose computeHGPose(frameCpu &frame, frameCpu &keyframe, rayDepthMeshScene &scene, int lvl);
    HGMap computeHGMap(frameCpu &frame, frameCpu &keyframe, rayDepthMeshScene &scene, int lvl);

private:
    HGPose errorPerIndex(frameCpu &frame, frameCpu &keyframe, camera &cam, int lvl, int ymin, int ymax);
    HGPose errorPerIndex2(frameCpu &frame, rayDepthMeshScene &scene, int lvl);
    HGPose HGPosePerIndex(frameCpu &frame, frameCpu &keyframe, camera &cam, int lvl, int ymin, int ymax);
    HGPose HGPosePerIndex2(frameCpu &frame, rayDepthMeshScene &scene, int lvl);
    HGMap HGMapPerIndex(frameCpu &frame, rayDepthMeshScene &scene, int lvl);
};
