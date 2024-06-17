#pragma once

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGMap.h"
#include "common/Error.h"
#include "cpu/frameCPU.h"
#include "cpu/IndexThreadReduce.h"

// #include "cpu/IndexThreadReduce.h"

class keyframeIdepthSceneCPU
{
public:
    keyframeIdepthSceneCPU(float fx, float fy, float cx, float cy, int width, int height);

    void init(frameCPU &frame, dataCPU<float> &idepth);

    dataCPU<float> computeFrameIdepth(frameCPU &frame, int lvl);
    float computeError(frameCPU &frame, int lvl);
    dataCPU<float> computeErrorImage(frameCPU &frame, int lvl);

    void optPose(frameCPU &frame);
    void optMap(std::vector<frameCPU> &frame);
    // void optPoseMap(std::vector<frameCPU> &frame);

private:

    frameCPU keyframe;
    dataCPU<float> keyframeIdepth;
    camera cam;

    IndexThreadReduce<Error> errorTreadReduce;
    IndexThreadReduce<HGPose> hgPoseTreadReduce;

    bool multiThreading;

    void idepthPerIndex(frameCPU &frame, dataCPU<float> &frameIdepth, int lvl, int ymin, int ymax, Error *dummy, int tid);
    void errorPerIndex(frameCPU &frame, int lvl, int ymin, int ymax, Error *hg, int tid);
    void errorImagePerIndex(frameCPU &frame, dataCPU<float> &errorImage, int lvl, int ymin, int ymax, Error *hg, int tid);

    HGPose computeHGPose(frameCPU &frame, int lvl);
    void HGPosePerIndex(frameCPU &frame, int lvl, int ymin, int ymax, HGPose *hg, int tid);
};