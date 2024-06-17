#pragma once

#include <Eigen/Core>

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGMap.h"
#include "common/Error.h"
#include "common/HGPoseMap.h"
#include "cpu/frameCPU.h"
#include "cpu/IndexThreadReduce.h"
#include "params.h"

class surfelSceneCPU
{
public:
    surfelSceneCPU(float fx, float fy, float cx, float cy, int width, int height);

    void init(frameCPU &frame, dataCPU<float> &idepth);

    dataCPU<float> computeFrameIdepth(frameCPU &frame, int lvl);
    dataCPU<float> computeErrorImage(frameCPU &frame, int lvl);

    void optPose(frameCPU &frame);
    void optMap(std::vector<frameCPU> &frame);
    void optPoseMap(std::vector<frameCPU> &frame);

    camera getCam()
    {
        return cam;
    }

private:
    // scene
    std::vector<float> scene_vertices;

    frameCPU keyframe;
    camera cam;

    dataCPU<float> z_buffer;

    void setFromIdepth(dataCPU<float> id);

    float computeError(frameCPU &frame, int lvl);
    HGPose computeHGPose(frameCPU &frame, int lvl);
    void computeHGMap(frameCPU &frame, HGMap &hg, int lvl);
    void computeHGPoseMap(frameCPU &frame, HGPoseMap &hg, int frame_index, int lvl);

    void errorPerIndex(frameCPU &frame, int lvl, int tmin, int tmax, Error *e, int tid);
    void HGPosePerIndex(frameCPU &frame, int lvl, int tmin, int tmax, HGPose *hg, int tid);

    float errorRegu();
    void HGRegu(HGMap &hgmap);

    IndexThreadReduce<Error> errorTreadReduce;
    IndexThreadReduce<HGPose> hgPoseTreadReduce;

    bool multiThreading;
};
