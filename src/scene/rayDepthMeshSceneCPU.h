#pragma once

#include <Eigen/Core>

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGMap.h"
#include "common/HGPoseMap.h"
#include "cpu/frameCPU.h"
#include "params.h"

class rayDepthMeshSceneCPU
{
public:
    rayDepthMeshSceneCPU(float fx, float fy, float cx, float cy, int width, int height);

    void init(frameCPU &frame, dataCPU<float> &idepth);

    dataCPU<float> computeFrameIdepth(frameCPU &frame, int lvl);
    float computeError(frameCPU &frame, int lvl);

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
    std::vector<unsigned int> scene_indices;

    frameCPU keyframe;
    camera cam;

    dataCPU<float> z_buffer;

    void setFromIdepth(dataCPU<float> id);

    void computeHGPose(frameCPU &frame, HGPose &hg, int lvl);
    void computeHGMap(frameCPU &frame, HGMap &hg, int lvl);
    void computeHGPoseMap(frameCPU &frame, HGPoseMap &hg, int frame_index, int lvl);

    float errorRegu();
    void HGRegu(HGMap &hgmap);

    //IndexThreadReduce<Vec10> treadReduce;
};
