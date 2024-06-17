#pragma once

#include <Eigen/Core>

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGMap.h"
#include "cpu/frameCPU.h"
#include "params.h"

class rayDepthMeshSceneCPU
{
public:
    rayDepthMeshSceneCPU(float fx, float fy, float cx, float cy, int width, int height);

    void init(frameCPU &frame, frameCPU &idepth);
    void setFromIdepth(data_cpu<float> id);

    dataCPU<float> computeFrameIdepth(frameCPU &frame, int lvl);
    float computeError(frameCPU &frame, int lvl);
    
    void optPose(frameCPU &frame);
    void optMap(frameCPU &frame);
    void optPoseMap(frameCPU &frame);


    std::vector<float> getVertices()
    {
        return scene_vertices;
    }

    void setVertices(std::vector<float> new_vertices)
    {
        scene_vertices = new_vertices;
    }

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

    HGPose computeHGPose(frameCPU &frame, int lvl);
    HGMap computeHGMap(frameCPU &frame,int lvl);
    HGMap computeHGPoseMap(frameCPU &frame,int lvl);

    float errorRegu();
    void HGRegu(HGMap &hgmap);
};
