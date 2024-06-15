#pragma once

#include <Eigen/Core>

#include "common/camera.h"
#include "cpu/frame_cpu.h"
#include "params.h"

class rayDepthMeshScene
{
public:
    rayDepthMeshScene(float fx, float fy, float cx, float cy, int width, int height);

    void init(frameCpu &f);
    void setIdepth();

    // scene
    std::vector<float> scene_vertices;
    std::vector<unsigned int> scene_indices;

    frameCpu frame;
    camera cam;

private:
};
