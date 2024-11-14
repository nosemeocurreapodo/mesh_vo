#pragma once

#include <opencv2/core.hpp>

#include <iostream>
#include <fstream>

#include "sophus/se3.hpp"

#include "optimizers/meshOptimizerCPU.h"
// #include "scene/keyframeIdepthSceneCPU.h"
#include "cpu/frameCPU.h"

class visualOdometry
{
public:
    visualOdometry(camera &cam);

    void initScene(dataCPU<float> &image, Sophus::SE3f pose = Sophus::SE3f());
    void initScene(dataCPU<float> &image, dataCPU<float> &idepth, dataCPU<float> &ivar, Sophus::SE3f pose = Sophus::SE3f());
    void initScene(dataCPU<float> &image, std::vector<vec2<float>> &pixels, std::vector<float> &idepths, Sophus::SE3f pose = Sophus::SE3f());

    void locAndMap(dataCPU<float> &image);
    void lightaffine(dataCPU<float> &image, Sophus::SE3f pose);
    void localization(dataCPU<float> &image);
    void mapping(dataCPU<float> &image, Sophus::SE3f pose, vec2<float> affine);

    Sophus::SE3f lastPose;
    vec2<float> lastAffine;

private:

    void checkFrameAndAddToList(frameCPU &frame)
    {
        //dataCPU<float> kIdepth = meshOptimizer.getIdepth(meshOptimizer.kframe.getPose(), 1);
    }

    int lastId;
    camera cam;
    Sophus::SE3f lastMovement;
    std::vector<frameCPU> lastFrames;
    std::vector<frameCPU> keyFrames;

    meshOptimizerCPU meshOptimizer;
    // keyframeIdepthSceneCPU scene;
};
