#pragma once

#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>

#include "sophus/se3.hpp"

#include "cpu/frameCPU.h"
//#include "cpu/ScenePatches.h"
#include "cpu/SceneMesh.h"
#include "cpu/Shapes.h"
//#include "cpu/SceneSurfels.h"
//#include "cpu/SceneMeshSmooth.h"

#include "optimizers/sceneOptimizerCPU.h"

class visualOdometry
{
public:
    visualOdometry(camera &cam);

    void init(frameCPU &frame);
    void init(frameCPU &frame, dataCPU<float> &idepth);
    void init(frameCPU &frame, std::vector<vec2<float>> &pixels, std::vector<float> &idepths);

    void locAndMap(dataCPU<float> &image);
    void lightaffine(dataCPU<float> &image, Sophus::SE3f globalPose);
    void localization(dataCPU<float> &image);
    void mapping(dataCPU<float> &image, Sophus::SE3f globalPose, vec2<float> affine);

    Sophus::SE3f lastPose;
    vec2<float> lastAffine;

private:

    //void checkFrameAndAddToList(frameCPU &frame)
    //{
    //  dataCPU<float> kIdepth = meshOptimizer.getIdepth(meshOptimizer.kframe.getPose(), 1);
    //}

    int lastId;
    camera cam;
    Sophus::SE3f lastMovement;
    std::vector<frameCPU> lastFrames;
    std::vector<frameCPU> keyFrames;

    SceneMesh kscene;
    frameCPU kframe;

    sceneOptimizerCPU<SceneMesh, vec3<float>, vec3<int>> sceneOptimizer;
    renderCPU<SceneMesh> renderer;
};
