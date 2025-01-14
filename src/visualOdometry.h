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

#include "optimizers/poseOptimizerCPU.h"
#include "optimizers/mapOptimizerCPU.h"
#include "optimizers/poseMapOptimizerCPU.h"
#include "optimizers/sceneOptimizerCPU.h"

template <typename sceneType>
class visualOdometry
{
public:
    visualOdometry(camera &_cam);

    void init(dataCPU<float> &image, Sophus::SE3f pose);
    void init(dataCPU<float> &image, dataCPU<float> &idepth, Sophus::SE3f pose);

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

    dataCPU<float> getIdepth(Sophus::SE3f pose, int lvl);
    float meanViewAngle(Sophus::SE3f pose1, Sophus::SE3f pose2);
    float getViewPercent(frameCPU &frame);

    int lastId;
    cameraMipMap cam;
    Sophus::SE3f lastMovement;
    std::vector<frameCPU> lastFrames;
    std::vector<frameCPU> keyFrames;

    SceneMesh scene;
    frameCPU kframe;

    poseOptimizerCPU<SceneMesh> poseOptimizer;
    mapOptimizerCPU<SceneMesh, vec3<float>, vec3<int>> mapOptimizer;
    poseMapOptimizerCPU<SceneMesh, vec3<float>, vec3<int>> poseMapOptimizer;
    //sceneOptimizerCPU<SceneMesh, vec3<float>, vec3<int>> sceneOptimizer;
    
    renderCPU<sceneType> renderer;
};
