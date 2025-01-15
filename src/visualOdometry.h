#pragma once

#include <iostream>
#include <fstream>

#include "cpu/frameCPU.h"
#include "cpu/Shapes.h"
#include "cpu/SceneMesh.h"

#include "optimizers/poseOptimizerCPU.h"
#include "optimizers/mapOptimizerCPU.h"
#include "optimizers/poseMapOptimizerCPU.h"
//#include "optimizers/sceneOptimizerCPU.h"

class visualOdometry
{
public:
    visualOdometry(camera &_cam);

    void init(dataCPU<float> &image, SE3f pose);
    void init(dataCPU<float> &image, dataCPU<float> &idepth, SE3f pose);

    void init(frameCPU &frame);
    void init(frameCPU &frame, dataCPU<float> &idepth);
    void init(frameCPU &frame, std::vector<vec2f> &pixels, std::vector<float> &idepths);

    void locAndMap(dataCPU<float> &image);
    void lightaffine(dataCPU<float> &image, Sophus::SE3f globalPose);
    void localization(dataCPU<float> &image);
    void mapping(dataCPU<float> &image, Sophus::SE3f globalPose, vec2f exposure);

    Sophus::SE3f lastPose;
    vec2f lastExposure;

private:

    dataCPU<float> getIdepth(SE3f pose, int lvl);
    float meanViewAngle(SE3f pose1, SE3f pose2);
    float getViewPercent(frameCPU &frame);

    int lastId;
    cameraMipMap cam;
    SE3f lastMovement;
    std::vector<frameCPU> lastFrames;
    std::vector<frameCPU> keyFrames;

    SceneMesh scene;
    frameCPU kframe;

    poseOptimizerCPU poseOptimizer;
    mapOptimizerCPU mapOptimizer;
    poseMapOptimizerCPU poseMapOptimizer;
    //sceneOptimizerCPU<SceneMesh, vec3<float>, vec3<int>> sceneOptimizer;
    
    renderCPU renderer;
};
