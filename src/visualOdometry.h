#pragma once

#include <iostream>
#include <fstream>

#include "cpu/frameCPU.h"
#include "cpu/keyFrameCPU.h"

#include "optimizers/poseOptimizerCPU.h"
#include "optimizers/mapOptimizerCPU.h"
#include "optimizers/poseMapOptimizerCPU.h"
//#include "optimizers/sceneOptimizerCPU.h"

class visualOdometry
{
public:
    visualOdometry(camera &_cam);

    void init(dataCPU<float> &image, SE3f pose);
    void init(dataCPU<float> &image, SE3f pose, dataCPU<float> &idepth, dataCPU<float> &weight);

    void locAndMap(dataCPU<float> &image);
    void lightaffine(dataCPU<float> &image, Sophus::SE3f globalPose);
    void localization(dataCPU<float> &image);
    void mapping(dataCPU<float> &image, Sophus::SE3f globalPose, vec2f exposure);

private:

    float meanViewAngle(SE3f pose1, SE3f pose2);
    float getViewPercent(frameCPU &frame);

    int lastId;
    cameraMipMap cam;

    std::vector<frameCPU> goodFrames;

    keyFrameCPU kframe;
    frameCPU lastFrame;

    poseOptimizerCPU poseOptimizer;
    mapOptimizerCPU mapOptimizer;
    poseMapOptimizerCPU poseMapOptimizer;
    //sceneOptimizerCPU<SceneMesh, vec3<float>, vec3<int>> sceneOptimizer;
    
    renderCPU renderer;

    SE3f lastLocalMovement;
};
