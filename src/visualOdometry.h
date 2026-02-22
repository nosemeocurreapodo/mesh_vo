#pragma once

#include <iostream>
#include <fstream>
#include <vector>

#include "cpu/frameCPU.h"
#include "cpu/keyFrameCPU.h"

#include "optimizers/poseOptimizerCPU.h"
#include "optimizers/mapOptimizerCPU.h"
#include "optimizers/poseMapOptimizerCPU.h"
//#include "optimizers/intrinsicPoseMapOptimizerCPU.h"

template <class Frame, class Keyframe, class Camera, class Optimizer>
static void optimize(Frame &frame, Keyframe &kframe, Camera &cam, Optimizer optimizer)
{
    for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_ini_lvl; lvl--)
    {
        optimizer.init(frame, kframe, cam, lvl);
        while (true)
        {
            optimizer.step(frame, kframe, cam, lvl);
            if (optimizer.converged())
                break;
        }
    }
} 

template <class Frame, class Keyframe, class Camera, class Optimizer>
static void optimizePoseDepth(std::vector<Frame> &frames, Keyframe &kframe, Camera &cam, Optimizer &optimizer)
{
    for (int lvl = mesh_vo::mapping_ini_lvl; lvl >= mesh_vo::mapping_ini_lvl; lvl--)
    {
        optimizer.init(frames, kframe, cam, lvl);
        while (true)
        {
            optimizer.step(frames, kframe, cam, lvl);
            if (optimizer.converged())
                break;
        }
    }
}

class visualOdometry
{
public:
    visualOdometry(TextureCPU<imageType> &image, SE3 globalPose, cameraType cam);
    visualOdometry(dataCPU<imageType> &image, dataCPU<float> &depth, dataCPU<float> &weight, SE3f globalPose, cameraType cam);

    int locAndMap(dataCPU<imageType> &image);
    void intrinsicAndLocAndMap(dataCPU<imageType> &image);
    void lightaffine(dataCPU<imageType> &image, SE3f globalPose);
    void localization(dataCPU<imageType> &image);
    void mapping(dataCPU<imageType> &image, SE3f globalPose, vec2f exposure);

    std::vector<frameCPU> getFrames();
    keyFrameCPU getKeyframe();

private:
   

void optimizePoseMap(std::vector<FrameCPU> &frames, KeyFrameCPU &kframe, CameraType &cam);

    float meanViewAngle(SE3f pose1, SE3f pose2);
    float getViewPercent(frameCPU &frame);

    int lastId;
    cameraType cam;

    std::vector<frameCPU> goodFrames;

    keyFrameCPU kframe;
    frameCPU lastFrame;

    poseOptimizerCPU poseOptimizer;
    mapOptimizerCPU mapOptimizer;
    poseMapOptimizerCPU poseMapOptimizer;
    //intrinsicPoseMapOptimizerCPU intrinsicPoseMapOptimizer;
    //sceneOptimizerCPU<SceneMesh, vec3<float>, vec3<int>> sceneOptimizer;
    
    SE3f lastLocalMovement;
};
