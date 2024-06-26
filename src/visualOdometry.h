#pragma once

#include <opencv2/core.hpp>

#include <iostream>
#include <fstream>

#include "sophus/se3.hpp"

#include "optimizers/meshOptimizerCPU.h"
//#include "scene/keyframeIdepthSceneCPU.h"
#include "cpu/frameCPU.h"

class visualOdometry
{
public:
    visualOdometry(float _fx, float _fy, float _cx, float _cy, int _width, int _height);

    void initScene(cv::Mat frame, Sophus::SE3f pose = Sophus::SE3f());
    void initScene(cv::Mat frame, cv::Mat idepth, Sophus::SE3f pose = Sophus::SE3f());

    void locAndMap(cv::Mat frame);
    void localization(cv::Mat frame);
    void mapping(cv::Mat _frame, Sophus::SE3f pose);

    dataCPU<float> getRandomIdepth(int lvl)
    {
        dataCPU<float> idepth(-1.0);

        for (int y = 0; y < idepth.sizes[lvl].height; y++) // los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
        {
            for (int x = 0; x < idepth.sizes[lvl].width; x++)
            {
                float _idepth = 0.1 + (1.0 - 0.1) * float(y) / idepth.sizes[lvl].height;
                idepth.set(_idepth, y, x, lvl);
            }
        }
        return idepth;
    }

private:
    frameCPU lastFrame;
    std::vector<frameCPU> frames;
    meshOptimizerCPU meshOptimizer;
    //keyframeIdepthSceneCPU scene;
};
