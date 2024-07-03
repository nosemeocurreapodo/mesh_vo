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

    void initScene(cv::Mat frame, Sophus::SE3f pose = Sophus::SE3f());
    void initScene(cv::Mat frame, cv::Mat idepth, Sophus::SE3f pose = Sophus::SE3f());

    void locAndMap(cv::Mat frame);
    void localization(cv::Mat frame);
    void mapping(cv::Mat _frame, Sophus::SE3f pose);

    dataCPU<float> getRandomIdepth()
    {
        dataCPU<float> idepth(cam.width, cam.height, -1.0);

        for (int y = 0; y < cam.height; y++)
        {
            for (int x = 0; x < cam.width; x++)
            {
                float _idepth = 0.1 + (1.0 - 0.1) * float(y) / (cam.height-1.0);
                idepth.set(_idepth, y, x, 0);
            }
        }
        return idepth;
    }

private:
    camera cam;
    frameCPU lastFrame;
    std::vector<frameCPU> frames;
    meshOptimizerCPU meshOptimizer;
    // keyframeIdepthSceneCPU scene;
};
