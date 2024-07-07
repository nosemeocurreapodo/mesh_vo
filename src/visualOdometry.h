#pragma once

#include <opencv2/core.hpp>

#include <iostream>
#include <fstream>

#include "sophus/se3.hpp"

#include "optimizers/meshOptimizerCPU.h"
// #include "scene/keyframeIdepthSceneCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/OpenCVDebug.h"

class visualOdometry
{
public:
    visualOdometry(camera &cam);

    void initScene(dataCPU<float> &image, Sophus::SE3f pose = Sophus::SE3f());
    void initScene(dataCPU<float> &image, dataCPU<float> &idepth, Sophus::SE3f pose = Sophus::SE3f());

    void locAndMap(dataCPU<float> &image);
    void localization(dataCPU<float> &image);
    void mapping(dataCPU<float> &image, Sophus::SE3f pose);

    dataCPU<float> getRandomIdepth()
    {
        dataCPU<float> idepth(cam.width, cam.height, -1.0);

        for (int y = 0; y < cam.height; y++)
        {
            for (int x = 0; x < cam.width; x++)
            {
                float _idepth = 0.5 + (1.0 - 0.5) * float(y) / (cam.height-1.0);
                idepth.set(_idepth, y, x, 0);
            }
        }
        idepth.generateMipmaps();
        return idepth;
    }

private:
    camera cam;
    frameCPU lastFrame;
    Sophus::SE3f lastMovement;
    std::vector<frameCPU> frames;
    meshOptimizerCPU meshOptimizer;
    // keyframeIdepthSceneCPU scene;
};
