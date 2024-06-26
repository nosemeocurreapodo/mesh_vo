#pragma once

#include "sophus/se3.hpp"

#include "dataCPU.h"
#include "params.h"

class frameCPU
{
public:
    frameCPU() : image(-1.0),
                 dx(0.0),
                 dy(0.0)
    {
        init = false;
    };

    void copyTo(frameCPU &frame)
    {
        image.copyTo(frame.image);
        dx.copyTo(frame.dx);
        dy.copyTo(frame.dy);

        frame.pose = pose;
        frame.init = init;
    }

    void set(cv::Mat frame)
    {
        image.set(frame);
        image.generateMipmaps();
        /*
        computeFrameDerivative(0);
        dx.generateMipmaps();
        dy.generateMipmaps();
        */

        for (int lvl = 0; lvl < MAX_LEVELS; lvl++)
        {
            computeFrameDerivative(lvl);
        }

        init = true;
    }

    void set(cv::Mat frame, Sophus::SE3f p)
    {
        set(frame);
        pose = p;
    }

    void computeFrameDerivative(int lvl)
    {
        dx.set(dx.nodata, lvl);
        dy.set(dy.nodata, lvl);

        for (int y = 1; y < image.sizes[lvl].height - 1; y++)
            for (int x = 1; x < image.sizes[lvl].width - 1; x++)
            {
                float _dx = (float(image.get(y, x + 1, lvl)) - float(image.get(y, x - 1, lvl))) / 2.0;
                float _dy = (float(image.get(y + 1, x, lvl)) - float(image.get(y - 1, x, lvl))) / 2.0;

                dx.set(_dx, y, x, lvl);
                dy.set(_dy, y, x, lvl);
            }
    }

    dataCPU<float> image;
    dataCPU<float> dx;
    dataCPU<float> dy;

    Sophus::SE3f pose;

    bool init;
};
