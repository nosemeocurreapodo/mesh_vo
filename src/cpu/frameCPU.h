#pragma once

#include "sophus/se3.hpp"

#include "dataCPU.h"
#include "params.h"

class frameCPU
{
public:
    frameCPU(int width, int height)
        : image(width, height, -1.0),
          dx(width, height, 0.0),
          dy(width, height, 0.0)
    {
        init = false;
        id = 0;
    };

    frameCPU(const frameCPU &other) : image(other.image),
                                      dx(other.dx),
                                      dy(other.dy)
    {
        init = other.init;
        id = other.id;
    }

    frameCPU &operator=(const frameCPU &other)
    {
        if (this != &other)
        {
            init = other.init;
            id = other.id;

            image = other.image;
            dx = other.dx;
            dy = other.dy;
        }
        return *this;
    }

    void set(const dataCPU<float> &im)
    {
        image = im;
        // image.generateMipmaps();
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

    void set(dataCPU<float> &im, Sophus::SE3f p)
    {
        set(im);
        pose = p;
    }

    void computeFrameDerivative(int lvl)
    {
        //dx.set(dx.nodata, lvl);
        //dy.set(dy.nodata, lvl);

        std::array<int, 2> size = image.getSize(lvl);
        for (int y = 0; y < size[1]; y++)
            for (int x = 0; x < size[0]; x++)
            {
                if (y == 0 || y == size[1] - 1 || x == 0 || x == size[0] - 1)
                {
                    dx.set(0.0, y, x, lvl);
                    dy.set(0.0, y, x, lvl);
                    continue;
                }

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
    int id;
};
