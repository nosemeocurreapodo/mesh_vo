#pragma once

#include "sophus/se3.hpp"

#include "dataCPU.h"
#include "params.h"

class frameCPU
{
public:
    frameCPU(int width, int height)
        : raw_image(width, height, -1.0),
          cor_image(width, height, -1),
          dx(width, height, 0.0),
          dy(width, height, 0.0)
    {
        init = false;
        id = 0;
        a = 0.0;
        b = 0.0;
    };

    frameCPU(const frameCPU &other) : raw_image(other.raw_image),
                                      cor_image(other.cor_image),
                                      dx(other.dx),
                                      dy(other.dy)
    {
        init = other.init;
        id = other.id;
        pose = other.pose;
        a = other.a;
        b = other.b;
    }

    frameCPU &operator=(const frameCPU &other)
    {
        if (this != &other)
        {
            init = other.init;
            id = other.id;
            pose = other.pose;
            a = other.a;
            b = other.b;

            raw_image = other.raw_image;
            cor_image = other.cor_image;
            dx = other.dx;
            dy = other.dy;
        }
        return *this;
    }

    void set(const dataCPU<float> &im)
    {
        raw_image = im;
        // image.generateMipmaps();
        /*
        computeFrameDerivative(0);
        dx.generateMipmaps();
        dy.generateMipmaps();
        */

        for (int lvl = 0; lvl < MAX_LEVELS; lvl++)
        {
            computeFrameDerivative(lvl);
            computeCorrImage(lvl);
        }

        init = true;
    }

    void set(dataCPU<float> &im, Sophus::SE3f p)
    {
        set(im);
        pose = p;
    }

    Sophus::SE3f getPose()
    {
        return pose;
    }

    float getRawPixel(int y, int x, int lvl)
    {
        return raw_image.get(y, x, lvl);
    }

    float getRawPixel(float y, float x, int lvl)
    {
        return raw_image.get(y, x, lvl);
    }

    float getRawNoData()
    {
        return raw_image.nodata;
    }

    float getCorPixel(int y, int x, int lvl)
    {
        return cor_image.get(y, x, lvl);
    }

    float getCorPixel(float y, float x, int lvl)
    {
        return cor_image.get(y, x, lvl);
    }

    float getCorNoData()
    {
        return cor_image.nodata;
    }

    float getDx(int y, int x, int lvl)
    {
        return dx.get(y, x, lvl);
    }

    float getDxNoData()
    {
        return dx.nodata;
    }

    float getDy(int y, int x, int lvl)
    {
        return dy.get(y, x, lvl);
    }

    float getDyNoData()
    {
        return dy.nodata;
    }

private:
    void computeFrameDerivative(int lvl)
    {
        // dx.set(dx.nodata, lvl);
        // dy.set(dy.nodata, lvl);

        std::array<int, 2> size = raw_image.getSize(lvl);
        for (int y = 0; y < size[1]; y++)
            for (int x = 0; x < size[0]; x++)
            {
                if (y == 0 || y == size[1] - 1 || x == 0 || x == size[0] - 1)
                {
                    dx.set(0.0, y, x, lvl);
                    dy.set(0.0, y, x, lvl);
                    continue;
                }

                float _dx = (float(raw_image.get(y, x + 1, lvl)) - float(raw_image.get(y, x - 1, lvl))) / 2.0;
                float _dy = (float(raw_image.get(y + 1, x, lvl)) - float(raw_image.get(y - 1, x, lvl))) / 2.0;

                dx.set(_dx, y, x, lvl);
                dy.set(_dy, y, x, lvl);
            }
    }

    void computeCorrImage(int lvl)
    {
        std::array<int, 2> size = raw_image.getSize(lvl);
        for (int y = 0; y < size[1]; y++)
            for (int x = 0; x < size[0]; x++)
            {
                float corr_val = std::exp(-a)*raw_image.get(y, x, lvl) - b;
                cor_image.set(corr_val, y, x, lvl);
            }
    }

    dataCPU<float> raw_image;
    dataCPU<float> cor_image;
    dataCPU<float> dx;
    dataCPU<float> dy;

    Sophus::SE3f pose;
    float a;
    float b;

    bool init;
    int id;
};
