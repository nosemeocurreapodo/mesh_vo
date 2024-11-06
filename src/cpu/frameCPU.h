#pragma once

#include "sophus/se3.hpp"

#include "dataCPU.h"
#include "params.h"

class frameCPU
{
public:
    frameCPU(int width, int height)
        : raw_image(width, height, -1.0f),
          cor_image(width, height, -1.0f),
          dx(width, height, 0.0f),
          dy(width, height, 0.0f)
    {
        init = false;
        id = 0;
        affine = {0.0f, 0.0f};
    };

    frameCPU(const frameCPU &other) : raw_image(other.raw_image),
                                      cor_image(other.cor_image),
                                      dx(other.dx),
                                      dy(other.dy)
    {
        init = other.init;
        id = other.id;
        pose = other.pose;
        affine = other.affine;
    }

    frameCPU &operator=(const frameCPU &other)
    {
        if (this != &other)
        {
            init = other.init;
            id = other.id;
            pose = other.pose;
            affine = other.affine;

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

        for (int lvl = 0; lvl < MAX_LEVELS; lvl++)
        {
            computeFrameDerivative(lvl);
            computeCorrImage(lvl);
        }

        init = true;
    }

    void setPose(Sophus::SE3f p)
    {
        pose = p;
    }

    Sophus::SE3f getPose()
    {
        return pose;
    }

    float applyAffine(float f)
    {
        return std::exp(-affine[0])*f - affine[1];
    }

    vec2<float> getdIdaffine(float f)
    {
        return vec2<float>(-std::exp(-affine[0])*f, -1.0f);
    }

    void setAffine(std::array<float, 2> _affine)
    {
        affine = _affine;
        
        for (int lvl = 0; lvl < MAX_LEVELS; lvl++)
        {
            computeCorrImage(lvl);
        }
    }

    std::array<float, 2> getAffine()
    {
        return affine;
    }

    float getRawPixel(int y, int x, int lvl)
    {
        return raw_image.get(y, x, lvl);
    }

    float getRawPixel(float y, float x, int lvl)
    {
        return raw_image.get(y, x, lvl);
    }

    float getCorPixel(int y, int x, int lvl)
    {
        return cor_image.get(y, x, lvl);
    }

    float getCorPixel(float y, float x, int lvl)
    {
        return cor_image.get(y, x, lvl);
    }

    vec2<float> getdIdp(int y, int x, int lvl)
    {
        return vec2<float>(dx.get(y, x, lvl), dx.get(y, x, lvl));
    }

    vec2<float> getdIdp(float y, float x, int lvl)
    {
        return vec2<float>(dx.get(y, x, lvl), dx.get(y, x, lvl));
    }

    float getImageNoData()
    {
        return raw_image.nodata;
    }

    float getDxNoData()
    {
        return dx.nodata;
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
                float f = raw_image.get(y, x, lvl);
                float cf = applyAffine(f);
                cor_image.set(cf, y, x, lvl);
            }
    }

    dataCPU<float> raw_image;
    dataCPU<float> cor_image;
    dataCPU<float> dx;
    dataCPU<float> dy;

    Sophus::SE3f pose;
    std::array<float, 2> affine;

    bool init;
    int id;
};
