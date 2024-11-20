#pragma once

#include "sophus/se3.hpp"

#include "dataCPU.h"
#include "params.h"

template <int width, int height>
class frameCPU
{
public:
    frameCPU()
        : raw_image(-1.0f),
          dIdpix_image(vec2<float>(0.0f, 0.0f)),
          idepth_image(-1.0f),
          residual_image(-1.0f)
    {
        id = 0;
        affine = {0.0f, 0.0f};
    };

    frameCPU(const frameCPU &other) : raw_image(other.raw_image),
                                      dIdpix_image(other.dIdpix_image),
                                      idepth_image(other.idepth_image),
                                      residual_image(other.residual_image)
    {
        id = other.id;
        pose = other.pose;
        affine = other.affine;
    }

    frameCPU &operator=(const frameCPU &other)
    {
        if (this != &other)
        {
            id = other.id;
            pose = other.pose;
            affine = other.affine;

            raw_image = other.raw_image;
            dIdpix_image = other.dIdpix_image;
            idepth_image = other.idepth_image;
            residual_image = other.residual_image;
        }
        return *this;
    }

    void setImage(const dataCPU<IMAGE_WIDTH, IMAGE_HEIGHT, float> &im, int _id)
    {
        raw_image = im;

        for (int lvl = 0; lvl < MAX_LEVELS; lvl++)
        {
            computeFrameDerivative(lvl);
        }

        id = _id;
    }

    void setAffine(vec2<float> _affine)
    {
        affine = _affine;
    }
    
    vec2<float> getAffine()
    {
        return affine;
    }

    void setPose(Sophus::SE3f p)
    {
        pose = p;
    }

    Sophus::SE3f getPose()
    {
        return pose;
    }

    dataCPU<width, height, float>& getRawImage()
    {
        return raw_image;
    }

    dataCPU<width, height, vec2<float>>& getdIdpixImage()
    {
        return dIdpix_image;
    }

    dataCPU<width, height, float>& getIdepthImage()
    {
        return idepth_image;
    }

    dataCPU<width, height, float>& getResidualImage()
    {
        return residual_image;
    }

    int getId()
    {
        return id;
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
                    //dx.set(0.0, y, x, lvl);
                    //dy.set(0.0, y, x, lvl);
                    dIdpix_image.set(vec2<float>(0.0f, 0.0f), y, x, lvl);
                    continue;
                }

                float _dx = (float(raw_image.get(y, x + 1, lvl)) - float(raw_image.get(y, x - 1, lvl))) / 2.0;
                float _dy = (float(raw_image.get(y + 1, x, lvl)) - float(raw_image.get(y - 1, x, lvl))) / 2.0;

                dIdpix_image.set(vec2<float>(_dx, _dy), y, x, lvl);
                //dx.set(_dx, y, x, lvl);
                //dy.set(_dy, y, x, lvl);
            }
    }

    dataCPU<width, height, float> raw_image;
    dataCPU<width, height, vec2<float>> dIdpix_image;
    dataCPU<width, height, float> idepth_image;
    dataCPU<width, height, float> residual_image;

    Sophus::SE3f pose;
    vec2<float> affine;

    int id;
};
