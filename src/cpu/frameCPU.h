#pragma once

#include "sophus/se3.hpp"

#include "common/types.h"
#include "dataCPU.h"
#include "params.h"

class frameCPU
{
public:
    frameCPU(int width, int height)
        : raw_image(width, height, -1.0f)
          //dIdpix_image(width, height, vec2<float>(0.0f, 0.0f)),
          //idepth_image(width, height, -1.0f),
          //residual_image(width, height, -1.0f)
    {
        id = 0;
        affine = {0.0f, 0.0f};
    };

    frameCPU(const frameCPU &other) : raw_image(other.raw_image)
                                      //dIdpix_image(other.dIdpix_image),
                                      //idepth_image(other.idepth_image),
                                      //residual_image(other.residual_image)
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
            //dIdpix_image = other.dIdpix_image;
            //idepth_image = other.idepth_image;
            //residual_image = other.residual_image;
        }
        return *this;
    }

    void setImage(const dataCPU<float> &im, int _id)
    {
        assert(raw_image.get(0).width == im.width && raw_image.get(0).height == im.height);

        raw_image.get(0) = im;
        raw_image.generateMipmaps();

        /*
        for (int lvl = 0; lvl < raw_image.lvls; lvl++)
        {
            computeFrameDerivative(lvl);
        }
        */

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

    dataCPU<float>& getRawImage(int lvl)
    {
        return raw_image.get(lvl);
    }

    /*
    dataCPU<vec2<float>>& getdIdpixImage()
    {
        return dIdpix_image;
    }

    dataCPU<float>& getIdepthImage()
    {
        return idepth_image;
    }

    dataCPU<float>& getResidualImage()
    {
        return residual_image;
    }
    */

    int getId()
    {
        return id;
    }

private:

    /*
    void computeFrameDerivative(int lvl)
    {
        // dx.set(dx.nodata, lvl);
        // dy.set(dy.nodata, lvl);

        dataCPU<float> image = raw_image.get(lvl);

        int width = image.width;
        int height = image.height;

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
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
    */

    dataMipMapCPU<float> raw_image;
    //dataMipMapCPU<vec2<float>> dIdpix_image;
    //dataMipMapCPU<float> idepth_image;
    //dataMipMapCPU<float> residual_image;

    Sophus::SE3f pose;
    vec2<float> affine;
    int id;
};
