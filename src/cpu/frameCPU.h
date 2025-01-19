#pragma once

#include "params.h"
#include "common/types.h"
#include "dataCPU.h"

class frameCPU
{
public:
    frameCPU(int width, int height)
        : raw_image(width, height, imageType(-1.0f)),
          dIdpix_image(width, height, vec2f(0.0f, 0.0f))
    // idepth_image(width, height, -1.0f),
    // residual_image(width, height, -1.0f)
    {
        id = 0;
        localPose = SE3f();
        localExp = {0.0f, 0.0f};
    };

    frameCPU(const frameCPU &other) : raw_image(other.raw_image),
                                      dIdpix_image(other.dIdpix_image)
    // idepth_image(other.idepth_image),
    // residual_image(other.residual_image)
    {
        id = other.id;
        localPose = other.localPose;
        localExp = other.localExp;
    }

    frameCPU &operator=(const frameCPU &other)
    {
        if (this != &other)
        {
            id = other.id;
            localPose = other.localPose;
            localExp = other.localExp;

            raw_image = other.raw_image;
            dIdpix_image = other.dIdpix_image;
            // idepth_image = other.idepth_image;
            // residual_image = other.residual_image;
        }
        return *this;
    }

    void setImage(const dataCPU<float> &im, int _id)
    {
        assert(raw_image.get(0).width == im.width && raw_image.get(0).height == im.height);

        raw_image.get(0) = im;
        raw_image.generateMipmaps();

        for (int lvl = 0; lvl < raw_image.getLvls(); lvl++)
        {
            computeFrameDerivative(lvl);
        }

        id = _id;
    }

    void setLocalExp(vec2f newLocalExp)
    {
        localExp = newLocalExp;
    }

    vec2f getLocalExp()
    {
        return localExp;
    }

    void setLocalPose(SE3f newLocalPose)
    {
        localPose = newLocalPose;
    }

    SE3f getLocalPose()
    {
        return localPose;
    }

    dataCPU<float> &getRawImage(int lvl)
    {
        return raw_image.get(lvl);
    }

    dataCPU<vec2f> &getdIdpixImage(int lvl)
    {
        return dIdpix_image.get(lvl);
    }

    /*
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
    void computeFrameDerivative(int lvl)
    {
        // dx.set(dx.nodata, lvl);
        // dy.set(dy.nodata, lvl);

        dataCPU<imageType> image = raw_image.get(lvl);

        int width = image.width;
        int height = image.height;

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
                {
                    // dx.set(0.0, y, x, lvl);
                    // dy.set(0.0, y, x, lvl);
                    dIdpix_image.set(vec2f(0.0f, 0.0f), y, x, lvl);
                    continue;
                }

                float _dx = (float(image.get(y, x + 1)) - float(image.get(y, x - 1))) / 2.0;
                float _dy = (float(image.get(y + 1, x)) - float(image.get(y - 1, x))) / 2.0;

                dIdpix_image.set(vec2f(_dx, _dy), y, x, lvl);
                // dx.set(_dx, y, x, lvl);
                // dy.set(_dy, y, x, lvl);
            }
    }

    dataMipMapCPU<imageType> raw_image;
    dataMipMapCPU<vec2f> dIdpix_image;
    // dataMipMapCPU<float> idepth_image;
    // dataMipMapCPU<float> residual_image;

    Sophus::SE3f localPose;
    vec2f localExp;
    int id;
};
