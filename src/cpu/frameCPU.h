#pragma once

#include "params.h"
#include "common/types.h"
#include "dataCPU.h"

class frameCPU
{
public:
    frameCPU()
    {
        id = 0;
        localPose = SE3f();
        globalPose = SE3f();
        localVel = jvelType::Zero();
        globalVel = jvelType::Zero();
        localExp = vec2f(0.0f, 0.0f);
    };

    frameCPU(const dataCPU<imageType> &im, int _id)
    {
        raw_image = dataMipMapCPU<imageType>(im);
        // raw_image.generateMipmaps();

        dIdpix_image = dataMipMapCPU<vec2f>(im.width, im.height, vec2f(0.0, 0.0));

        for (int lvl = 0; lvl < raw_image.getLvls(); lvl++)
        {
            dIdpix_image.get(lvl) = raw_image.get(lvl).computeFrameDerivative();
        }

        id = _id;
        localPose = SE3f();
        globalPose = SE3f();
        localVel = jvelType::Zero();
        globalVel = jvelType::Zero();
        localExp = vec2f(0.0, 0.0);
    }

    frameCPU(const frameCPU &other)
    {
        raw_image = other.raw_image;
        dIdpix_image = other.dIdpix_image;

        id = other.id;
        localPose = other.localPose;
        globalPose = other.globalPose;
        localVel = other.localVel;
        globalVel = other.globalVel;
        localExp = other.localExp;
    }

    frameCPU &operator=(const frameCPU &other)
    {
        if (this != &other)
        {
            id = other.id;
            localPose = other.localPose;
            globalPose = other.globalPose;
            localVel = other.localVel;
            globalVel = other.globalVel;
            localExp = other.localExp;

            raw_image = other.raw_image;
            dIdpix_image = other.dIdpix_image;
        }
        return *this;
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

    void setLocalVel(jvelType newLocalVel)
    {
        localVel = newLocalVel;
    }

    jvelType getLocalVel()
    {
        return localVel;
    }

    void setGlobalVel(jvelType newGlobalVel)
    {
        globalVel = newGlobalVel;
    }

    jvelType getGlobalVel()
    {
        return globalVel;
    }

    void setGlobalPose(SE3f newGlobalPose)
    {
        globalPose = newGlobalPose;
    }

    SE3f getGlobalPose()
    {
        return globalPose;
    }

    dataCPU<imageType> &getRawImage(int lvl)
    {
        return raw_image.get(lvl);
    }

    dataCPU<vec2f> &getdIdpixImage(int lvl)
    {
        return dIdpix_image.get(lvl);
    }

    int getId()
    {
        return id;
    }

private:
    dataMipMapCPU<imageType> raw_image;
    dataMipMapCPU<vec2f> dIdpix_image;

    SE3f localPose;
    SE3f globalPose;
    jvelType localVel;
    jvelType globalVel;
    vec2f localExp;
    int id;
};
