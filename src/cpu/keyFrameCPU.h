#pragma once

#include "params.h"
#include "common/types.h"
#include "common/camera.h"
#include "cpu/GeometryMesh.h"
#include "dataCPU.h"

class keyFrameCPU
{
public:
    keyFrameCPU(int width, int height)
        : raw_image(width, height, imageType(-1.0f))
    {
        globalPose = SE3f();
        globalExp = {0.0f, 0.0f};
        globalScale = 1.0;
    };

    keyFrameCPU(const keyFrameCPU &other)
        : raw_image(other.raw_image)
    {
        globalPose = other.globalPose;
        globalExp = other.globalExp;
        globalScale = other.globalScale;
        geometry = other.geometry;
    }

    keyFrameCPU &operator=(const keyFrameCPU &other)
    {
        if (this != &other)
        {
            globalPose = other.globalPose;
            globalExp = other.globalExp;
            raw_image = other.raw_image;
            globalScale = other.globalScale;
            geometry = other.geometry;
        }
        return *this;
    }

    void init(dataCPU<imageType> &im, vec2f _globalExp, SE3f _globalPose, float _globalScale, dataCPU<float> &idepth, dataCPU<float> &weight, camera cam)
    {
        assert(im.width == idepth.width && im.height == idepth.height);
        assert(cam.width == im.width && cam.height == im.height);
        assert(cam.width == weight.width && cam.height == weight.height);

        raw_image.get(0) = im;
        raw_image.generateMipmaps();

        globalExp = _globalExp;
        globalPose = _globalPose;
        globalScale = _globalScale;
        //pose = SIM3f(scale, p.unit_quaternion(), p.translation());

        geometry.init(idepth, weight, cam);
    }

    SE3f localPoseToGlobal(SE3f _localPose)
    {
        SE3f localPoseScaled = _localPose;
        localPoseScaled.translation() /= globalScale;
        SE3f _globalPose = localPoseScaled * globalPose;
        return _globalPose;
    }

    SE3f globalPoseToLocal(SE3f _globalPose)
    {
        SE3f _localPose = _globalPose * globalPose.inverse();
        _localPose.translation() *= globalScale;
        return _localPose;
    }

    vec2f localExpToGlobal(vec2f _localExp)
    {
        vec2f _globalExp;
        float alpha1 = std::exp(_localExp(0));
        float alpha2 = std::exp(globalExp(0));
        _globalExp(0) = std::log(alpha1/alpha2);
        _globalExp(1) = _localExp(1) - globalExp(1)/alpha1;
        return _globalExp;
    }

    vec2f globalExpToLocal(vec2f _globalExp)
    {
        vec2f _localExp;
        float alpha1 = std::exp(-_globalExp(0));
        float alpha2 = std::exp(-globalExp(0));
        _localExp(0) = std::log(alpha1/alpha2);
        _localExp(1) = -_globalExp(1) + (alpha2/alpha1)*globalExp(1);
        return _localExp;
    }

    void scaleVertices(float scale)
    {
        globalScale *= scale;
        geometry.scaleVertices(scale);
    }

    float getGlobalScale()
    {
        return globalScale;
    }

    void setGlobalExp(vec2f newGlobalExp)
    {
        globalExp = newGlobalExp;
    }

    vec2f getGlobalExp()
    {
        return globalExp;
    }

    void setGlobalPose(SE3f newGlobalPose)
    {
        globalPose = newGlobalPose;
    }

    SE3f getGlobalPose()
    {
        return globalPose;
    }

    dataCPU<float> &getRawImage(int lvl)
    {
        return raw_image.get(lvl);
    }

    geometryType &getGeometry()
    {
        return geometry;
    }

private:
    dataMipMapCPU<imageType> raw_image;
    geometryType geometry;
    float globalScale;

    SE3f globalPose;
    vec2f globalExp;
};
