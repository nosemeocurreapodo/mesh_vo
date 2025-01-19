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

    void init(dataCPU<imageType> &im, vec2f localExp, SE3f localPose, float localScale, dataCPU<float> &idepth, camera cam)
    {
        assert(im.width == idepth.width && im.height == idepth.height);

        raw_image.get(0) = im;
        raw_image.generateMipmaps();

        globalExp = localExp;
        globalPose = localPose*globalPose;
        globalScale = localScale*globalScale;
        //pose = SIM3f(scale, p.unit_quaternion(), p.translation());

        geometry.init(idepth, cam);
    }

    void init(frameCPU &frame, float localScale, dataCPU<float> &idepth, camera cam)
    {
        assert(frame.getRawImage(0).width == idepth.width && frame.getRawImage(0).height == idepth.height);
        assert(cam.width == idepth.width && cam.height == idepth.height);

        raw_image.get(0) = frame.getRawImage(0);
        raw_image.generateMipmaps();

        globalExp = frame.getLocalExp();
        globalPose = frame.getLocalPose() * globalPose;
        globalScale = localScale * globalScale;
        //pose = SIM3f(scale, frame.getPose().unit_quaternion(), frame.getPose().translation());

        geometry.init(idepth, cam);
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
