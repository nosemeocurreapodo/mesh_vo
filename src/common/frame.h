#pragma once

#include "params.h"
// #include "common/types.h"
#include "core/types.h"
#include "common/types.h"
#include "backends/cpu/texturecpu.h"
#include "backends/cpu/renderercpu.h"

class Frame
{
public:
    /*
        Frame()
        {
            id = 0;
            localPose = SE3();
            globalPose = SE3();
            localVel = jvelType::Zero();
            globalVel = jvelType::Zero();
            localExp = Vec2(0.0f, 0.0f);
        };
        */

    Frame(const Texture<Image> &im, int id,
          SE3 localPose = SE3(),
          SE3 globalPose = SE3(),
          SE3 localExp = Vec2(0.0, 0.0)) : image_(im),
                                           didxy_(im.width, im.height, Vec2(0.0, 0.0))
    {
        for (int lvl = 0; lvl < image.getLvls(); lvl++)
        {
            didxy_renderer_.Render(two_triangles_mesh_, SE3(), Camera(), image_, didxy_, lvl, lvl);
            // dIdpix_image.get(lvl) = raw_image.get(lvl).computeFrameDerivative();
        }

        id_ = id;
        localPose_ = localPose;
        globalPose_ = globalPose;
        // localVel = JvelType::Zero();
        // globalVel = JvelType::Zero();
        localExp_ = localExp;
    }

    Frame(const Frame &other)
    {
        image_ = other.image_;
        didxy_ = other.didxy_;

        id_ = other.id_;
        localPose_ = other.localPose_;
        globalPose_ = other.globalPose_;
        // localVel = other.localVel;
        // globalVel = other.globalVel;
        localExp_ = other.localExp_;
    }

    Frame &operator=(const Frame &other)
    {
        if (this != &other)
        {
            id_ = other.id_;
            localPose_ = other.localPose_;
            globalPose_ = other.globalPose_;
            // localVel = other.localVel;
            // globalVel = other.globalVel;
            localExp_ = other.localExp_;

            image_ = other.image_;
            didxy_ = other.didxy_;
        }
        return *this;
    }

    /*
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

    void setGlobalPose(SE3 newGlobalPose)
    {
        globalPose = newGlobalPose;
    }

    SE3 getGlobalPose()
    {
        return globalPose;
    }

    TextureCPU<imageType> &getRawImage(int lvl)
    {
        return raw_image.get(lvl);
    }

    TextureCPU<Vec2> &getdIdpixImage(int lvl)
    {
        return dIdpix_image.get(lvl);
    }

    int getId()
    {
        return id;
    }
    */

protected:
    Texture<Image> image_;
    Texture<Vec2> didxy_;

    DIDxyRenderer didxy_renderer_;

    Mesh two_triangles_mesh_;

    SE3 localPose_;
    SE3 globalPose_;
    // JvelType localVel;
    // JvelType globalVel;
    Vec2 localExp_;
    int id_;
};
