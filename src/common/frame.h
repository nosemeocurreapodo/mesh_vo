#pragma once

#include "params.h"
#include "common/types.h"

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

    Frame(const Texture<unsigned char> &im,
          const Texture<Vec3f> &di,
          int id,
          SE3f local_pose = SE3f(),
          SE3f global_pose = SE3f(),
          Vec2f local_exp = Vec2f(0.0, 0.0)) : image_(im),
                                               didxy_(di)
    {
        id_ = id;
        local_pose_ = local_pose;
        global_pose_ = global_pose;
        // localVel = JvelType::Zero();
        // globalVel = JvelType::Zero();
        local_exp_ = local_exp;
    }

    Frame(const Frame &other)
    {
        image_ = other.image_;
        didxy_ = other.didxy_;

        id_ = other.id_;
        local_pose_ = other.local_pose_;
        global_pose_ = other.global_pose_;
        // localVel = other.localVel;
        // globalVel = other.globalVel;
        local_exp_ = other.local_exp_;
    }

    Frame &operator=(const Frame &other)
    {
        if (this != &other)
        {
            id_ = other.id_;
            local_pose_ = other.local_pose_;
            global_pose_ = other.global_pose_;
            // localVel = other.localVel;
            // globalVel = other.globalVel;
            local_exp_ = other.local_exp_;

            image_ = other.image_;
            didxy_ = other.didxy_;
        }
        return *this;
    }

    const Texture<unsigned char> &image() const
    {
        return image_;
    }

    const Texture<Vec3f> &didxy() const
    {
        return didxy_;
    }

    const SE3f &local_pose() const
    {
        return local_pose_;
    }

    const SE3f &global_pose() const
    {
        return global_pose_;
    }

    SE3f &local_pose()
    {
        return local_pose_;
    }

    SE3f &global_pose()
    {
        return global_pose_;
    }

protected:
    Texture<unsigned char> image_;
    Texture<Vec3f> didxy_;

    SE3f local_pose_;
    SE3f global_pose_;
    // JvelType localVel;
    // JvelType globalVel;
    Vec2f local_exp_;
    int id_;
};
