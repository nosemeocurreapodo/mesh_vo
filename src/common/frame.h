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

    Frame(const Texture<ImageType> &im,
          const Texture<Vec3f> &di,
          int id,
          SE3f local_pose = SE3f(),
          Vec2f local_exp = Vec2f(0.0, 0.0),
          Vec6f local_vel = Vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) : image_(im),
                                                                   didxy_(di)
    {
        id_ = id;
        local_pose_ = local_pose;
        local_vel_ = local_vel;
        // globalVel = JvelType::Zero();
        local_exp_ = local_exp;
    }

    Frame(const Frame &other)
    {
        image_ = other.image_;
        didxy_ = other.didxy_;

        id_ = other.id_;
        local_pose_ = other.local_pose_;
        local_vel_ = other.local_vel_;
        // globalVel = other.globalVel;
        local_exp_ = other.local_exp_;
    }

    Frame &operator=(const Frame &other)
    {
        if (this != &other)
        {
            id_ = other.id_;
            local_pose_ = other.local_pose_;
            local_vel_ = other.local_vel_;
            // globalVel = other.globalVel;
            local_exp_ = other.local_exp_;

            image_ = other.image_;
            didxy_ = other.didxy_;
        }
        return *this;
    }

    int id() const
    {
        return id_;
    }

    const Texture<ImageType> &image() const
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

    SE3f &local_pose()
    {
        return local_pose_;
    }

    const Vec2f &local_exposure() const
    {
        return local_exp_;
    }

    Vec2f &local_exposure()
    {
        return local_exp_;
    }

    const Vec6f &local_vel() const
    {
        return local_vel_;
    }

    Vec6f &local_vel()
    {
        return local_vel_;
    }

    void scalePose(float scale)
    {
        SE3f local_pose_scaled = local_pose_;
        local_pose_scaled.translation() /= scale;
        local_pose_ = local_pose_scaled;
    }

protected:
    Texture<ImageType> image_;
    Texture<Vec3f> didxy_;

    SE3f local_pose_;
    Vec6f local_vel_;
    // JvelType globalVel;
    Vec2f local_exp_;
    int id_;
};
