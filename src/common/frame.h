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

    Frame(const TextureCPU<float> &im,
          const TextureCPU<Vec3> &di,
          int id,
          SE3 local_pose = SE3(),
          SE3 global_pose = SE3(),
          Vec2 local_exp = Vec2(0.0, 0.0)) : image_(im),
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

    const TextureCPU<float> &image() const
    {
        return image_;
    }

    const TextureCPU<Vec3> &didxy() const
    {
        return didxy_;
    }

    const SE3 &local_pose() const
    {
        return local_pose_;
    }

    const SE3 &global_pose() const
    {
        return global_pose_;
    }

    SE3 &local_pose()
    {
        return local_pose_;
    }

    SE3 &global_pose()
    {
        return global_pose_;
    }

protected:
    TextureCPU<float> image_;
    TextureCPU<Vec3> didxy_;

    SE3 local_pose_;
    SE3 global_pose_;
    // JvelType localVel;
    // JvelType globalVel;
    Vec2 local_exp_;
    int id_;
};
