#pragma once
#include "params.h"
#include "common/types.h"

class Frame
{
public:
    // Allocate once (pool uses this)
    Frame(int width, int height,
          ImageType img_nodata = ImageType(-1),
          Vec3f didxy_nodata = Vec3f(0.f, 0.f, 0.f))
        : image_(width, height, img_nodata),
          didxy_(width, height, didxy_nodata),
          local_pose_(SE3f()),
          local_vel_(Vec6f(0,0,0,0,0,0)),
          local_exp_(Vec2f(0,0)),
          id_(0),
          kframe_id_(0)
    {}

    // Move-in constructor (optional, if you ever want to build frames from temp textures)
    Frame(Texture<ImageType> im,
          Texture<Vec3f> di,
          int id,
          int kframe_id,
          SE3f local_pose = SE3f(),
          Vec2f local_exp = Vec2f(0.f, 0.f),
          Vec6f local_vel = Vec6f(0,0,0,0,0,0))
        : image_(std::move(im)),
          didxy_(std::move(di)),
          local_pose_(local_pose),
          local_vel_(local_vel),
          local_exp_(local_exp),
          id_(id),
          kframe_id_(kframe_id)
    {}

    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
    Frame(Frame&&) noexcept = default;
    Frame& operator=(Frame&&) noexcept = default;

    // Reset metadata when reusing from pool
    void reset(int id, int kframe_id)
    {
        id_ = id;
        kframe_id_ = kframe_id;
        local_pose_ = SE3f();
        local_vel_  = Vec6f(0,0,0,0,0,0);
        local_exp_  = Vec2f(0,0);
    }

    const int& id() const { return id_; }
    int& id() { return id_; }

    const int& keyframe_id() const { return kframe_id_; }
    int& keyframe_id() { return kframe_id_; }

    // ✅ allow writing into textures (for preprocessing)
    Texture<ImageType>& image() { return image_; }
    const Texture<ImageType>& image() const { return image_; }

    Texture<Vec3f>& didxy() { return didxy_; }
    const Texture<Vec3f>& didxy() const { return didxy_; }

    SE3f& local_pose() { return local_pose_; }
    const SE3f& local_pose() const { return local_pose_; }

    Vec2f& local_exposure() { return local_exp_; }
    const Vec2f& local_exposure() const { return local_exp_; }

    Vec6f& local_vel() { return local_vel_; }
    const Vec6f& local_vel() const { return local_vel_; }

    void scalePose(float scale)
    {
        SE3f tmp = local_pose_;
        tmp.translation() /= scale;
        local_pose_ = tmp;
    }

private:
    Texture<ImageType> image_;
    Texture<Vec3f> didxy_;
    SE3f local_pose_;
    Vec6f local_vel_;
    Vec2f local_exp_;
    int id_;
    int kframe_id_;
};