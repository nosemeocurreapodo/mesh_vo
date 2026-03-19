#pragma once
#include "params.h"
#include "common/types.h"

class Frame
{
public:
	// Allocate once (pool uses this)
	Frame(int width, int height,
		  ImageType img_nodata = ImageType(-1),
		  Vec3f didxy_nodata = Vec3f(0.f, 0.f, 0.f),
		  float depth_nodata = 1.0f)
		: image_(width, height, img_nodata),
		  didxy_(width, height, didxy_nodata),
		  global_pose_(SE3d()),
		  local_exp_(Vec2f(0, 0)),
		  pose_lambda_(Mat6f::Zero()),
		  exp_lambda_(Vec2f(0, 0)),
		  id_(0)
	{
	}

	/*
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
	*/

	Frame(const Frame &) = delete;
	Frame &operator=(const Frame &) = delete;
	Frame(Frame &&) noexcept = default;
	Frame &operator=(Frame &&) noexcept = default;

	// Reset metadata when reusing from pool
	void reset(int id, int kframe_id)
	{
		id_ = id;
		global_pose_ = SE3d();
		local_exp_ = Vec2f(0, 0);
		pose_lambda_ = Mat6f::Zero();
		exp_lambda_ = Vec2f(0, 0);
	}

	const int &id() const { return id_; }
	int &id() { return id_; }

	// ✅ allow writing into textures (for preprocessing)
	Texture<ImageType> &image() { return image_; }
	const Texture<ImageType> &image() const { return image_; }

	Texture<Vec3f> &didxy() { return didxy_; }
	const Texture<Vec3f> &didxy() const { return didxy_; }

	SE3d &global_pose() { return global_pose_; }
	const SE3d &global_pose() const { return global_pose_; }

	Vec2f &local_exposure() { return local_exp_; }
	const Vec2f &local_exposure() const { return local_exp_; }

	Mat6f &pose_lambda() { return pose_lambda_; }
	const Mat6f &pose_lambda() const { return pose_lambda_; }

	Vec2f &exp_lambda() { return exp_lambda_; }
	const Vec2f &exp_lambda() const { return exp_lambda_; }

private:
	Texture<ImageType> image_;
	Texture<Vec3f> didxy_;
	SE3d global_pose_;
	Vec2f local_exp_;
	Mat6f pose_lambda_;
	Vec2f exp_lambda_;
	int id_;
};