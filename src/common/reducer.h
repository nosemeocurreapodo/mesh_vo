#pragma once

#include <thread>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>

#include "params.h"
#include "common/types.h"
#include "common/error.h"
#include "common/depthParam.h"
#include "common/DenseLinearProblem.h"

// Generic, thread-safe reducer base. Splits [0, N) into contiguous chunks and aggregates results.
template <class Derived, typename OutType>
class BaseReducerCPU
{
public:
	BaseReducerCPU() noexcept : threads_(2 /*std::max(1u, std::thread::hardware_concurrency())*/) {}
	explicit BaseReducerCPU(unsigned threads) noexcept : threads_(threads == 0 ? 1u : threads) {}
	virtual ~BaseReducerCPU() = default;

protected:
	void reduce_(OutType &total)
	{
		const int N = derived_().size();
		total += derived_().reducepartial(0, N);
		/*
		const unsigned T = std::min<unsigned>(threads_, static_cast<unsigned>(N));
		const int chunk = (N + static_cast<int>(T) - 1) / static_cast<int>(T);

		std::vector<OutType> partial(T);
		std::vector<std::thread> pool;
		pool.reserve(T > 0 ? T - 1 : 0);

		for (unsigned t = 1; t < T; ++t)
		{
			const int begin = static_cast<int>(t) * chunk;
			const int end = std::min(N, begin + chunk);
			pool.emplace_back([&, t, begin, end]()
							  { partial[t] = derived_().reducepartial(begin, end); });
		}
		{
			const int begin = 0;
			const int end = std::min(N, chunk);
			partial[0] = derived_().reducepartial(begin, end);
		}
		for (auto &th : pool)
			th.join();

		// if (T == 0) return OutType{};
		for (unsigned t = 0; t < T; ++t)
			total += partial[t];
			*/
	}

private:
	Derived &derived_() { return *static_cast<Derived *>(this); }
	const Derived &derived_() const { return *static_cast<const Derived *>(this); }

	unsigned threads_;
};

class NodataReducerCPU : public BaseReducerCPU<NodataReducerCPU, Error>
{
public:
	using Base = BaseReducerCPU<NodataReducerCPU, Error>;
	explicit NodataReducerCPU() : Base() {}

	void reduce(int lvl, const Texture<ImageType> &r_texture, Error &total)
	{
		r_texture_ = r_texture.MapRead(lvl);

		reduce_(total);
	}

	int size()
	{
		return r_texture_.width() * r_texture_.height();
	}

	Error reducepartial(int begin, int end)
	{
		Error err;

		for (int i = begin; i < end; ++i)
		{
			const float r = r_texture_[i];
			if (r == r_texture_.nodata())
				err += 1.0f;
			else
				err += 0.0f;
		}
		return err;
	}

private:
	TextureViewRead<ImageType> r_texture_;
};

// Photometric L2 with Huber loss.
class ResidualReducerCPU : public BaseReducerCPU<ResidualReducerCPU, Error>
{
public:
	using Base = BaseReducerCPU<ResidualReducerCPU, Error>;
	explicit ResidualReducerCPU()
		: Base()
	{
	}

	void reduce(int lvl, const Texture<ImageType> &texture_1, const Texture<ImageType> &texture_2, Error &total)
	{
		assert(texture_1.width(lvl) == texture_2.width(lvl) && texture_2.height(lvl) == texture_2.height(lvl));

		texture_1_ = texture_1.MapRead(lvl);
		texture_2_ = texture_2.MapRead(lvl);

		reduce_(total);
	}

	int size()
	{
		return texture_1_.width() * texture_1_.height();
	}

	Error reducepartial(int begin, int end)
	{
		Error err;

		for (int i = begin; i < end; ++i)
		{
			const float tex1 = texture_1_[i];
			const float tex2 = texture_2_[i];
			if (tex1 == texture_1_.nodata() || tex2 == texture_2_.nodata())
				continue;
			const float r = tex1 - tex2;
			const float w = huber_weight(r, mesh_vo::huber_thresh_pix);
			err += w * r * r;
		}

		return err;
	}

private:
	TextureViewRead<ImageType> texture_1_;
	TextureViewRead<ImageType> texture_2_;
};

// Pose-only Jacobian -> DenseLinearProblem reducer
class HGPoseReducerCPU : public BaseReducerCPU<HGPoseReducerCPU, DenseLinearProblem<6>>
{
public:
	using Base = BaseReducerCPU<HGPoseReducerCPU, DenseLinearProblem<6>>;
	explicit HGPoseReducerCPU()
		: Base()
	{
	}

	void reduce(int lvl, const Texture<Vec3f> &jtra_texture, const Texture<Vec3f> &jrot_texture, const Texture<ImageType> &image_texture, const Texture<ImageType> ref_texture, DenseLinearProblem<6> &total)
	{
		jtra_texture_ = jtra_texture.MapRead(lvl);
		jrot_texture_ = jrot_texture.MapRead(lvl);
		image_texture_ = image_texture.MapRead(lvl);
		ref_texture_ = ref_texture.MapRead(lvl);

		reduce_(total);
	}

	int size()
	{
		return image_texture_.width() * image_texture_.height();
	}

	DenseLinearProblem<6> reducepartial(int begin, int end)
	{
		DenseLinearProblem<6> hg;

		// Vec8i ids(0, 1, 2, 3, 4, 5, 6, 7);
		Vec8i ids;
		for (int i = 0; i < 8; i++)
			ids(i) = i;

		for (int i = begin; i < end; ++i)
		{
			const ImageType image = image_texture_[i];
			const ImageType ref_image = ref_texture_[i];
			const Vec3f jtra = jtra_texture_[i];
			const Vec3f jrot = jrot_texture_[i];
			if (image == image_texture_.nodata() ||
				ref_image == ref_texture_.nodata() ||
				jtra == jtra_texture_.nodata() ||
				jrot == jrot_texture_.nodata())
				continue;
			float res = ref_image - image;
			// Vec8f J(jtra(0), jtra(1), jtra(2), jrot(0), jrot(1), jrot(2), jexp(0), jexp(1));
			Vec6f J;
			J(0) = jtra(0);
			J(1) = jtra(1);
			J(2) = jtra(2);
			J(3) = jrot(0);
			J(4) = jrot(1);
			J(5) = jrot(2);

			const float w = huber_weight(res, mesh_vo::huber_thresh_pix);

			// hg.add(J, res, w, ids);
			hg.add(J, res, w);
		}
		return hg;
	}

private:
	TextureViewRead<Vec3f> jtra_texture_;
	TextureViewRead<Vec3f> jrot_texture_;
	TextureViewRead<ImageType> image_texture_;
	TextureViewRead<ImageType> ref_texture_;
};

// Pose-only Jacobian -> DenseLinearProblem reducer
class HGPoseExpReducerCPU : public BaseReducerCPU<HGPoseExpReducerCPU, DenseLinearProblem<8>>
{
public:
	using Base = BaseReducerCPU<HGPoseExpReducerCPU, DenseLinearProblem<8>>;
	explicit HGPoseExpReducerCPU()
		: Base()
	{
	}

	void reduce(int lvl, const Texture<Vec3f> &jtra_texture, const Texture<Vec3f> &jrot_texture, const Texture<Vec3f> &jexp_texture, const Texture<ImageType> &image_texture, const Texture<ImageType> &ref_texture, DenseLinearProblem<8> &total)
	{
		jtra_texture_ = jtra_texture.MapRead(lvl);
		jrot_texture_ = jrot_texture.MapRead(lvl);
		jexp_texture_ = jexp_texture.MapRead(lvl);
		image_texture_ = image_texture.MapRead(lvl);
		ref_texture_ = ref_texture.MapRead(lvl);

		reduce_(total);
	}

	int size()
	{
		return image_texture_.width() * image_texture_.height();
	}

	DenseLinearProblem<8> reducepartial(int begin, int end)
	{
		DenseLinearProblem<8> hg;

		// Vec8i ids(0, 1, 2, 3, 4, 5, 6, 7);
		Vec8i ids;
		for (int i = 0; i < 8; i++)
			ids(i) = i;

		for (int i = begin; i < end; ++i)
		{
			const ImageType image = image_texture_[i];
			const ImageType ref = ref_texture_[i];
			const Vec3f jtra = jtra_texture_[i];
			const Vec3f jrot = jrot_texture_[i];
			const Vec3f jexp = jexp_texture_[i];
			if (image == image_texture_.nodata() ||
				ref == ref_texture_.nodata() ||
				jtra == jtra_texture_.nodata() ||
				jrot == jrot_texture_.nodata() ||
				jexp == jexp_texture_.nodata())
				continue;
			float res = ref - image;
			// Vec8f J(jtra(0), jtra(1), jtra(2), jrot(0), jrot(1), jrot(2), jexp(0), jexp(1));
			Vec8f J;
			J(0) = jtra(0);
			J(1) = jtra(1);
			J(2) = jtra(2);
			J(3) = jrot(0);
			J(4) = jrot(1);
			J(5) = jrot(2);
			J(6) = jexp(0);
			J(7) = jexp(1);

			const float w = huber_weight(res, mesh_vo::huber_thresh_pix);

			// hg.add(J, res, w, ids);
			hg.add(J, res, w);
		}

		return hg;
	}

private:
	TextureViewRead<Vec3f> jtra_texture_;
	TextureViewRead<Vec3f> jrot_texture_;
	TextureViewRead<Vec3f> jexp_texture_;
	TextureViewRead<ImageType> image_texture_;
	TextureViewRead<ImageType> ref_texture_;
};

// Pose-only Jacobian -> DenseLinearProblem reducer
class HGDepthReducerCPU : public BaseReducerCPU<HGDepthReducerCPU, DenseLinearProblemx>
{
public:
	using Base = BaseReducerCPU<HGDepthReducerCPU, DenseLinearProblemx>;
	explicit HGDepthReducerCPU()
		: Base()
	{
	}

	void reduce(int lvl, int frame_id, int num_frames, int num_vertices, const Texture<Vec3f> &jdepth_texture, const Texture<Vec3<PidType>> &pids_texture, const Texture<ImageType> &image_texture, const Texture<ImageType> &ref_texture, const Mesh &mesh, DenseLinearProblemx &total)
	{
		jdepth_texture_ = jdepth_texture.MapRead(lvl);
		pids_texture_ = pids_texture.MapRead(lvl);
		image_texture_ = image_texture.MapRead(lvl);
		ref_texture_ = ref_texture.MapRead(lvl);

		depths_ = get_depths(mesh);

		num_frames_ = num_frames;
		num_vertices_ = num_vertices;
		frame_id_ = frame_id;

		reduce_(total);
	}

	int size()
	{
		return image_texture_.width() * image_texture_.height();
	}

	DenseLinearProblemx reducepartial(int begin, int end)
	{
		DenseLinearProblemx hg(num_vertices_);

		for (int i = begin; i < end; ++i)
		{
			Vec3f jdepth = jdepth_texture_[i];
			Vec3<PidType> pids = pids_texture_[i];
			ImageType image = image_texture_[i];
			ImageType ref = ref_texture_[i];

			if (image == image_texture_.nodata() ||
				ref == ref_texture_.nodata() ||
				jdepth == jdepth_texture_.nodata() ||
				pids == pids_texture_.nodata())
				continue;

			float res = ref - image;
			const Vec3i pids_(pids(0), pids(1), pids(2));
			// check if there is blending of different triangles
			float pid0_diff = std::abs(pids_(0) - pids(0));
			float pid1_diff = std::abs(pids_(1) - pids(1));
			float pid2_diff = std::abs(pids_(2) - pids(2));
			if (pid0_diff > 0 || pid1_diff > 0 || pid2_diff > 0)
				continue;
			const float depth0 = depths_[pids_(0)];
			const float depth1 = depths_[pids_(1)];
			const float depth2 = depths_[pids_(2)];
			const float d_depth0_d_param = d_depth_d_param(depth0);
			const float d_depth1_d_param = d_depth_d_param(depth1);
			const float d_depth2_d_param = d_depth_d_param(depth2);
			Vec3f J;
			J(0) = jdepth(0) * d_depth0_d_param;
			J(1) = jdepth(1) * d_depth1_d_param;
			J(2) = jdepth(2) * d_depth2_d_param;

			const float w = huber_weight(res, mesh_vo::huber_thresh_pix);

			hg.add(J, res, w, pids_);
		}
		return hg;
	}

private:
	TextureViewRead<Vec3f> jdepth_texture_;
	TextureViewRead<Vec3<PidType>> pids_texture_;
	TextureViewRead<ImageType> image_texture_;
	TextureViewRead<ImageType> ref_texture_;

	std::vector<float> depths_;

	int num_frames_;
	int num_vertices_;
	int frame_id_;
};

// Pose-only Jacobian -> DenseLinearProblem reducer
class HGVertexReducerCPU : public BaseReducerCPU<HGVertexReducerCPU, DenseLinearProblemx>
{
public:
	using Base = BaseReducerCPU<HGVertexReducerCPU, DenseLinearProblemx>;
	explicit HGVertexReducerCPU()
		: Base()
	{
	}

	void reduce(int lvl, int frame_id, int num_frames, int num_vertices, const Texture<Vec3f> &jv0_texture, const Texture<Vec3f> &jv1_texture, const Texture<Vec3f> &jv2_texture, const Texture<Vec3<PidType>> &pids_texture, const Texture<ImageType> &image_texture, const Texture<ImageType> &ref_texture, const Mesh &mesh, DenseLinearProblemx &total)
	{
		jv0_texture_ = jv0_texture.MapRead(lvl);
		jv1_texture_ = jv1_texture.MapRead(lvl);
		jv2_texture_ = jv2_texture.MapRead(lvl);
		pids_texture_ = pids_texture.MapRead(lvl);
		image_texture_ = image_texture.MapRead(lvl);
		ref_texture_ = ref_texture.MapRead(lvl);

		num_frames_ = num_frames;
		num_vertices_ = num_vertices;
		frame_id_ = frame_id;

		reduce_(total);
	}

	int size()
	{
		return image_texture_.width() * image_texture_.height();
	}

	DenseLinearProblemx reducepartial(int begin, int end)
	{
		DenseLinearProblemx hg(num_vertices_ * 3);

		for (int i = begin; i < end; ++i)
		{
			Vec3f jv0 = jv0_texture_[i];
			Vec3f jv1 = jv1_texture_[i];
			Vec3f jv2 = jv2_texture_[i];
			Vec3<PidType> pids = pids_texture_[i];
			ImageType image = image_texture_[i];
			ImageType ref = ref_texture_[i];

			if (image == image_texture_.nodata() ||
				ref == ref_texture_.nodata() ||
				jv0 == jv0_texture_.nodata() ||
				jv1 == jv1_texture_.nodata() ||
				jv2 == jv2_texture_.nodata() ||
				pids == pids_texture_.nodata())
				continue;

			float res = ref - image;
			int pid0 = int(pids(0));
			int pid1 = int(pids(1));
			int pid2 = int(pids(2));

			// check if there is blending of different triangles
			float pid0_diff = std::abs(pid0 - pids(0));
			float pid1_diff = std::abs(pid1 - pids(1));
			float pid2_diff = std::abs(pid2 - pids(2));
			if (pid0_diff > 0 || pid1_diff > 0 || pid2_diff > 0)
				continue;

			const float w = huber_weight(res, mesh_vo::huber_thresh_pix);

			Vec<float, 9> J;
			J(0) = jv0(0);
			J(1) = jv0(1);
			J(2) = jv0(2);
			J(3) = jv1(0);
			J(4) = jv1(1);
			J(5) = jv1(2);
			J(6) = jv2(0);
			J(7) = jv2(1);
			J(8) = jv2(2);

			Vec<int, 9> ids;
			ids(0) = pid0 * 3 + 0;
			ids(1) = pid0 * 3 + 1;
			ids(2) = pid0 * 3 + 2;
			ids(3) = pid1 * 3 + 0;
			ids(4) = pid1 * 3 + 1;
			ids(5) = pid1 * 3 + 2;
			ids(6) = pid2 * 3 + 0;
			ids(7) = pid2 * 3 + 1;
			ids(8) = pid2 * 3 + 2;

			hg.add(J, res, w, ids);
		}
		return hg;
	}

private:
	TextureViewRead<Vec3f> jv0_texture_;
	TextureViewRead<Vec3f> jv1_texture_;
	TextureViewRead<Vec3f> jv2_texture_;
	TextureViewRead<Vec3<PidType>> pids_texture_;
	TextureViewRead<ImageType> image_texture_;
	TextureViewRead<ImageType> ref_texture_;

	int num_frames_;
	int num_vertices_;
	int frame_id_;
};

// Pose-only Jacobian -> DenseLinearProblem reducer
class HGDepthExpReducerCPU : public BaseReducerCPU<HGDepthExpReducerCPU, DenseLinearProblemx>
{
public:
	using Base = BaseReducerCPU<HGDepthExpReducerCPU, DenseLinearProblemx>;
	explicit HGDepthExpReducerCPU()
		: Base()
	{
	}

	void reduce(int lvl, int frame_id, int num_frames, int num_vertices, const Texture<Vec3f> &jdepth_texture, const Texture<Vec3f> &jexp_texture, const Texture<Vec3<PidType>> &pids_texture, const Texture<ImageType> &image_texture, const Texture<ImageType> &ref_texture, const Mesh &mesh, DenseLinearProblemx &total)
	{
		jdepth_texture_ = jdepth_texture.MapRead(lvl);
		jexp_texture_ = jexp_texture.MapRead(lvl);
		pids_texture_ = pids_texture.MapRead(lvl);
		image_texture_ = image_texture.MapRead(lvl);
		ref_texture_ = ref_texture.MapRead(lvl);

		depths = get_depths(mesh);

		num_frames_ = num_frames;
		num_vertices_ = num_vertices;
		frame_id_ = frame_id;

		reduce_(total);
	}

	int size()
	{
		return image_texture_.width() * image_texture_.height();
	}

	DenseLinearProblemx reducepartial(int begin, int end)
	{
		DenseLinearProblemx hg(num_vertices_ + num_frames_ * 2);

		// Vec6i ids(0, 1, 2, 3, 4, 5);

		for (int i = begin; i < end; ++i)
		{
			Vec3f jdepth = jdepth_texture_[i];
			Vec3f jexp = jexp_texture_[i];
			Vec3<PidType> pids = pids_texture_[i];
			ImageType image = image_texture_[i];
			ImageType ref = ref_texture_[i];

			if (image == image_texture_.nodata() ||
				ref == ref_texture_.nodata() ||
				jdepth == jdepth_texture_.nodata() ||
				jexp == jexp_texture_.nodata() ||
				pids == pids_texture_.nodata())
				continue;

			float res = ref - image;
			const Vec5i pids_(pids(0), pids(1), pids(2), num_vertices_ + frame_id_ * 2, num_vertices_ + frame_id_ * 2 + 1);
			const float depth0 = depths[pids_(0)];
			const float depth1 = depths[pids_(1)];
			const float depth2 = depths[pids_(2)];
			const float d_depth0_d_param = d_depth_d_param(depth0);
			const float d_depth1_d_param = d_depth_d_param(depth1);
			const float d_depth2_d_param = d_depth_d_param(depth2);
			Vec5f J;
			J(0) = jdepth(0) * d_depth0_d_param;
			J(1) = jdepth(1) * d_depth1_d_param;
			J(2) = jdepth(2) * d_depth2_d_param;
			J(3) = jexp(0);
			J(4) = jexp(1);

			const float w = huber_weight(res, mesh_vo::huber_thresh_pix);

			hg.add(J, res, w, pids_);
		}

		return hg;
	}

private:
	TextureViewRead<Vec3f> jdepth_texture_;
	TextureViewRead<Vec3f> jexp_texture_;
	TextureViewRead<Vec3<PidType>> pids_texture_;
	TextureViewRead<ImageType> image_texture_;
	TextureViewRead<ImageType> ref_texture_;
	std::vector<float> depths;
	int num_frames_;
	int num_vertices_;
	int frame_id_;
};

class HGPoseDepthReducerCPU : public BaseReducerCPU<HGPoseDepthReducerCPU, DenseLinearProblemx>
{
public:
	using Base = BaseReducerCPU<HGPoseDepthReducerCPU, DenseLinearProblemx>;
	explicit HGPoseDepthReducerCPU()
		: Base()
	{
	}

	void reduce(int lvl, int frame_id, int num_frames, int num_vertices, const Texture<Vec3f> &jtra_texture, const Texture<Vec3f> &jrot_texture, const Texture<Vec3f> &jdepth_texture, const Texture<Vec3<PidType>> &pids_texture, const Texture<ImageType> &image_texture, const Texture<ImageType> &ref_texture, const Mesh &mesh, DenseLinearProblemx &total)
	{
		jtra_texture_ = jtra_texture.MapRead(lvl);
		jrot_texture_ = jrot_texture.MapRead(lvl);
		jdepth_texture_ = jdepth_texture.MapRead(lvl);
		pids_texture_ = pids_texture.MapRead(lvl);
		image_texture_ = image_texture.MapRead(lvl);
		ref_texture_ = ref_texture.MapRead(lvl);
		depths_ = get_depths(mesh);

		num_frames_ = num_frames;
		num_vertices_ = num_vertices;
		frame_id_ = frame_id;

		reduce_(total);
	}

	int size()
	{
		return image_texture_.width() * image_texture_.height();
	}

	DenseLinearProblemx reducepartial(int begin, int end)
	{
		DenseLinearProblemx hg(num_vertices_ + num_frames_ * 6);

		// Vec6i ids(0, 1, 2, 3, 4, 5);

		for (int i = begin; i < end; ++i)
		{
			Vec3f jtra = jtra_texture_[i];
			Vec3f jrot = jrot_texture_[i];
			Vec3f jdepth = jdepth_texture_[i];
			Vec3<PidType> pids = pids_texture_[i];
			ImageType image = image_texture_[i];
			ImageType ref = ref_texture_[i];

			if (image == image_texture_.nodata() ||
				ref == ref_texture_.nodata() ||
				jtra == jtra_texture_.nodata() ||
				jrot == jrot_texture_.nodata() ||
				jdepth == jdepth_texture_.nodata() ||
				pids == pids_texture_.nodata())
				continue;

			float res = ref - image;

			Vec<int, 9> pids_;
			pids_(0) = pids(0);
			pids_(1) = pids(1);
			pids_(2) = pids(2);
			pids_(3) = num_vertices_ + frame_id_ * 6;
			pids_(4) = num_vertices_ + frame_id_ * 6 + 1;
			pids_(5) = num_vertices_ + frame_id_ * 6 + 2;
			pids_(6) = num_vertices_ + frame_id_ * 6 + 3;
			pids_(7) = num_vertices_ + frame_id_ * 6 + 4;
			pids_(8) = num_vertices_ + frame_id_ * 6 + 5;

			const float depth0 = depths_[pids_(0)];
			const float depth1 = depths_[pids_(1)];
			const float depth2 = depths_[pids_(2)];
			const float d_depth0_d_param = d_depth_d_param(depth0);
			const float d_depth1_d_param = d_depth_d_param(depth1);
			const float d_depth2_d_param = d_depth_d_param(depth2);
			Vec<float, 9> J;
			J(0) = jdepth(0) * d_depth0_d_param;
			J(1) = jdepth(1) * d_depth1_d_param;
			J(2) = jdepth(2) * d_depth2_d_param;
			J(3) = jtra(0);
			J(4) = jtra(1);
			J(5) = jtra(2);
			J(6) = jrot(0);
			J(7) = jrot(1);
			J(8) = jrot(2);

			const float w = huber_weight(res, mesh_vo::huber_thresh_pix);

			hg.add(J, res, w, pids_);
		}
		return hg;
	}

private:
	TextureViewRead<Vec3f> jtra_texture_;
	TextureViewRead<Vec3f> jrot_texture_;
	TextureViewRead<Vec3f> jdepth_texture_;
	TextureViewRead<Vec3<PidType>> pids_texture_;
	TextureViewRead<ImageType> image_texture_;
	TextureViewRead<ImageType> ref_texture_;

	std::vector<float> depths_;
	int num_frames_;
	int num_vertices_;
	int frame_id_;
};

class HGPoseExpDepthReducerCPU : public BaseReducerCPU<HGPoseExpDepthReducerCPU, DenseLinearProblemx>
{
public:
	using Base = BaseReducerCPU<HGPoseExpDepthReducerCPU, DenseLinearProblemx>;
	explicit HGPoseExpDepthReducerCPU()
		: Base()
	{
	}

	void reduce(int lvl, int frame_id, int num_frames, int num_vertices, const Texture<Vec3f> &jtra_texture, const Texture<Vec3f> &jrot_texture, const Texture<Vec3f> &jexp_texture, const Texture<Vec3f> &jdepth_texture, const Texture<Vec3<PidType>> &pids_texture, const Texture<ImageType> &image_texture, const Texture<ImageType> &ref_texture, const Mesh &mesh, DenseLinearProblemx &total)
	{
		jtra_texture_ = jtra_texture.MapRead(lvl);
		jrot_texture_ = jrot_texture.MapRead(lvl);
		jexp_texture_ = jexp_texture.MapRead(lvl);
		jdepth_texture_ = jdepth_texture.MapRead(lvl);
		pids_texture_ = pids_texture.MapRead(lvl);
		image_texture_ = image_texture.MapRead(lvl);
		ref_texture_ = ref_texture.MapRead(lvl);

		depths = get_depths(mesh);

		num_frames_ = num_frames;
		num_vertices_ = num_vertices;
		frame_id_ = frame_id;

		reduce_(total);
	}

	int size()
	{
		return image_texture_.width() * image_texture_.height();
	}

	DenseLinearProblemx reducepartial(int begin, int end)
	{
		DenseLinearProblemx hg(num_vertices_ + num_frames_ * 8);

		// Vec6i ids(0, 1, 2, 3, 4, 5);

		for (int i = begin; i < end; ++i)
		{
			Vec3f jtra = jtra_texture_[i];
			Vec3f jrot = jrot_texture_[i];
			Vec3f jexp = jexp_texture_[i];
			Vec3f jdepth = jdepth_texture_[i];
			Vec3<PidType> pids = pids_texture_[i];
			ImageType image = image_texture_[i];
			ImageType ref = ref_texture_[i];

			if (image == image_texture_.nodata() ||
				ref == ref_texture_.nodata() ||
				jtra == jtra_texture_.nodata() ||
				jrot == jrot_texture_.nodata() ||
				jexp == jexp_texture_.nodata() ||
				jdepth == jdepth_texture_.nodata() ||
				pids == pids_texture_.nodata())
				continue;

			float res = ref - image;

			Vec<int, 11> pids_;
			pids_(0) = pids(0);
			pids_(1) = pids(1);
			pids_(2) = pids(2);
			pids_(3) = num_vertices_ + frame_id_ * 8;
			pids_(4) = num_vertices_ + frame_id_ * 8 + 1;
			pids_(5) = num_vertices_ + frame_id_ * 8 + 2;
			pids_(6) = num_vertices_ + frame_id_ * 8 + 3;
			pids_(7) = num_vertices_ + frame_id_ * 8 + 4;
			pids_(8) = num_vertices_ + frame_id_ * 8 + 5;
			pids_(9) = num_vertices_ + frame_id_ * 8 + 6;
			pids_(10) = num_vertices_ + frame_id_ * 8 + 7;

			const float depth0 = depths[pids_(0)];
			const float depth1 = depths[pids_(1)];
			const float depth2 = depths[pids_(2)];
			const float d_depth0_d_param = d_depth_d_param(depth0);
			const float d_depth1_d_param = d_depth_d_param(depth1);
			const float d_depth2_d_param = d_depth_d_param(depth2);
			Vec<float, 11> J;
			J(0) = jdepth(0) * d_depth0_d_param;
			J(1) = jdepth(1) * d_depth1_d_param;
			J(2) = jdepth(2) * d_depth2_d_param;
			J(3) = jtra(0);
			J(4) = jtra(1);
			J(5) = jtra(2);
			J(6) = jrot(0);
			J(7) = jrot(1);
			J(8) = jrot(2);
			J(9) = jexp(0);
			J(10) = jexp(1);

			const float w = huber_weight(res, mesh_vo::huber_thresh_pix);

			hg.add(J, res, w, pids_);
		}
		return hg;
	}

private:
	TextureViewRead<Vec3f> jtra_texture_;
	TextureViewRead<Vec3f> jrot_texture_;
	TextureViewRead<Vec3f> jexp_texture_;
	TextureViewRead<Vec3f> jdepth_texture_;
	TextureViewRead<Vec3<PidType>> pids_texture_;
	TextureViewRead<ImageType> image_texture_;
	TextureViewRead<ImageType> ref_texture_;

	std::vector<float> depths;
	int num_frames_;
	int num_vertices_;
	int frame_id_;
};

/*
// ===== Map Jacobian container with fixed arity K per observation =====
template <int K>
struct MapJacobianBlock
{
	Eigen::Matrix<float, K, 1> J; // values
	Eigen::Matrix<int, K, 1> ids; // global parameter indices
	uint8_t nnz{0};				  // number of valid entries in [0, K]
};

// Map-only reducer: residuals + map jacobians (sparse with ids)
template <int K>
class HGMapReducerCPU : public BaseReducerCPU<float, float, MapJacobianBlock<K>, DenseLinearProblem>
{
public:
	using Block = MapJacobianBlock<K>;
	explicit HGMapReducerCPU(int num_map_params,
							 unsigned threads = std::max(1u, std::thread::hardware_concurrency()))
		: num_map_params_(num_map_params), BaseReducerCPU<float, float, Block, DenseLinearProblem>(threads) {}

protected:
	DenseLinearProblem reducepartial(int begin, int end,
									 const TextureCPU<float> &residuals,
									 const TextureCPU<float> & */
/*unused*/ /*,
const TextureCPU<Block> &jmap,
int lvl) override
{
DenseLinearProblem hg(num_map_params_);
auto rbuf = residuals.MapRead(lvl);
auto mbuf = jmap.MapRead(lvl);

Eigen::Matrix<float, K, 1> Jtmp;
Eigen::Matrix<int, K, 1> Itmp;

for (int i = begin; i < end; ++i)
{
const float res = rbuf[i];
if (res == residuals.nodata())
continue;
const float w = BaseReducerCPU<float, float, Block, DenseLinearProblem>::huber_weight(res, mesh_vo::huber_thresh_pix);
const Block &b = mbuf[i];
const int m = static_cast<int>(b.nnz);
if (m <= 0)
continue;
Jtmp.head(m) = b.J.head(m);
Itmp.head(m) = b.ids.head(m);
hg.add(Jtmp.head(m), res, w, Itmp.head(m));
}
return hg;
}

private:
int num_map_params_;
};
*/

/*
// Joint pose+map reducer: residuals + pose jacobians + map jacobians
template <int K>
class HGPoseMapReducerCPU : public BaseReducerCPU<float, Vec6, MapJacobianBlock<K>, DenseLinearProblem>
{
public:
	using Block = MapJacobianBlock<K>;
	explicit HGPoseMapReducerCPU(int num_map_params,
								 unsigned threads = std::max(1u, std::thread::hardware_concurrency()))
		: num_map_params_(num_map_params), BaseReducerCPU<float, Vec6, Block, DenseLinearProblem>(threads) {}

protected:
	DenseLinearProblem reducepartial(int begin, int end,
									 const TextureCPU<float> &residuals,
									 const TextureCPU<Vec6> &jpose,
									 const TextureCPU<Block> &jmap,
									 int lvl) override
	{
		const int N = 6 + num_map_params_;
		DenseLinearProblem hg(N);
		auto rbuf = residuals.MapRead(lvl);
		auto pbuf = jpose.MapRead(lvl);
		auto mbuf = jmap.MapRead(lvl);

		Vec6i poseIds;
		poseIds << 0, 1, 2, 3, 4, 5;
		Eigen::Matrix<float, K, 1> Jm;
		Eigen::Matrix<int, K, 1> Im;

		for (int i = begin; i < end; ++i)
		{
			const float res = rbuf[i];
			if (res == residuals.nodata())
				continue;
			const float w = BaseReducerCPU<float, Vec6, Block, DenseLinearProblem>::huber_weight(res, mesh_vo::huber_thresh_pix);

			// Pose block
			const Vec6 Jp = pbuf[i];
			hg.add(Jp, res, w, poseIds);

			// Map block with offset
			const Block &b = mbuf[i];
			const int m = static_cast<int>(b.nnz);
			if (m > 0)
			{
				for (int t = 0; t < m; ++t)
					Im(t) = b.ids(t) + 6; // offset after pose
				Jm.head(m) = b.J.head(m);
				hg.add(Jm.head(m), res, w, Im.head(m));
			}
		}
		return hg;
	}

private:
	int num_map_params_;
};
*/
