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
	BaseReducerCPU() noexcept : threads_(1 /*std::max(1u, std::thread::hardware_concurrency())*/) {}
	explicit BaseReducerCPU(unsigned threads) noexcept : threads_(threads == 0 ? 1u : threads) {}
	virtual ~BaseReducerCPU() = default;

protected:
	void reduce_(OutType &total)
	{
		const int N = derived_().size();
		derived_().reducepartial(0, N, total);

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
			pool.emplace_back([&, begin, end]()
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
		OutType total = partial[0];
		for (unsigned t = 1; t < T; ++t)
			total += partial[t];
		return total;
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
	explicit NodataReducerCPU(unsigned threads = 1 /*std::max(1u, std::thread::hardware_concurrency())*/) : Base(threads) {}

	void reduce(int lvl, const Texture<ImageType> &r_texture, Error &total)
	{
		r_texture_ = &r_texture;
		lvl_ = lvl;

		reduce_(total);
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	void reducepartial(int begin, int end, Error &err)
	{
		auto rmap = r_texture_->MapRead(lvl_);

		for (int i = begin; i < end; ++i)
		{
			const float r = rmap[i];
			if (r == r_texture_->nodata())
				err += 1.0f;
			else
				err += 0.0f;
		}
	}

private:
	const Texture<ImageType> *r_texture_;
	int lvl_;
};

// Photometric L2 with Huber loss. Third input unused.
class ResidualReducerCPU : public BaseReducerCPU<ResidualReducerCPU, Error>
{
public:
	using Base = BaseReducerCPU<ResidualReducerCPU, Error>;
	explicit ResidualReducerCPU(unsigned threads = 1 /*std::max(1u, std::thread::hardware_concurrency())*/)
		: Base(threads)
	{
	}

	void reduce(int lvl, const Texture<float> &r_texture, Error &total)
	{
		r_texture_ = &r_texture;
		lvl_ = lvl;

		reduce_(total);
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	void reducepartial(int begin, int end, Error &err)
	{
		auto rmap = r_texture_->MapRead(lvl_);

		for (int i = begin; i < end; ++i)
		{
			const float r = rmap[i];
			if (r == r_texture_->nodata())
				continue;
			const float w = huber_weight(r, mesh_vo::huber_thresh_pix);
			err += w * r * r;
		}
	}

private:
	const Texture<float> *r_texture_;
	int lvl_;
};

// Pose-only Jacobian -> DenseLinearProblem reducer
class HGPoseReducerCPU : public BaseReducerCPU<HGPoseReducerCPU, DenseLinearProblem<6>>
{
public:
	using Base = BaseReducerCPU<HGPoseReducerCPU, DenseLinearProblem<6>>;
	explicit HGPoseReducerCPU(unsigned threads = 1 /*std::max(1u, std::thread::hardware_concurrency())*/)
		: Base(threads)
	{
	}

	void reduce(int lvl, const Texture<Vec3f> &jtra_texture, const Texture<Vec3f> &jrot_texture, const Texture<float> &r_texture, DenseLinearProblem<6> &total)
	{
		jtra_texture_ = &jtra_texture;
		jrot_texture_ = &jrot_texture;
		r_texture_ = &r_texture;
		lvl_ = lvl;

		reduce_(total);
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	void reducepartial(int begin, int end, DenseLinearProblem<6> &hg)
	{
		auto r_map = r_texture_->MapRead(lvl_);
		auto jtra_map = jtra_texture_->MapRead(lvl_);
		auto jrot_map = jrot_texture_->MapRead(lvl_);
		// Vec8i ids(0, 1, 2, 3, 4, 5, 6, 7);
		Vec8i ids;
		for (int i = 0; i < 8; i++)
			ids(i) = i;

		for (int i = begin; i < end; ++i)
		{
			const float res = r_map[i];
			const Vec3f jtra = jtra_map[i];
			const Vec3f jrot = jrot_map[i];
			if (res == r_texture_->nodata() || jtra == jtra_texture_->nodata() || jrot == jrot_texture_->nodata())
				continue;
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
	}

private:
	const Texture<Vec3f> *jtra_texture_;
	const Texture<Vec3f> *jrot_texture_;
	const Texture<float> *r_texture_;
	int lvl_;
	// DenseLinearProblem<6> hg_;
};

// Pose-only Jacobian -> DenseLinearProblem reducer
class HGPoseExpReducerCPU : public BaseReducerCPU<HGPoseExpReducerCPU, DenseLinearProblem<8>>
{
public:
	using Base = BaseReducerCPU<HGPoseExpReducerCPU, DenseLinearProblem<8>>;
	explicit HGPoseExpReducerCPU(unsigned threads = 1 /*std::max(1u, std::thread::hardware_concurrency())*/)
		: Base(threads)
	{
	}

	void reduce(int lvl, const Texture<Vec3f> &jtra_texture, const Texture<Vec3f> &jrot_texture, const Texture<Vec3f> &jexp_texture, const Texture<float> &r_texture, DenseLinearProblem<8> &total)
	{
		jtra_texture_ = &jtra_texture;
		jrot_texture_ = &jrot_texture;
		jexp_texture_ = &jexp_texture;
		r_texture_ = &r_texture;
		lvl_ = lvl;

		reduce_(total);
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	void reducepartial(int begin, int end, DenseLinearProblem<8> &hg)
	{
		auto r_map = r_texture_->MapRead(lvl_);
		auto jtra_map = jtra_texture_->MapRead(lvl_);
		auto jrot_map = jrot_texture_->MapRead(lvl_);
		auto jexp_map = jexp_texture_->MapRead(lvl_);
		// Vec8i ids(0, 1, 2, 3, 4, 5, 6, 7);
		Vec8i ids;
		for (int i = 0; i < 8; i++)
			ids(i) = i;

		for (int i = begin; i < end; ++i)
		{
			const float res = r_map[i];
			const Vec3f jtra = jtra_map[i];
			const Vec3f jrot = jrot_map[i];
			const Vec3f jexp = jexp_map[i];
			if (res == r_texture_->nodata() || jtra == jtra_texture_->nodata() || jrot == jrot_texture_->nodata() || jexp == jexp_texture_->nodata())
				continue;
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
	}

private:
	const Texture<Vec3f> *jtra_texture_;
	const Texture<Vec3f> *jrot_texture_;
	const Texture<Vec3f> *jexp_texture_;
	const Texture<float> *r_texture_;
	int lvl_;
	// DenseLinearProblem<6> hg_;
};

// Pose-only Jacobian -> DenseLinearProblem reducer
class HGPoseVelReducerCPU : public BaseReducerCPU<HGPoseVelReducerCPU, DenseLinearProblem<12>>
{
public:
	using Base = BaseReducerCPU<HGPoseVelReducerCPU, DenseLinearProblem<12>>;
	explicit HGPoseVelReducerCPU(unsigned threads = 1 /*std::max(1u, std::thread::hardware_concurrency())*/)
		: Base(threads)
	{
	}

	void reduce(int lvl, const Texture<Vec3f> &jtra_texture, const Texture<Vec3f> &jrot_texture, const Texture<Vec3f> &jtravel_texture, const Texture<Vec3f> &jrotvel_texture, const Texture<float> &r_texture, DenseLinearProblem<12> &total)
	{
		jtra_texture_ = &jtra_texture;
		jrot_texture_ = &jrot_texture;
		jtravel_texture_ = &jtravel_texture;
		jrotvel_texture_ = &jrotvel_texture;
		r_texture_ = &r_texture;
		lvl_ = lvl;

		return reduce_(total);
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	void reducepartial(int begin, int end, DenseLinearProblem<12> &hg)
	{
		auto r_map = r_texture_->MapRead(lvl_);
		auto jtra_map = jtra_texture_->MapRead(lvl_);
		auto jrot_map = jrot_texture_->MapRead(lvl_);
		auto jtravel_map = jtravel_texture_->MapRead(lvl_);
		auto jrotvel_map = jrotvel_texture_->MapRead(lvl_);
		// Vec8i ids(0, 1, 2, 3, 4, 5, 6, 7);
		Vec<int, 12> ids;
		for (int i = 0; i < 12; i++)
			ids(i) = i;

		for (int i = begin; i < end; ++i)
		{
			const float res = r_map[i];
			const Vec3f jtra = jtra_map[i];
			const Vec3f jrot = jrot_map[i];
			const Vec3f jtravel = jtravel_map[i];
			const Vec3f jrotvel = jrotvel_map[i];
			if (res == r_texture_->nodata() || jtra == jtra_texture_->nodata() || jrot == jrot_texture_->nodata() || jtravel == jtravel_texture_->nodata() || jrotvel == jrotvel_texture_->nodata())
				continue;
			// Vec8f J(jtra(0), jtra(1), jtra(2), jrot(0), jrot(1), jrot(2), jexp(0), jexp(1));
			Vec<float, 12> J;
			J(0) = jtra(0);
			J(1) = jtra(1);
			J(2) = jtra(2);
			J(3) = jrot(0);
			J(4) = jrot(1);
			J(5) = jrot(2);
			J(6) = jtravel(0);
			J(7) = jtravel(1);
			J(8) = jtravel(2);
			J(9) = jrotvel(0);
			J(10) = jrotvel(1);
			J(11) = jrotvel(2);

			const float w = huber_weight(res, mesh_vo::huber_thresh_pix);

			// hg.add(J, res, w, ids);
			hg.add(J, res, w);
		}
	}

private:
	const Texture<Vec3f> *jtra_texture_;
	const Texture<Vec3f> *jrot_texture_;
	const Texture<Vec3f> *jtravel_texture_;
	const Texture<Vec3f> *jrotvel_texture_;
	const Texture<float> *r_texture_;
	int lvl_;
	// DenseLinearProblem<6> hg_;
};

// Pose-only Jacobian -> DenseLinearProblem reducer
class HGPoseVelExpReducerCPU : public BaseReducerCPU<HGPoseVelExpReducerCPU, DenseLinearProblem<14>>
{
public:
	using Base = BaseReducerCPU<HGPoseVelExpReducerCPU, DenseLinearProblem<14>>;
	explicit HGPoseVelExpReducerCPU(unsigned threads = 1 /*std::max(1u, std::thread::hardware_concurrency())*/)
		: Base(threads)
	{
	}

	void reduce(int lvl, const Texture<Vec3f> &jtra_texture, const Texture<Vec3f> &jrot_texture, const Texture<Vec3f> &jtravel_texture, const Texture<Vec3f> &jrotvel_texture, const Texture<Vec3f> &jexp_texture, const Texture<float> &r_texture, DenseLinearProblem<14> &total)
	{
		jtra_texture_ = &jtra_texture;
		jrot_texture_ = &jrot_texture;
		jtravel_texture_ = &jtravel_texture;
		jrotvel_texture_ = &jrotvel_texture;
		jexp_texture_ = &jexp_texture;
		r_texture_ = &r_texture;
		lvl_ = lvl;

		reduce_(total);
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	void reducepartial(int begin, int end, DenseLinearProblem<14> &hg)
	{
		auto r_map = r_texture_->MapRead(lvl_);
		auto jtra_map = jtra_texture_->MapRead(lvl_);
		auto jrot_map = jrot_texture_->MapRead(lvl_);
		auto jtravel_map = jtravel_texture_->MapRead(lvl_);
		auto jrotvel_map = jrotvel_texture_->MapRead(lvl_);
		auto jexp_map = jexp_texture_->MapRead(lvl_);
		// Vec8i ids(0, 1, 2, 3, 4, 5, 6, 7);
		Vec<int, 14> ids;
		for (int i = 0; i < 14; i++)
			ids(i) = i;

		for (int i = begin; i < end; ++i)
		{
			const float res = r_map[i];
			const Vec3f jtra = jtra_map[i];
			const Vec3f jrot = jrot_map[i];
			const Vec3f jtravel = jtravel_map[i];
			const Vec3f jrotvel = jrotvel_map[i];
			const Vec3f jexp = jexp_map[i];

			if (res == r_texture_->nodata() || jtra == jtra_texture_->nodata() || jrot == jrot_texture_->nodata() || jtravel == jtravel_texture_->nodata() || jrotvel == jrotvel_texture_->nodata() || jexp == jexp_texture_->nodata())
				continue;
			// Vec8f J(jtra(0), jtra(1), jtra(2), jrot(0), jrot(1), jrot(2), jexp(0), jexp(1));
			Vec<float, 14> J;
			J(0) = jtra(0);
			J(1) = jtra(1);
			J(2) = jtra(2);
			J(3) = jrot(0);
			J(4) = jrot(1);
			J(5) = jrot(2);
			J(6) = jtravel(0);
			J(7) = jtravel(1);
			J(8) = jtravel(2);
			J(9) = jrotvel(0);
			J(10) = jrotvel(1);
			J(11) = jrotvel(2);
			J(12) = jexp(0);
			J(13) = jexp(1);

			const float w = huber_weight(res, mesh_vo::huber_thresh_pix);

			// hg.add(J, res, w, ids);
			hg.add(J, res, w);
		}
	}

private:
	const Texture<Vec3f> *jtra_texture_;
	const Texture<Vec3f> *jrot_texture_;
	const Texture<Vec3f> *jtravel_texture_;
	const Texture<Vec3f> *jrotvel_texture_;
	const Texture<Vec3f> *jexp_texture_;
	const Texture<float> *r_texture_;
	int lvl_;
	// DenseLinearProblem<6> hg_;
};

// Pose-only Jacobian -> DenseLinearProblem reducer
class HGMapReducerCPU : public BaseReducerCPU<HGMapReducerCPU, DenseLinearProblemx>
{
public:
	using Base = BaseReducerCPU<HGMapReducerCPU, DenseLinearProblemx>;
	explicit HGMapReducerCPU(unsigned threads = 1 /*std::max(1u, std::thread::hardware_concurrency())*/)
		: Base(threads)
	{
	}

	void reduce(int lvl, int frame_id, int num_frames, int num_vertices, const Texture<Vec3f> &jmap_texture, const Texture<Vec3<PidType>> &pids_texture, const Texture<float> &r_texture, const Mesh &mesh, DenseLinearProblemx &total)
	{
		jmap_texture_ = &jmap_texture;
		pids_texture_ = &pids_texture;
		r_texture_ = &r_texture;
		mesh_ = &mesh;

		lvl_ = lvl;
		num_frames_ = num_frames;
		num_vertices_ = num_vertices;
		frame_id_ = frame_id;

		// DenseLinearProblem hg(total);

		reduce_(total);
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	void reducepartial(int begin, int end, DenseLinearProblemx &hg)
	{
		auto r_map = r_texture_->MapRead(lvl_);
		auto jmap_map = jmap_texture_->MapRead(lvl_);
		auto pids_map = pids_texture_->MapRead(lvl_);
		auto depths = get_depths(*mesh_);
		// Vec6i ids(0, 1, 2, 3, 4, 5);

		for (int i = begin; i < end; ++i)
		{
			Vec3f jmap = jmap_map[i];
			const Vec3<PidType> pids = pids_map[i];
			const float res = r_map[i];

			if (res == r_texture_->nodata() || jmap == jmap_texture_->nodata() || pids == pids_texture_->nodata())
				continue;

			const Vec3i pids_(pids(0), pids(1), pids(2));
			const float depth0 = depths[pids_(0)];
			const float depth1 = depths[pids_(1)];
			const float depth2 = depths[pids_(2)];
			const float d_depth0_d_param = d_depth_d_param(depth0);
			const float d_depth1_d_param = d_depth_d_param(depth1);
			const float d_depth2_d_param = d_depth_d_param(depth2);
			Vec3f J;
			J(0) = jmap(0) * d_depth0_d_param;
			J(1) = jmap(1) * d_depth1_d_param;
			J(2) = jmap(2) * d_depth2_d_param;

			const float w = huber_weight(res, mesh_vo::huber_thresh_pix);

			hg.add(J, res, w, pids_);
		}
	}

private:
	const Texture<Vec3f> *jmap_texture_;
	const Texture<Vec3<PidType>> *pids_texture_;
	const Texture<float> *r_texture_;
	const Mesh *mesh_;
	int lvl_;
	int num_frames_;
	int num_vertices_;
	int frame_id_;
	// DenseLinearProblemx hg_;
};

// Pose-only Jacobian -> DenseLinearProblem reducer
class HGMapExpReducerCPU : public BaseReducerCPU<HGMapExpReducerCPU, DenseLinearProblemx>
{
public:
	using Base = BaseReducerCPU<HGMapExpReducerCPU, DenseLinearProblemx>;
	explicit HGMapExpReducerCPU(unsigned threads = 1 /*std::max(1u, std::thread::hardware_concurrency())*/)
		: Base(threads)
	{
	}

	void reduce(int lvl, int frame_id, int num_frames, int num_vertices, const Texture<Vec3f> &jmap_texture, const Texture<Vec3f> &jexp_texture, const Texture<Vec3<PidType>> &pids_texture, const Texture<float> &r_texture, const Mesh &mesh, DenseLinearProblemx &total)
	{
		jmap_texture_ = &jmap_texture;
		jexp_texture_ = &jexp_texture;
		pids_texture_ = &pids_texture;
		r_texture_ = &r_texture;
		mesh_ = &mesh;

		lvl_ = lvl;
		num_frames_ = num_frames;
		num_vertices_ = num_vertices;
		frame_id_ = frame_id;

		// DenseLinearProblem hg(total);

		reduce_(total);
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	void reducepartial(int begin, int end, DenseLinearProblemx &hg)
	{
		auto r_map = r_texture_->MapRead(lvl_);
		auto jmap_map = jmap_texture_->MapRead(lvl_);
		auto jexp_map = jexp_texture_->MapRead(lvl_);
		auto pids_map = pids_texture_->MapRead(lvl_);
		auto depths = get_depths(*mesh_);
		// Vec6i ids(0, 1, 2, 3, 4, 5);

		for (int i = begin; i < end; ++i)
		{
			Vec3f jmap = jmap_map[i];
			Vec3f jexp = jexp_map[i];
			const Vec3<PidType> pids = pids_map[i];
			const float res = r_map[i];

			if (res == r_texture_->nodata() || jmap == jmap_texture_->nodata() || jexp == jexp_texture_->nodata() || pids == pids_texture_->nodata())
				continue;

			const Vec5i pids_(pids(0), pids(1), pids(2), num_vertices_ + frame_id_ * 2, num_vertices_ + frame_id_ * 2 + 1);
			const float depth0 = depths[pids_(0)];
			const float depth1 = depths[pids_(1)];
			const float depth2 = depths[pids_(2)];
			const float d_depth0_d_param = d_depth_d_param(depth0);
			const float d_depth1_d_param = d_depth_d_param(depth1);
			const float d_depth2_d_param = d_depth_d_param(depth2);
			Vec5f J;
			J(0) = jmap(0) * d_depth0_d_param;
			J(1) = jmap(1) * d_depth1_d_param;
			J(2) = jmap(2) * d_depth2_d_param;
			J(3) = jexp(0);
			J(4) = jexp(1);

			const float w = huber_weight(res, mesh_vo::huber_thresh_pix);

			hg.add(J, res, w, pids_);
		}
	}

private:
	const Texture<Vec3f> *jmap_texture_;
	const Texture<Vec3f> *jexp_texture_;
	const Texture<Vec3<PidType>> *pids_texture_;
	const Texture<float> *r_texture_;
	const Mesh *mesh_;
	int lvl_;
	int num_frames_;
	int num_vertices_;
	int frame_id_;
	// DenseLinearProblemx hg_;
};

class HGPoseMapReducerCPU : public BaseReducerCPU<HGPoseMapReducerCPU, DenseLinearProblemx>
{
public:
	using Base = BaseReducerCPU<HGPoseMapReducerCPU, DenseLinearProblemx>;
	explicit HGPoseMapReducerCPU(unsigned threads = 1 /*std::max(1u, std::thread::hardware_concurrency())*/)
		: Base(threads)
	{
	}

	void reduce(int lvl, int frame_id, int num_frames, int num_vertices, const Texture<Vec3f> &jtra_texture, const Texture<Vec3f> &jrot_texture, const Texture<Vec3f> &jmap_texture, const Texture<Vec3<PidType>> &pids_texture, const Texture<float> &r_texture, const Mesh &mesh, DenseLinearProblemx &total)
	{
		jtra_texture_ = &jtra_texture;
		jrot_texture_ = &jrot_texture;
		jmap_texture_ = &jmap_texture;
		pids_texture_ = &pids_texture;
		r_texture_ = &r_texture;
		mesh_ = &mesh;

		lvl_ = lvl;
		num_frames_ = num_frames;
		num_vertices_ = num_vertices;
		frame_id_ = frame_id;

		// DenseLinearProblem hg(total);

		reduce_(total);
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	void reducepartial(int begin, int end, DenseLinearProblemx &hg)
	{
		auto r_map = r_texture_->MapRead(lvl_);
		auto jtra_map = jtra_texture_->MapRead(lvl_);
		auto jrot_map = jrot_texture_->MapRead(lvl_);
		auto jmap_map = jmap_texture_->MapRead(lvl_);
		auto pids_map = pids_texture_->MapRead(lvl_);
		auto depths = get_depths(*mesh_);
		// Vec6i ids(0, 1, 2, 3, 4, 5);

		for (int i = begin; i < end; ++i)
		{
			Vec3f jtra = jtra_map[i];
			Vec3f jrot = jrot_map[i];
			Vec3f jmap = jmap_map[i];
			const Vec3<PidType> pids = pids_map[i];
			const float res = r_map[i];

			if (res == r_texture_->nodata() || jtra == jtra_texture_->nodata() || jrot == jrot_texture_->nodata() || jmap == jmap_texture_->nodata() || pids == pids_texture_->nodata())
				continue;

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

			const float depth0 = depths[pids_(0)];
			const float depth1 = depths[pids_(1)];
			const float depth2 = depths[pids_(2)];
			const float d_depth0_d_param = d_depth_d_param(depth0);
			const float d_depth1_d_param = d_depth_d_param(depth1);
			const float d_depth2_d_param = d_depth_d_param(depth2);
			Vec<float, 9> J;
			J(0) = jmap(0) * d_depth0_d_param;
			J(1) = jmap(1) * d_depth1_d_param;
			J(2) = jmap(2) * d_depth2_d_param;
			J(3) = jtra(0);
			J(4) = jtra(1);
			J(5) = jtra(2);
			J(6) = jrot(0);
			J(7) = jrot(1);
			J(8) = jrot(2);

			const float w = huber_weight(res, mesh_vo::huber_thresh_pix);

			hg.add(J, res, w, pids_);
		}
	}

private:
	const Texture<Vec3f> *jtra_texture_;
	const Texture<Vec3f> *jrot_texture_;
	const Texture<Vec3f> *jmap_texture_;
	const Texture<Vec3<PidType>> *pids_texture_;
	const Texture<float> *r_texture_;
	const Mesh *mesh_;
	int lvl_;
	int num_frames_;
	int num_vertices_;
	int frame_id_;
	// DenseLinearProblemx hg_;
};

class HGPoseExpMapReducerCPU : public BaseReducerCPU<HGPoseExpMapReducerCPU, DenseLinearProblemx>
{
public:
	using Base = BaseReducerCPU<HGPoseExpMapReducerCPU, DenseLinearProblemx>;
	explicit HGPoseExpMapReducerCPU(unsigned threads = 1 /*std::max(1u, std::thread::hardware_concurrency())*/)
		: Base(threads)
	{
	}

	void reduce(int lvl, int frame_id, int num_frames, int num_vertices, const Texture<Vec3f> &jtra_texture, const Texture<Vec3f> &jrot_texture, const Texture<Vec3f> &jexp_texture, const Texture<Vec3f> &jmap_texture, const Texture<Vec3<PidType>> &pids_texture, const Texture<float> &r_texture, const Mesh &mesh, DenseLinearProblemx &total)
	{
		jtra_texture_ = &jtra_texture;
		jrot_texture_ = &jrot_texture;
		jexp_texture_ = &jexp_texture;
		jmap_texture_ = &jmap_texture;
		pids_texture_ = &pids_texture;
		r_texture_ = &r_texture;
		mesh_ = &mesh;

		lvl_ = lvl;
		num_frames_ = num_frames;
		num_vertices_ = num_vertices;
		frame_id_ = frame_id;

		// DenseLinearProblem hg(total);

		reduce_(total);
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	void reducepartial(int begin, int end, DenseLinearProblemx &hg)
	{
		auto r_map = r_texture_->MapRead(lvl_);
		auto jtra_map = jtra_texture_->MapRead(lvl_);
		auto jrot_map = jrot_texture_->MapRead(lvl_);
		auto jexp_map = jexp_texture_->MapRead(lvl_);
		auto jmap_map = jmap_texture_->MapRead(lvl_);
		auto pids_map = pids_texture_->MapRead(lvl_);
		auto depths = get_depths(*mesh_);
		// Vec6i ids(0, 1, 2, 3, 4, 5);

		for (int i = begin; i < end; ++i)
		{
			Vec3f jtra = jtra_map[i];
			Vec3f jrot = jrot_map[i];
			Vec3f jexp = jexp_map[i];
			Vec3f jmap = jmap_map[i];
			const Vec3<PidType> pids = pids_map[i];
			const float res = r_map[i];

			if (res == r_texture_->nodata() || jtra == jtra_texture_->nodata() || jrot == jrot_texture_->nodata() || jexp == jexp_texture_->nodata() || jmap == jmap_texture_->nodata() || pids == pids_texture_->nodata())
				continue;

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
			J(0) = jmap(0) * d_depth0_d_param;
			J(1) = jmap(1) * d_depth1_d_param;
			J(2) = jmap(2) * d_depth2_d_param;
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
	}

private:
	const Texture<Vec3f> *jtra_texture_;
	const Texture<Vec3f> *jrot_texture_;
	const Texture<Vec3f> *jexp_texture_;
	const Texture<Vec3f> *jmap_texture_;
	const Texture<Vec3<PidType>> *pids_texture_;
	const Texture<float> *r_texture_;
	const Mesh *mesh_;
	int lvl_;
	int num_frames_;
	int num_vertices_;
	int frame_id_;
	// DenseLinearProblemx hg_;
};

class HGPoseVelMapReducerCPU : public BaseReducerCPU<HGPoseVelMapReducerCPU, DenseLinearProblemx>
{
public:
	using Base = BaseReducerCPU<HGPoseVelMapReducerCPU, DenseLinearProblemx>;
	explicit HGPoseVelMapReducerCPU(unsigned threads = 1 /*std::max(1u, std::thread::hardware_concurrency())*/)
		: Base(threads)
	{
	}

	void reduce(int lvl, int frame_id, int num_frames, int num_vertices, const Texture<Vec3f> &jtra_texture, const Texture<Vec3f> &jrot_texture, const Texture<Vec3f> &jtravel_texture, const Texture<Vec3f> &jrotvel_texture, const Texture<Vec3f> &jmap_texture, const Texture<Vec3<PidType>> &pids_texture, const Texture<float> &r_texture, const Mesh &mesh, DenseLinearProblemx &total)
	{
		jtra_texture_ = &jtra_texture;
		jrot_texture_ = &jrot_texture;
		jtravel_texture_ = &jtravel_texture;
		jrotvel_texture_ = &jrotvel_texture;
		jmap_texture_ = &jmap_texture;
		pids_texture_ = &pids_texture;
		r_texture_ = &r_texture;
		mesh_ = &mesh;

		lvl_ = lvl;
		num_frames_ = num_frames;
		num_vertices_ = num_vertices;
		frame_id_ = frame_id;

		// DenseLinearProblem hg(total);

		reduce_(total);
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	void reducepartial(int begin, int end, DenseLinearProblemx &hg)
	{
		auto r_map = r_texture_->MapRead(lvl_);
		auto jtra_map = jtra_texture_->MapRead(lvl_);
		auto jrot_map = jrot_texture_->MapRead(lvl_);
		auto jtravel_map = jtravel_texture_->MapRead(lvl_);
		auto jrotvel_map = jrotvel_texture_->MapRead(lvl_);
		auto jmap_map = jmap_texture_->MapRead(lvl_);
		auto pids_map = pids_texture_->MapRead(lvl_);
		auto depths = get_depths(*mesh_);
		// Vec6i ids(0, 1, 2, 3, 4, 5);

		for (int i = begin; i < end; ++i)
		{
			Vec3f jtra = jtra_map[i];
			Vec3f jrot = jrot_map[i];
			Vec3f jtravel = jtravel_map[i];
			Vec3f jrotvel = jrotvel_map[i];
			Vec3f jmap = jmap_map[i];
			const Vec3<PidType> pids = pids_map[i];
			const float res = r_map[i];

			if (res == r_texture_->nodata() ||
				jtra == jtra_texture_->nodata() ||
				jrot == jrot_texture_->nodata() ||
				jtravel == jtravel_texture_->nodata() ||
				jrotvel == jrotvel_texture_->nodata() ||
				jmap == jmap_texture_->nodata() ||
				pids == pids_texture_->nodata())
				continue;

			Vec<int, 15> pids_;
			pids_(0) = pids(0);
			pids_(1) = pids(1);
			pids_(2) = pids(2);
			pids_(3) = num_vertices_ + frame_id_ * 12;
			pids_(4) = num_vertices_ + frame_id_ * 12 + 1;
			pids_(5) = num_vertices_ + frame_id_ * 12 + 2;
			pids_(6) = num_vertices_ + frame_id_ * 12 + 3;
			pids_(7) = num_vertices_ + frame_id_ * 12 + 4;
			pids_(8) = num_vertices_ + frame_id_ * 12 + 5;
			pids_(9) = num_vertices_ + frame_id_ * 12 + 6;
			pids_(10) = num_vertices_ + frame_id_ * 12 + 7;
			pids_(11) = num_vertices_ + frame_id_ * 12 + 8;
			pids_(12) = num_vertices_ + frame_id_ * 12 + 9;
			pids_(13) = num_vertices_ + frame_id_ * 12 + 10;
			pids_(14) = num_vertices_ + frame_id_ * 12 + 11;

			const float depth0 = depths[pids_(0)];
			const float depth1 = depths[pids_(1)];
			const float depth2 = depths[pids_(2)];
			const float d_depth0_d_param = d_depth_d_param(depth0);
			const float d_depth1_d_param = d_depth_d_param(depth1);
			const float d_depth2_d_param = d_depth_d_param(depth2);
			Vec<float, 15> J;
			J(0) = jmap(0) * d_depth0_d_param;
			J(1) = jmap(1) * d_depth1_d_param;
			J(2) = jmap(2) * d_depth2_d_param;
			J(3) = jtra(0);
			J(4) = jtra(1);
			J(5) = jtra(2);
			J(6) = jrot(0);
			J(7) = jrot(1);
			J(8) = jrot(2);
			J(9) = jtravel(0);
			J(10) = jtravel(1);
			J(11) = jtravel(2);
			J(12) = jrotvel(0);
			J(13) = jrotvel(1);
			J(14) = jrotvel(2);

			const float w = huber_weight(res, mesh_vo::huber_thresh_pix);

			hg.add(J, res, w, pids_);
		}
	}

private:
	const Texture<Vec3f> *jtra_texture_;
	const Texture<Vec3f> *jrot_texture_;
	const Texture<Vec3f> *jtravel_texture_;
	const Texture<Vec3f> *jrotvel_texture_;
	const Texture<Vec3f> *jmap_texture_;
	const Texture<Vec3<PidType>> *pids_texture_;
	const Texture<float> *r_texture_;
	const Mesh *mesh_;
	int lvl_;
	int num_frames_;
	int num_vertices_;
	int frame_id_;
	// DenseLinearProblemx hg_;
};

class HGPoseVelExpMapReducerCPU : public BaseReducerCPU<HGPoseVelExpMapReducerCPU, DenseLinearProblemx>
{
public:
	using Base = BaseReducerCPU<HGPoseVelExpMapReducerCPU, DenseLinearProblemx>;
	explicit HGPoseVelExpMapReducerCPU(unsigned threads = 1 /*std::max(1u, std::thread::hardware_concurrency())*/)
		: Base(threads)
	{
	}

	void reduce(int lvl, int frame_id, int num_frames, int num_vertices, const Texture<Vec3f> &jtra_texture, const Texture<Vec3f> &jrot_texture, const Texture<Vec3f> &jtravel_texture, const Texture<Vec3f> &jrotvel_texture, const Texture<Vec3f> &jexp_texture, const Texture<Vec3f> &jmap_texture, const Texture<Vec3<PidType>> &pids_texture, const Texture<float> &r_texture, const Mesh &mesh, DenseLinearProblemx &total)
	{
		jtra_texture_ = &jtra_texture;
		jrot_texture_ = &jrot_texture;
		jtravel_texture_ = &jtravel_texture;
		jrotvel_texture_ = &jrotvel_texture;
		jexp_texture_ = &jexp_texture;
		jmap_texture_ = &jmap_texture;
		pids_texture_ = &pids_texture;
		r_texture_ = &r_texture;
		mesh_ = &mesh;

		lvl_ = lvl;
		num_frames_ = num_frames;
		num_vertices_ = num_vertices;
		frame_id_ = frame_id;

		// DenseLinearProblem hg(total);

		reduce_(total);
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	void reducepartial(int begin, int end, DenseLinearProblemx &hg)
	{
		auto r_map = r_texture_->MapRead(lvl_);
		auto jtra_map = jtra_texture_->MapRead(lvl_);
		auto jrot_map = jrot_texture_->MapRead(lvl_);
		auto jtravel_map = jtravel_texture_->MapRead(lvl_);
		auto jrotvel_map = jrotvel_texture_->MapRead(lvl_);
		auto jexp_map = jexp_texture_->MapRead(lvl_);
		auto jmap_map = jmap_texture_->MapRead(lvl_);
		auto pids_map = pids_texture_->MapRead(lvl_);
		auto depths = get_depths(*mesh_);
		// Vec6i ids(0, 1, 2, 3, 4, 5);

		for (int i = begin; i < end; ++i)
		{
			Vec3f jtra = jtra_map[i];
			Vec3f jrot = jrot_map[i];
			Vec3f jtravel = jtravel_map[i];
			Vec3f jrotvel = jrotvel_map[i];
			Vec3f jexp = jexp_map[i];
			Vec3f jmap = jmap_map[i];
			const Vec3<PidType> pids = pids_map[i];
			const float res = r_map[i];

			if (res == r_texture_->nodata() ||
				jtra == jtra_texture_->nodata() ||
				jrot == jrot_texture_->nodata() ||
				jtravel == jtravel_texture_->nodata() ||
				jrotvel == jrotvel_texture_->nodata() ||
				jexp == jexp_texture_->nodata() ||
				jmap == jmap_texture_->nodata() ||
				pids == pids_texture_->nodata())
				continue;

			Vec<int, 17> pids_;
			pids_(0) = pids(0);
			pids_(1) = pids(1);
			pids_(2) = pids(2);
			pids_(3) = num_vertices_ + frame_id_ * 14;
			pids_(4) = num_vertices_ + frame_id_ * 14 + 1;
			pids_(5) = num_vertices_ + frame_id_ * 14 + 2;
			pids_(6) = num_vertices_ + frame_id_ * 14 + 3;
			pids_(7) = num_vertices_ + frame_id_ * 14 + 4;
			pids_(8) = num_vertices_ + frame_id_ * 14 + 5;
			pids_(9) = num_vertices_ + frame_id_ * 14 + 6;
			pids_(10) = num_vertices_ + frame_id_ * 14 + 7;
			pids_(11) = num_vertices_ + frame_id_ * 14 + 8;
			pids_(12) = num_vertices_ + frame_id_ * 14 + 9;
			pids_(13) = num_vertices_ + frame_id_ * 14 + 10;
			pids_(14) = num_vertices_ + frame_id_ * 14 + 11;
			pids_(15) = num_vertices_ + frame_id_ * 14 + 12;
			pids_(16) = num_vertices_ + frame_id_ * 14 + 13;

			const float depth0 = depths[pids_(0)];
			const float depth1 = depths[pids_(1)];
			const float depth2 = depths[pids_(2)];
			const float d_depth0_d_param = d_depth_d_param(depth0);
			const float d_depth1_d_param = d_depth_d_param(depth1);
			const float d_depth2_d_param = d_depth_d_param(depth2);
			Vec<float, 17> J;
			J(0) = jmap(0) * d_depth0_d_param;
			J(1) = jmap(1) * d_depth1_d_param;
			J(2) = jmap(2) * d_depth2_d_param;
			J(3) = jtra(0);
			J(4) = jtra(1);
			J(5) = jtra(2);
			J(6) = jrot(0);
			J(7) = jrot(1);
			J(8) = jrot(2);
			J(9) = jtravel(0);
			J(10) = jtravel(1);
			J(11) = jtravel(2);
			J(12) = jrotvel(0);
			J(13) = jrotvel(1);
			J(14) = jrotvel(2);
			J(15) = jexp(0);
			J(16) = jexp(1);

			const float w = huber_weight(res, mesh_vo::huber_thresh_pix);

			hg.add(J, res, w, pids_);
		}
	}

private:
	const Texture<Vec3f> *jtra_texture_;
	const Texture<Vec3f> *jrot_texture_;
	const Texture<Vec3f> *jtravel_texture_;
	const Texture<Vec3f> *jrotvel_texture_;
	const Texture<Vec3f> *jexp_texture_;
	const Texture<Vec3f> *jmap_texture_;
	const Texture<Vec3<PidType>> *pids_texture_;
	const Texture<float> *r_texture_;
	const Mesh *mesh_;
	int lvl_;
	int num_frames_;
	int num_vertices_;
	int frame_id_;
	// DenseLinearProblemx hg_;
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
