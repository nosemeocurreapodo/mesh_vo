#pragma once

#include <thread>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>

#include "params.h"
#include "common/types.h"
#include "common/error.h"
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
	OutType reduce_()
	{
		const int N = derived_().size();
		return derived_().reducepartial(0, N);

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

		if (T == 0) return OutType{};
		OutType total = partial[0];
		for (unsigned t = 1; t < T; ++t)
			total += partial[t];
		return total;
		*/
	}

	static inline float huber_weight_(float r, float thresh) noexcept
	{
		const float a = std::fabs(r);
		if (a <= thresh || a == 0.0f)
			return 1.0f;
		return thresh / a;
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

	Error reduce(int lvl, const Texture<ImageType> &r_texture)
	{
		r_texture_ = &r_texture;
		lvl_ = lvl;

		return reduce_();
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	Error reducepartial(int begin, int end)
	{
		auto rmap = r_texture_->MapRead(lvl_);

		Error err;
		for (int i = begin; i < end; ++i)
		{
			const float r = rmap[i];
			if (r == r_texture_->nodata())
				err += 1.0f;
			else
				err += 0.0f;
		}
		return err;
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

	Error reduce(int lvl, const Texture<float> &r_texture)
	{
		r_texture_ = &r_texture;
		lvl_ = lvl;

		return reduce_();
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	Error reducepartial(int begin, int end)
	{
		auto rmap = r_texture_->MapRead(lvl_);

		Error err;
		for (int i = begin; i < end; ++i)
		{
			const float r = rmap[i];
			if (r == r_texture_->nodata())
				continue;
			const float w = huber_weight_(r, mesh_vo::huber_thresh_pix);
			err += w * r * r;
		}
		return err;
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

	DenseLinearProblem<6> reduce(int lvl, const Texture<Vec3f> &jtra_texture, const Texture<Vec3f> &jrot_texture, const Texture<float> &r_texture)
	{
		jtra_texture_ = &jtra_texture;
		jrot_texture_ = &jrot_texture;
		r_texture_ = &r_texture;
		lvl_ = lvl;

		return reduce_();
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	DenseLinearProblem<6> reducepartial(int begin, int end)
	{
		DenseLinearProblem<6> hg;
		auto r_map = r_texture_->MapRead(lvl_);
		auto jtra_map = jtra_texture_->MapRead(lvl_);
		auto jrot_map = jrot_texture_->MapRead(lvl_);
		Vec6i ids(0, 1, 2, 3, 4, 5);

		for (int i = begin; i < end; ++i)
		{
			const float res = r_map[i];
			const Vec3f jtra = jtra_map[i];
			const Vec3f jrot = jrot_map[i];
			if (res == r_texture_->nodata() || jtra == jtra_texture_->nodata() || jrot == jrot_texture_->nodata())
				continue;
			Vec6f J(jtra(0), jtra(1), jtra(2), jrot(0), jrot(1), jrot(2));
			const float w = huber_weight_(res, mesh_vo::huber_thresh_pix);

			// hg.add(J, res, w, ids);
			hg.add(J, res, w);
		}
		return hg;
	}

private:
	const Texture<Vec3f> *jtra_texture_;
	const Texture<Vec3f> *jrot_texture_;
	const Texture<float> *r_texture_;
	int lvl_;
	//DenseLinearProblem<6> hg_;
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

	DenseLinearProblemx reduce(int lvl, int total, const Texture<Vec3f> &jmap_texture, const Texture<Vec3<PidType>> &pids_texture, const Texture<float> &r_texture)
	{
		jmap_texture_ = &jmap_texture;
		pids_texture_ = &pids_texture;
		r_texture_ = &r_texture;

		lvl_ = lvl;
		total_ = total;

		// DenseLinearProblem hg(total);

		return reduce_();
	}

	int size()
	{
		return r_texture_->width(lvl_) * r_texture_->height(lvl_);
	}

	DenseLinearProblemx reducepartial(int begin, int end)
	{
		DenseLinearProblemx hg(total_);
		auto r_map = r_texture_->MapRead(lvl_);
		auto jmap_map = jmap_texture_->MapRead(lvl_);
		auto pids_map = pids_texture_->MapRead(lvl_);
		// Vec6i ids(0, 1, 2, 3, 4, 5);

		for (int i = begin; i < end; ++i)
		{
			const float res = r_map[i];
			const Vec3f jmap = jmap_map[i];
			const Vec3i pids = pids_map[i];
			if (res == r_texture_->nodata() || jmap == jmap_texture_->nodata() || pids == pids_texture_->nodata())
				continue;
			const float w = huber_weight_(res, mesh_vo::huber_thresh_pix);
			const Vec3i pids_(pids(0), pids(1), pids(2));

			hg.add(jmap, res, w, pids_);
		}
		return hg;
	}

private:
	const Texture<Vec3f> *jmap_texture_;
	const Texture<Vec3<PidType>> *pids_texture_;
	const Texture<float> *r_texture_;
	int lvl_;
	int total_;
	//DenseLinearProblemx hg_;
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
