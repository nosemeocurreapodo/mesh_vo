#pragma once

#include <thread>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>

#include "params.h"
#include "core/types.h"
#include "common/types.h"
#include "common/DenseLinearProblem.h"
#include "common/error.h"

// Generic, thread-safe reducer base. Splits [0, N) into contiguous chunks and aggregates results.
template <class Derived, typename OutType>
class BaseReducerCPU
{
public:
	BaseReducerCPU() noexcept : threads_(1 /*std::max(1u, std::thread::hardware_concurrency())*/) {}
	explicit BaseReducerCPU(unsigned threads) noexcept : threads_(threads == 0 ? 1u : threads) {}
	virtual ~BaseReducerCPU() = default;

	template <typename... Texture>
	OutType reduce(int lvl, const Texture &...texs)
	{
		// const int n1 = reduce1.size(lvl);
		// const int n2 = reduce2.size(lvl);
		// const int n3 = reduce3.size(lvl);
		// const int N = std::min({n1, n2, n3});
		// assert(N >= 0);
		// if (N <= 0)
		//	return OutType{};

		const int N = min_size(lvl, texs...);

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
							  { partial[t] = derived().reducepartial(begin, end, lvl, texs...); });
		}
		{
			const int begin = 0;
			const int end = std::min(N, chunk);
			partial[0] = derived().reducepartial(begin, end, lvl, texs...);
		}
		for (auto &th : pool)
			th.join();

		OutType total{};
		for (unsigned t = 0; t < T; ++t)
			total += partial[t];
		return total;
	}

protected:
	// template <typename... Ts>
	// virtual OutType reducepartial(int begin, int end,
	//							  const TextureCPU<Ts> &...texs,
	//							  int lvl) = 0;

	static inline float huber_weight(float r, float thresh) noexcept
	{
		const float a = std::fabs(r);
		if (a <= thresh || a == 0.0f)
			return 1.0f;
		return thresh / a;
	}

	Derived &derived() { return *static_cast<Derived *>(this); }
	const Derived &derived() const { return *static_cast<const Derived *>(this); }

private:
	template <class V>
	static inline int size_of(int lvl, const V &v) { return v.size(lvl); }
	template <class V0, class... Vn>
	static inline int min_size(int lvl, const V0 &v0, const Vn &...vn)
	{
		int m = size_of(lvl, v0);
		((m = std::min(m, size_of(lvl, vn))), ...);
		return m;
	}

	unsigned threads_;
};

// Photometric L2 with Huber loss. Third input unused.
class ErrorReducerCPU : public BaseReducerCPU<ErrorReducerCPU, Error>
{
public:
	using Base = BaseReducerCPU<ErrorReducerCPU, Error>;
	explicit ErrorReducerCPU(unsigned threads = 1 /*std::max(1u, std::thread::hardware_concurrency())*/) : Base(threads) {}

	Error reducepartial(int begin, int end, int lvl,
						const TextureCPU<float> &image,
						const TextureCPU<float> &kimage_projected)
	{
		auto img = image.MapRead(lvl);
		auto kin = kimage_projected.MapRead(lvl);

		Error err;
		for (int i = begin; i < end; ++i)
		{
			const float v = img[i];
			const float k = kin[i];
			if (v == image.nodata() || k == kimage_projected.nodata())
				continue;
			const float r = v - k;
			const float w = huber_weight(r, mesh_vo::huber_thresh_pix);
			err += w * r * r;
		}
		return err;
	}
};

// Pose-only Jacobian -> DenseLinearProblem reducer
class HGPoseReducerCPU : public BaseReducerCPU<HGPoseReducerCPU, DenseLinearProblem>
{
public:
	using Base = BaseReducerCPU<HGPoseReducerCPU, DenseLinearProblem>;
	explicit HGPoseReducerCPU(unsigned threads = 1 /*std::max(1u, std::thread::hardware_concurrency())*/) : Base(threads) {}

	DenseLinearProblem reducepartial(int begin, int end, int lvl,
									 const TextureCPU<float> &image,
									 const TextureCPU<float> &kimage_projected,
									 const TextureCPU<Vec6> &jpose)
	{
		DenseLinearProblem hg(6);
		auto ibuf = image.MapRead(lvl);
		auto kbuf = kimage_projected.MapRead(lvl);
		auto jbuf = jpose.MapRead(lvl);
		Vec6i ids(0, 1, 2, 3, 4, 5);

		for (int i = begin; i < end; ++i)
		{
			const float img = ibuf[i];
			const float kimg = kbuf[i];
			const Vec6 J = jbuf[i];
			if (img == image.nodata() || kimg == kimage_projected.nodata() || J == jpose.nodata())
				continue;
			float res = img - kimg;
			const float w = huber_weight(res, mesh_vo::huber_thresh_pix);

			hg.add(J, res, w, ids);
		}
		return hg;
	}
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
