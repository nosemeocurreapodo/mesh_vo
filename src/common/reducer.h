#pragma once

#include <thread>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "params.h"
#include "common/types.h"
#include "common/error.h"
#include "common/depthParam.h"
#include "common/DenseLinearProblem.h"

// -------------------- Options --------------------

struct ReduceOptions
{
	unsigned threads = std::max(1u, std::thread::hardware_concurrency());
};

// -------------------- RAII joiner for C++17 threads --------------------

struct JoinThreads
{
	std::vector<std::thread> &threads;
	~JoinThreads()
	{
		for (auto &t : threads)
			if (t.joinable())
				t.join();
	}
};

// -------------------- nodata helpers (NaN-safe for floating) --------------------

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, bool>::type
is_nodata_scalar(T v, T nodata) noexcept
{
	return std::isnan(nodata) ? std::isnan(v) : (v == nodata);
}

template <typename T>
inline typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
is_nodata_scalar(const T &v, const T &nodata) noexcept
{
	return v == nodata;
}

// Assumes Vec3<T> supports operator()(int). You use j(0), j(1), j(2) already.
template <typename T>
inline bool is_nodata_vec3(const Vec3<T> &v, const Vec3<T> &nodata) noexcept
{
	return is_nodata_scalar(v(0), nodata(0)) &&
		   is_nodata_scalar(v(1), nodata(1)) &&
		   is_nodata_scalar(v(2), nodata(2));
}

// -------------------- parallel reduce over [0, N) with identity factory --------------------
// Fn signature: void(begin, end, Out& out) ; out is assumed to start as identity
template <class Out, class Fn, class MakeIdentity>
[[nodiscard]] Out parallel_reduce_1d(std::size_t N,
									 Fn &&fn,
									 MakeIdentity &&make_identity,
									 ReduceOptions opt = {})
{
	Out out;
	fn(0, N, out);
	return out;
	/*
	if (N == 0)
		return make_identity();

	unsigned T = std::min<unsigned>(opt.threads, static_cast<unsigned>(N));
	T = std::max(1u, T);

	const std::size_t chunk = (N + T - 1) / T;

	std::vector<Out> partial;
	partial.reserve(T);
	for (unsigned t = 0; t < T; ++t)
		partial.emplace_back(make_identity());

	std::vector<std::thread> pool;
	pool.reserve(T > 1 ? T - 1 : 0);
	JoinThreads joiner{pool};

	for (unsigned t = 1; t < T; ++t)
	{
		const std::size_t begin = static_cast<std::size_t>(t) * chunk;
		const std::size_t end = std::min(N, begin + chunk);
		pool.emplace_back([&, t, begin, end]
						  { fn(begin, end, partial[t]); });
	}

	// Do chunk 0 on caller thread
	{
		const std::size_t begin = 0;
		const std::size_t end = std::min(N, chunk);
		fn(begin, end, partial[0]);
	}

	Out total = make_identity();
	for (auto &p : partial)
		total += p;
	return total;
	*/
}

template <typename T>
class NodataReducerCPU
{
public:
	explicit NodataReducerCPU(ReduceOptions opt = {}) : opt_(opt) {}

	void reduce(int lvl, const Texture<ImageType> &r_texture, Error<T> &total) const
	{
		total += compute(lvl, r_texture);
	}

	[[nodiscard]] Error<T> compute(int lvl, const Texture<ImageType> &r_texture) const
	{
		auto view = r_texture.MapRead(lvl);

		const std::size_t w = static_cast<std::size_t>(view.width());
		const std::size_t h = static_cast<std::size_t>(view.height());
		const std::size_t N = w * h;

		const auto nod = view.nodata();

		return parallel_reduce_1d<Error<T>>(
			N,
			[&](std::size_t begin, std::size_t end, Error<T> &out)
			{
				for (std::size_t i = begin; i < end; ++i)
				{
					const ImageType r = view[i];
					if (is_nodata_scalar(r, nod))
						out += 1;
				}
			},
			[]
			{ return Error<T>{}; },
			opt_);
	}

private:
	ReduceOptions opt_;
};

template <typename T>
class ResidualReducerCPU
{
public:
	explicit ResidualReducerCPU(ReduceOptions opt = {}) : opt_(opt) {}

	void reduce(int lvl,
				const Texture<ImageType> &res_texture,
				Error<T> &total) const
	{
		total += compute(lvl, res_texture);
	}

	[[nodiscard]] Error<T> compute(int lvl,
								   const Texture<ImageType> &res_texture) const
	{
		auto res = res_texture.MapRead(lvl);

		const std::size_t w = static_cast<std::size_t>(res.width());
		const std::size_t h = static_cast<std::size_t>(res.height());
		const std::size_t N = w * h;

		const auto nod = res.nodata();

		return parallel_reduce_1d<Error<T>>(
			N,
			[&](std::size_t begin, std::size_t end, Error<T> &out)
			{
				for (std::size_t i = begin; i < end; ++i)
				{
					const ImageType r = res[i];
					if (is_nodata_scalar(r, nod))
						continue;

					const T w = huber_weight(r, mesh_vo::huber_thresh_pix);
					out += w * r * r;
				}
			},
			[]
			{ return Error<T>{}; },
			opt_);
	}

private:
	ReduceOptions opt_;
};

template <typename T>
class HGPoseReducerCPU
{
public:
	explicit HGPoseReducerCPU(ReduceOptions opt = {}) : opt_(opt) {}

	void reduce(int lvl,
				const Texture<Vec3f> &jtra_texture,
				const Texture<Vec3f> &jrot_texture,
				const Texture<float> &res_texture,
				DenseLinearProblem<T, 6> &total) const
	{
		total += compute(lvl, jtra_texture, jrot_texture, res_texture);
	}

	[[nodiscard]] DenseLinearProblem<T, 6> compute(int lvl,
												   const Texture<Vec3f> &jtra_texture,
												   const Texture<Vec3f> &jrot_texture,
												   const Texture<float> &res_texture) const
	{
		auto jtra = jtra_texture.MapRead(lvl);
		auto jrot = jrot_texture.MapRead(lvl);
		auto res = res_texture.MapRead(lvl);

		const std::size_t w = static_cast<std::size_t>(res.width());
		const std::size_t h = static_cast<std::size_t>(res.height());
		const std::size_t N = w * h;

		const auto nod_res = res.nodata();
		const auto nod_jtra = jtra.nodata();
		const auto nod_jrot = jrot.nodata();

		return parallel_reduce_1d<DenseLinearProblem<T, 6>>(
			N,
			[&](std::size_t begin, std::size_t end, DenseLinearProblem<T, 6> &hg)
			{
				for (std::size_t i = begin; i < end; ++i)
				{
					const ImageType res_i = res[i];
					const Vec3f jt = jtra[i];
					const Vec3f jr = jrot[i];

					if (is_nodata_scalar(res_i, nod_res) ||
						is_nodata_vec3(jt, nod_jtra) ||
						is_nodata_vec3(jr, nod_jrot))
						continue;

					Vec<T, 6> J;
					J(0) = jt(0);
					J(1) = jt(1);
					J(2) = jt(2);
					J(3) = jr(0);
					J(4) = jr(1);
					J(5) = jr(2);

					const T w = huber_weight(res_i, mesh_vo::huber_thresh_pix);
					hg.add(J, res_i, w);
				}
			},
			[]
			{ return DenseLinearProblem<T, 6>{}; },
			opt_);
	}

private:
	ReduceOptions opt_;
};

template <typename T>
class HGPoseExpReducerCPU
{
public:
	explicit HGPoseExpReducerCPU(ReduceOptions opt = {}) : opt_(opt) {}

	void reduce(int lvl,
				const Texture<Vec3f> &jtra_texture,
				const Texture<Vec3f> &jrot_texture,
				const Texture<Vec3f> &jexp_texture,
				const Texture<ImageType> &res_texture,
				DenseLinearProblem<T, 8> &total) const
	{
		total += compute(lvl, jtra_texture, jrot_texture, jexp_texture, res_texture);
	}

	[[nodiscard]] DenseLinearProblem<T, 8> compute(int lvl,
												   const Texture<Vec3f> &jtra_texture,
												   const Texture<Vec3f> &jrot_texture,
												   const Texture<Vec3f> &jexp_texture,
												   const Texture<ImageType> &res_texture) const
	{
		auto jtra = jtra_texture.MapRead(lvl);
		auto jrot = jrot_texture.MapRead(lvl);
		auto jexp = jexp_texture.MapRead(lvl);
		auto res = res_texture.MapRead(lvl);

		const std::size_t w = static_cast<std::size_t>(res.width());
		const std::size_t h = static_cast<std::size_t>(res.height());
		const std::size_t N = w * h;

		const auto nod_res = res.nodata();
		const auto nod_jtra = jtra.nodata();
		const auto nod_jrot = jrot.nodata();
		const auto nod_jexp = jexp.nodata();

		return parallel_reduce_1d<DenseLinearProblem<T, 8>>(
			N,
			[&](std::size_t begin, std::size_t end, DenseLinearProblem<T, 8> &hg)
			{
				for (std::size_t i = begin; i < end; ++i)
				{
					const ImageType res_i = res[i];
					const Vec3f jt = jtra[i];
					const Vec3f jr = jrot[i];
					const Vec3f je = jexp[i];

					if (is_nodata_scalar(res_i, nod_res) ||
						is_nodata_vec3(jt, nod_jtra) ||
						is_nodata_vec3(jr, nod_jrot) ||
						is_nodata_vec3(je, nod_jexp))
						continue;

					Vec<T, 8> J;
					J(0) = jt(0);
					J(1) = jt(1);
					J(2) = jt(2);
					J(3) = jr(0);
					J(4) = jr(1);
					J(5) = jr(2);
					J(6) = je(0);
					J(7) = je(1);

					const T w = huber_weight(res_i, mesh_vo::huber_thresh_pix);
					hg.add(J, res_i, w);
				}
			},
			[]
			{ return DenseLinearProblem<T, 8>{}; },
			opt_);
	}

private:
	ReduceOptions opt_;
};

template <typename T>
class HGDepthReducerCPU
{
public:
	// For DenseLinearProblemx, default to 1 thread unless you KNOW n is small.
	explicit HGDepthReducerCPU() {}

	void reduce(int lvl,
				int frame_id,
				int num_frames,
				int num_vertices,
				const Texture<Vec3f> &jdepth_texture,
				const Texture<Vec3<PidType>> &pids_texture,
				const Texture<float> &res_texture,
				const Mesh &mesh,
				DenseLinearProblemx<T> &total) const
	{
		// const int dof = num_vertices; // Depth params only

		// If caller wants accumulation across calls, we must ensure consistent size.
		// if (total.size() == 0)
		//	total.clear(dof);
		// else
		//	assert(total.size() == dof);

		compute(lvl, num_vertices,
				jdepth_texture, pids_texture, res_texture, mesh, total);
	}

	void compute(int lvl,
				 int num_vertices,
				 const Texture<Vec3f> &jdepth_texture,
				 const Texture<Vec3<PidType>> &pids_texture,
				 const Texture<float> &res_texture,
				 const Mesh &mesh,
				 DenseLinearProblemx<T> &partial) const
	{
		auto jd = jdepth_texture.MapRead(lvl);
		auto pids = pids_texture.MapRead(lvl);
		auto res = res_texture.MapRead(lvl);

		const std::vector<float> depths = get_depths(mesh);

		const int N = res.width() * res.height();

		const auto nod_res = res.nodata();
		const auto nod_jd = jd.nodata();
		const auto nod_pid = pids.nodata();

		for (int i = 0; i < N; ++i)
		{
			const Vec3f jdepth_i = jd[i];
			const Vec3<PidType> p_i = pids[i];
			const ImageType res_i = res[i];

			// NOTE: keep your existing == checks if your nodata isn't NaN.
			// If nodata might be NaN, use NaN-aware helpers.
			if (res_i == nod_res ||
				jdepth_i == nod_jd ||
				p_i == nod_pid)
				continue;

			const Vec3i pid_int(p_i(0), p_i(1), p_i(2));

			// Reject blended triangle IDs
			const float d0 = std::abs(static_cast<float>(pid_int(0)) - static_cast<float>(p_i(0)));
			const float d1 = std::abs(static_cast<float>(pid_int(1)) - static_cast<float>(p_i(1)));
			const float d2 = std::abs(static_cast<float>(pid_int(2)) - static_cast<float>(p_i(2)));
			if (d0 > 0.f || d1 > 0.f || d2 > 0.f)
				continue;

			const float depth0 = depths[pid_int(0)];
			const float depth1 = depths[pid_int(1)];
			const float depth2 = depths[pid_int(2)];

			const float dd0 = d_depth_d_param(depth0);
			const float dd1 = d_depth_d_param(depth1);
			const float dd2 = d_depth_d_param(depth2);

			Vec<T, 3> J;
			J(0) = jdepth_i(0) * dd0;
			J(1) = jdepth_i(1) * dd1;
			J(2) = jdepth_i(2) * dd2;

			const float w = huber_weight(res_i, mesh_vo::huber_thresh_pix);
			partial.add(J, res_i, w, pid_int);
		}
	}

private:
};

template <typename T>
class HGVertexReducerCPU
{
public:
	explicit HGVertexReducerCPU() {}

	void reduce(int lvl,
				int num_vertices,
				const Texture<Vec3f> &jv0_texture,
				const Texture<Vec3f> &jv1_texture,
				const Texture<Vec3f> &jv2_texture,
				const Texture<Vec3<PidType>> &pids_texture,
				const Texture<float> &res_texture,
				DenseLinearProblemx<T> &total) const
	{
		compute(lvl, num_vertices,
				jv0_texture, jv1_texture, jv2_texture,
				pids_texture, res_texture,
				total);
	}

	void compute(int lvl,
				 int num_vertices,
				 const Texture<Vec3f> &jv0_texture,
				 const Texture<Vec3f> &jv1_texture,
				 const Texture<Vec3f> &jv2_texture,
				 const Texture<Vec3<PidType>> &pids_texture,
				 const Texture<float> &res_texture,
				 DenseLinearProblemx<T> &partial) const
	{
		auto jv0 = jv0_texture.MapRead(lvl);
		auto jv1 = jv1_texture.MapRead(lvl);
		auto jv2 = jv2_texture.MapRead(lvl);
		auto pids = pids_texture.MapRead(lvl);
		auto res = res_texture.MapRead(lvl);

		const std::size_t w = static_cast<std::size_t>(res.width());
		const std::size_t h = static_cast<std::size_t>(res.height());
		const std::size_t N = w * h;

		const auto nod_res = res.nodata();
		const auto nod_jv0 = jv0.nodata();
		const auto nod_jv1 = jv1.nodata();
		const auto nod_jv2 = jv2.nodata();
		const auto nod_pid = pids.nodata();

		for (std::size_t i = 0; i < N; ++i)
		{
			const Vec3f j0 = jv0[i];
			const Vec3f j1 = jv1[i];
			const Vec3f j2 = jv2[i];
			const Vec3<PidType> p = pids[i];
			const ImageType res_i = res[i];

			if (is_nodata_scalar(res_i, nod_res) ||
				is_nodata_vec3(j0, nod_jv0) ||
				is_nodata_vec3(j1, nod_jv1) ||
				is_nodata_vec3(j2, nod_jv2) ||
				is_nodata_vec3(p, nod_pid))
				continue;

			const int pid0 = static_cast<int>(p(0));
			const int pid1 = static_cast<int>(p(1));
			const int pid2 = static_cast<int>(p(2));

			// reject blended ids
			const float d0 = std::abs(static_cast<float>(pid0) - static_cast<float>(p(0)));
			const float d1 = std::abs(static_cast<float>(pid1) - static_cast<float>(p(1)));
			const float d2 = std::abs(static_cast<float>(pid2) - static_cast<float>(p(2)));
			if (d0 > 0.f || d1 > 0.f || d2 > 0.f)
				continue;

			const float w = huber_weight(res_i, mesh_vo::huber_thresh_pix);

			Vec<T, 9> J;
			J(0) = j0(0);
			J(1) = j0(1);
			J(2) = j0(2);
			J(3) = j1(0);
			J(4) = j1(1);
			J(5) = j1(2);
			J(6) = j2(0);
			J(7) = j2(1);
			J(8) = j2(2);

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

			partial.add(J, res_i, w, ids);
		}
	}

private:
};

template <typename T>
class HGDepthExpReducerCPU
{
public:
	explicit HGDepthExpReducerCPU() {}

	void reduce(int lvl,
				int frame_id,
				int num_frames,
				int num_vertices,
				const Texture<Vec3f> &jdepth_texture,
				const Texture<Vec3f> &jexp_texture,
				const Texture<Vec3<PidType>> &pids_texture,
				const Texture<float> &res_texture,
				const Mesh &mesh,
				DenseLinearProblemx<T> &total) const
	{
		compute(lvl, frame_id, num_frames, num_vertices,
				jdepth_texture, jexp_texture, pids_texture,
				res_texture, mesh, total);
	}

	void compute(int lvl,
				 int frame_id,
				 int num_frames,
				 int num_vertices,
				 const Texture<Vec3f> &jdepth_texture,
				 const Texture<Vec3f> &jexp_texture,
				 const Texture<Vec3<PidType>> &pids_texture,
				 const Texture<float> &res_texture,
				 const Mesh &mesh,
				 DenseLinearProblemx<T> &partial) const
	{
		auto jd = jdepth_texture.MapRead(lvl);
		auto je = jexp_texture.MapRead(lvl);
		auto pids = pids_texture.MapRead(lvl);
		auto res = res_texture.MapRead(lvl);

		const std::vector<float> depths = get_depths(mesh);

		const std::size_t w = static_cast<std::size_t>(res.width());
		const std::size_t h = static_cast<std::size_t>(res.height());
		const std::size_t N = w * h;

		const auto nod_res = res.nodata();
		const auto nod_jd = jd.nodata();
		const auto nod_je = je.nodata();
		const auto nod_pid = pids.nodata();

		for (std::size_t i = 0; i < N; ++i)
		{
			const Vec3f jdepth_i = jd[i];
			const Vec3f jexp_i = je[i];
			const Vec3<PidType> p = pids[i];
			const ImageType res_i = res[i];

			if (is_nodata_scalar(res_i, nod_res) ||
				is_nodata_vec3(jdepth_i, nod_jd) ||
				is_nodata_vec3(jexp_i, nod_je) ||
				is_nodata_vec3(p, nod_pid))
				continue;

			const Vec5i ids(p(0), p(1), p(2),
							num_vertices + frame_id * 2,
							num_vertices + frame_id * 2 + 1);

			const float depth0 = depths[ids(0)];
			const float depth1 = depths[ids(1)];
			const float depth2 = depths[ids(2)];

			const float dd0 = d_depth_d_param(depth0);
			const float dd1 = d_depth_d_param(depth1);
			const float dd2 = d_depth_d_param(depth2);

			Vec<T, 5> J;
			J(0) = jdepth_i(0) * dd0;
			J(1) = jdepth_i(1) * dd1;
			J(2) = jdepth_i(2) * dd2;
			J(3) = jexp_i(0);
			J(4) = jexp_i(1);

			const float w = huber_weight(res_i, mesh_vo::huber_thresh_pix);
			partial.add(J, res_i, w, ids);
		}
	}

private:
};

template <typename T>
class HGPoseDepthReducerCPU
{
public:
	explicit HGPoseDepthReducerCPU() {}

	void reduce(int lvl,
				int frame_id,
				int num_frames,
				int num_vertices,
				const Texture<Vec3f> &jtra_texture,
				const Texture<Vec3f> &jrot_texture,
				const Texture<Vec3f> &jdepth_texture,
				const Texture<Vec3<PidType>> &pids_texture,
				const Texture<float> &res_texture,
				const Mesh &mesh,
				DenseLinearProblemx<T> &total) const
	{
		compute(lvl, frame_id, num_frames, num_vertices,
				jtra_texture, jrot_texture, jdepth_texture, pids_texture,
				res_texture, mesh, total);
	}

	void compute(int lvl,
				 int frame_id,
				 int num_frames,
				 int num_vertices,
				 const Texture<Vec3f> &jtra_texture,
				 const Texture<Vec3f> &jrot_texture,
				 const Texture<Vec3f> &jdepth_texture,
				 const Texture<Vec3<PidType>> &pids_texture,
				 const Texture<float> &res_texture,
				 const Mesh &mesh,
				 DenseLinearProblemx<T> &partial) const
	{
		auto jt = jtra_texture.MapRead(lvl);
		auto jr = jrot_texture.MapRead(lvl);
		auto jd = jdepth_texture.MapRead(lvl);
		auto pids = pids_texture.MapRead(lvl);
		auto res = res_texture.MapRead(lvl);

		const std::vector<float> depths = get_depths(mesh);

		const std::size_t w = static_cast<std::size_t>(res.width());
		const std::size_t h = static_cast<std::size_t>(res.height());
		const std::size_t N = w * h;

		const auto nod_res = res.nodata();
		const auto nod_jt = jt.nodata();
		const auto nod_jr = jr.nodata();
		const auto nod_jd = jd.nodata();
		const auto nod_pid = pids.nodata();

		for (std::size_t i = 0; i < N; ++i)
		{
			const Vec3f jtra_i = jt[i];
			const Vec3f jrot_i = jr[i];
			const Vec3f jdepth_i = jd[i];
			const Vec3<PidType> p = pids[i];
			const float res_i = res[i];

			if (is_nodata_scalar(res_i, nod_res) ||
				is_nodata_vec3(jtra_i, nod_jt) ||
				is_nodata_vec3(jrot_i, nod_jr) ||
				is_nodata_vec3(jdepth_i, nod_jd) ||
				is_nodata_vec3(p, nod_pid))
				continue;

			Vec<int, 9> ids;
			ids(0) = p(0);
			ids(1) = p(1);
			ids(2) = p(2);
			ids(3) = num_vertices + frame_id * 6;
			ids(4) = num_vertices + frame_id * 6 + 1;
			ids(5) = num_vertices + frame_id * 6 + 2;
			ids(6) = num_vertices + frame_id * 6 + 3;
			ids(7) = num_vertices + frame_id * 6 + 4;
			ids(8) = num_vertices + frame_id * 6 + 5;

			const float depth0 = depths[ids(0)];
			const float depth1 = depths[ids(1)];
			const float depth2 = depths[ids(2)];

			const float dd0 = d_depth_d_param(depth0);
			const float dd1 = d_depth_d_param(depth1);
			const float dd2 = d_depth_d_param(depth2);

			Vec<T, 9> J;
			J(0) = jdepth_i(0) * dd0;
			J(1) = jdepth_i(1) * dd1;
			J(2) = jdepth_i(2) * dd2;
			J(3) = jtra_i(0);
			J(4) = jtra_i(1);
			J(5) = jtra_i(2);
			J(6) = jrot_i(0);
			J(7) = jrot_i(1);
			J(8) = jrot_i(2);

			const float w = huber_weight(res_i, mesh_vo::huber_thresh_pix);
			partial.add(J, res_i, w, ids);
		}
	}

private:
};

template <typename T>
class HGPoseExpDepthReducerCPU
{
public:
	explicit HGPoseExpDepthReducerCPU() {}

	void reduce(int lvl,
				int frame_id,
				int num_frames,
				int num_vertices,
				const Texture<Vec3f> &jtra_texture,
				const Texture<Vec3f> &jrot_texture,
				const Texture<Vec3f> &jexp_texture,
				const Texture<Vec3f> &jdepth_texture,
				const Texture<Vec3<PidType>> &pids_texture,
				const Texture<float> &res_texture,
				const Mesh &mesh,
				DenseLinearProblemx<T> &total) const
	{
		compute(lvl, frame_id, num_frames, num_vertices,
				jtra_texture, jrot_texture, jexp_texture, jdepth_texture, pids_texture,
				res_texture, mesh, total);
	}

	void compute(int lvl,
				 int frame_id,
				 int num_frames,
				 int num_vertices,
				 const Texture<Vec3f> &jtra_texture,
				 const Texture<Vec3f> &jrot_texture,
				 const Texture<Vec3f> &jexp_texture,
				 const Texture<Vec3f> &jdepth_texture,
				 const Texture<Vec3<PidType>> &pids_texture,
				 const Texture<float> &res_texture,
				 const Mesh &mesh,
				 DenseLinearProblemx<T> &partial) const
	{
		auto jt = jtra_texture.MapRead(lvl);
		auto jr = jrot_texture.MapRead(lvl);
		auto je = jexp_texture.MapRead(lvl);
		auto jd = jdepth_texture.MapRead(lvl);
		auto pids = pids_texture.MapRead(lvl);
		auto res = res_texture.MapRead(lvl);

		const std::vector<float> depths = get_depths(mesh);

		const std::size_t w = static_cast<std::size_t>(res.width());
		const std::size_t h = static_cast<std::size_t>(res.height());
		const std::size_t N = w * h;

		const auto nod_res = res.nodata();
		const auto nod_jt = jt.nodata();
		const auto nod_jr = jr.nodata();
		const auto nod_je = je.nodata();
		const auto nod_jd = jd.nodata();
		const auto nod_pid = pids.nodata();

		for (std::size_t i = 0; i < N; ++i)
		{
			const Vec3f jtra_i = jt[i];
			const Vec3f jrot_i = jr[i];
			const Vec3f jexp_i = je[i];
			const Vec3f jdepth_i = jd[i];
			const Vec3<PidType> p = pids[i];
			const float res_i = res[i];

			if (is_nodata_scalar(res_i, nod_res) ||
				is_nodata_vec3(jtra_i, nod_jt) ||
				is_nodata_vec3(jrot_i, nod_jr) ||
				is_nodata_vec3(jexp_i, nod_je) ||
				is_nodata_vec3(jdepth_i, nod_jd) ||
				is_nodata_vec3(p, nod_pid))
				continue;

			Vec<int, 11> ids;
			ids(0) = p(0);
			ids(1) = p(1);
			ids(2) = p(2);
			ids(3) = num_vertices + frame_id * 8;
			ids(4) = num_vertices + frame_id * 8 + 1;
			ids(5) = num_vertices + frame_id * 8 + 2;
			ids(6) = num_vertices + frame_id * 8 + 3;
			ids(7) = num_vertices + frame_id * 8 + 4;
			ids(8) = num_vertices + frame_id * 8 + 5;
			ids(9) = num_vertices + frame_id * 8 + 6;
			ids(10) = num_vertices + frame_id * 8 + 7;

			const float depth0 = depths[ids(0)];
			const float depth1 = depths[ids(1)];
			const float depth2 = depths[ids(2)];

			const float dd0 = d_depth_d_param(depth0);
			const float dd1 = d_depth_d_param(depth1);
			const float dd2 = d_depth_d_param(depth2);

			Vec<T, 11> J;
			J(0) = jdepth_i(0) * dd0;
			J(1) = jdepth_i(1) * dd1;
			J(2) = jdepth_i(2) * dd2;
			J(3) = jtra_i(0);
			J(4) = jtra_i(1);
			J(5) = jtra_i(2);
			J(6) = jrot_i(0);
			J(7) = jrot_i(1);
			J(8) = jrot_i(2);
			J(9) = jexp_i(0);
			J(10) = jexp_i(1);

			const float w = huber_weight(res_i, mesh_vo::huber_thresh_pix);
			partial.add(J, res_i, w, ids);
		}
	}

private:
};
