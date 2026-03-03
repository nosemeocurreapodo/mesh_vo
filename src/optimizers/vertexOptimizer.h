#pragma once

#include <iostream>
#include <vector>
#include "params.h"
#include "mpdr/common/types.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "common/DenseLinearProblem.h"
#include "common/depthParam.h"
#include "common/reducer.h"
#include "optimizers/baseOptimizer.h"

class VertexOptimizer : public BaseOptimizer<VertexOptimizer,
                                             Matx<float>,
                                             Vecx<float>,
                                             DenseLinearProblemx,
                                             Solverx<float>>
{
public:
    using Base = BaseOptimizer<VertexOptimizer,
                               Matx<float>,
                               Vecx<float>,
                               DenseLinearProblemx,
                               Solverx<float>>;

    VertexOptimizer(int w, int h, bool printlog = false)
        : Base(printlog),
          jv0_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          jv1_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          jv2_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          jexp_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          pids_texture_(w, h, Vec3<PidType>(-1, -1, -1)),
          res_texture_(w, h, 0.0)
    {
    }

    int numParams() const
    {
        return numParams_;
    }

    void reset(const std::span<const Frame *const> frames, const KeyFrame &kframe, DenseLinearProblemx &problem, Solverx<float> &solver)
    {
        best_vertex_ = get_vertices(kframe.mesh());

        // triangles_ = get_indices(kframe.mesh());
        edges_ = get_edges(kframe.mesh());
        numVertex_ = kframe.mesh().vertex_count();
        numParams_ = numVertex_ * 3;

        vertex_ = best_vertex_;

        problem = DenseLinearProblemx(numParams_);
        solver = Solverx<float>(numParams_);
    }

    float regu_error() const
    {
        return 0; // return regu_depth(depths_, edges_);
    }

    void regu_jacobian(DenseLinearProblemx &problem) const
    {
        // regu_depth_jacobian(depths_, edges_, problem);
    }

    void apply_inc(std::span<Frame *const> frames, KeyFrame &kframe, const Vecx<float> &inc)
    {
        vertex_.clear();
        for (size_t i = 0; i < numVertex_; i++)
        {
            Vec3<float> vinc(inc(i * 3 + 0), inc(i * 3 + 1), inc(i * 3 + 2));
            Vec3<float> new_vertex = vertex_[i] + vinc;

            vertex_.push_back(new_vertex);
        }
    }

    void update_params(std::span<Frame *const> frames, KeyFrame &kframe)
    {
        set_vertices(kframe.mesh(), best_vertex_);
    }

    void update_best_params()
    {
        best_vertex_ = vertex_;
    }

    void restore_best_params(std::span<Frame *const> frames, KeyFrame &kframe)
    {
        vertex_ = best_vertex_;

        set_vertices(kframe.mesh(), best_vertex_);
    }

    Error compute_error(std::span<const Frame *const> frames, const KeyFrame &kframe, const Camera &cam, int in_lvl, int out_lvl)
    {
        Error total;
        for (std::size_t frame_idx = 0; frame_idx < frames.size(); frame_idx++)
        {
            residualrenderer_.Render(kframe.mesh(), local_poses_[frame_idx], frames[frame_idx]->local_exposure(), cam, in_lvl, out_lvl, kframe.image(), frames[frame_idx]->image(), res_texture_);
            residualreducer_.reduce(out_lvl, res_texture_, total);
        }
        return total;
    }

    void compute_problem(std::span<const Frame *const> frames, const KeyFrame &kframe, const Camera &cam, int in_lvl, int out_lvl, DenseLinearProblemx &total)
    {
        for (std::size_t frame_idx = 0; frame_idx < frames.size(); frame_idx++)
        {
            jdepthrenderer_.Render(kframe.mesh(),
                                   local_poses_[frame_idx],
                                   frames[frame_idx]->local_exposure(),
                                   cam,
                                   in_lvl, out_lvl,
                                   kframe.image(),
                                   frames[frame_idx]->image(),
                                   frames[frame_idx]->didxy(),
                                   jv0_texture_,
                                   jv1_texture_,
                                   jv2_texture_,
                                   jexp_texture_,
                                   pids_texture_,
                                   res_texture_);
            hgmapreducer_.reduce(out_lvl,
                                 kframe.mesh().vertex_count(),
                                 jv0_texture_,
                                 jv1_texture_,
                                 jv2_texture_,
                                 pids_texture_,
                                 res_texture_,
                                 total);
        }
    }

private:
    JVertexExpRenderer jdepthrenderer_;
    HGVertexReducerCPU hgmapreducer_;
    ResidualRenderer residualrenderer_;
    ResidualReducerCPU residualreducer_;

    Texture<Vec3<float>> jv0_texture_;
    Texture<Vec3<float>> jv1_texture_;
    Texture<Vec3<float>> jv2_texture_;
    Texture<Vec3<float>> jexp_texture_;
    Texture<Vec3<PidType>> pids_texture_;
    Texture<float> res_texture_;

    std::vector<Vec3<float>> best_vertex_;
    std::vector<Vec3<float>> vertex_;

    std::vector<SE3f> local_poses_;
    // std::vector<Vec3<int>> triangles_;
    std::vector<Vec2<int>> edges_;

    int numVertex_;
    int numParams_;
};
