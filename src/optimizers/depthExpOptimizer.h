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

class DepthExpOptimizer : public BaseOptimizer<DepthExpOptimizer,
                                               Matx<float>,
                                               Vecx<float>,
                                               Error<float>,
                                               DenseLinearProblemx<float>,
                                               Solverx<float>>
{
public:
    using Base = BaseOptimizer<DepthExpOptimizer,
                               Matx<float>,
                               Vecx<float>,
                               Error<float>,
                               DenseLinearProblemx<float>,
                               Solverx<float>>;

    DepthExpOptimizer(int w, int h, bool printlog = false)
        : Base(printlog),
          jdepth_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          jexp_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          pids_texture_(w, h, Vec3<PidType>(-1, -1, -1)),
          res_texture_(w, h, 0.0)
    {
    }

    int numParams()
    {
        return numParams_;
    }

    void reset(std::span<const Frame *const> frames, const KeyFrame &kframe, DenseLinearProblemx<float> &problem, Solverx<float> &solver)
    {
        init_depths_ = get_depths(kframe.mesh());

        local_poses_.clear();
        init_exps_.clear();
        for (auto frame : frames)
        {
            local_poses_.push_back(kframe.global_pose_to_local(frame->global_pose()));
            init_exps_.push_back(frame->local_exposure());
        }

        // triangles_ = get_indices(kframe.mesh());
        edges_ = get_edges(kframe.mesh());
        numDepths_ = kframe.mesh().vertex_count();
        numParams_ = numDepths_ + 2 * frames.size();

        best_depths_ = init_depths_;
        best_exps_ = init_exps_;

        depths_ = best_depths_;
        exps_ = best_exps_;

        problem = DenseLinearProblemx<float>(numParams_);
        solver = Solverx<float>(numParams_);
    }

    float regu_error() const
    {
        return regu_depth(depths_, edges_);
    }

    float prior_error() const
    {
        return prior_depth(depths_, init_depths_);
    }

    void regu_jacobian(DenseLinearProblemx<float> &problem)
    {
        regu_depth_jacobian(depths_, edges_, problem);
    }

    void prior_jacobian(DenseLinearProblemx<float> &problem)
    {
        prior_depth_jacobian(depths_, init_depths_, problem);
    }

    void apply_inc(std::span<Frame *const> frames, KeyFrame &kframe, const Vecx<float> &inc)
    {
        depths_.clear();
        exps_.clear();
        for (size_t i = 0; i < numDepths_; i++)
        {
            float new_depth = fromParamToDepth(fromDepthToParam(best_depths_[i]) + inc(i));
            if (new_depth < RenderConstants::NEAR_PLANE)
                new_depth = RenderConstants::NEAR_PLANE;
            if (new_depth > RenderConstants::FAR_PLANE)
                new_depth = RenderConstants::FAR_PLANE;

            depths_.push_back(new_depth);
        }

        for (size_t i = 0; i < frames.size(); i++)
        {
            Vec2<float> exp_inc(inc(numDepths_ + i * 2 + 0), inc(numDepths_ + i * 2 + 1));
            Vec2<float> new_exp = best_exps_[i] + exp_inc;
            exps_.push_back(new_exp);
        }
    }

    void update_params(std::span<Frame *const> frames, KeyFrame &kframe)
    {
        set_depths(kframe.mesh(), best_depths_);

        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i]->local_exposure() = best_exps_[i];
        }
    }

    void update_best_params()
    {
        best_depths_ = depths_;
        best_exps_ = exps_;
    }

    void restore_best_params(std::span<Frame *const> frames, KeyFrame &kframe)
    {
        depths_ = best_depths_;
        exps_ = best_exps_;
    }

    Error<float> compute_error(std::span<const Frame *const> frames, const KeyFrame &kframe, const Cameraf &cam, int in_lvl, int out_lvl)
    {
        Error<float> total;
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            residualrenderer_.Render(kframe.mesh(), local_poses_[i], frames[i]->local_exposure(), cam, in_lvl, out_lvl, kframe.image(), frames[i]->image(), res_texture_);
            residualreducer_.reduce(out_lvl, res_texture_, total);
        }
        return total;
    }

    void compute_problem(std::span<const Frame *const> frames, const KeyFrame &kframe, const Cameraf &cam, int in_lvl, int out_lvl, DenseLinearProblemx<float> &total)
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
                                   jdepth_texture_,
                                   jexp_texture_,
                                   pids_texture_,
                                   res_texture_);
            hgmapreducer_.reduce(out_lvl,
                                 frame_idx,
                                 frames.size(),
                                 kframe.mesh().vertex_count(),
                                 jdepth_texture_,
                                 jexp_texture_,
                                 pids_texture_,
                                 res_texture_,
                                 kframe.mesh(),
                                 total);
        }
    }

private:
    JDepthExpRenderer jdepthrenderer_;
    HGDepthExpReducerCPU<float> hgmapreducer_;
    ResidualRenderer residualrenderer_;
    ResidualReducerCPU<float> residualreducer_;

    Texture<Vec3f> jdepth_texture_;
    Texture<Vec3f> jexp_texture_;
    Texture<Vec3<PidType>> pids_texture_;
    Texture<float> res_texture_;

    std::vector<float> init_depths_;
    std::vector<Vec2f> init_exps_;

    std::vector<float> best_depths_;
    std::vector<Vec2f> best_exps_;

    std::vector<float> depths_;
    std::vector<Vec2f> exps_;

    std::vector<SE3f> local_poses_;
    // std::vector<Vec3<int>> triangles_;
    std::vector<Vec2i> edges_;

    int numDepths_;
    int numParams_;
};
