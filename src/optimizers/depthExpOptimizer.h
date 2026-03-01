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
                                               DenseLinearProblemx,
                                               Solverx<float>>
{
public:
    using Base = BaseOptimizer<DepthExpOptimizer,
                               Matx<float>,
                               Vecx<float>,
                               DenseLinearProblemx,
                               Solverx<float>>;

    DepthExpOptimizer(int w, int h, bool printlog = false)
        : Base(w, h, printlog),
          jdepth_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          jexp_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          pids_texture_(w, h, Vec3<PidType>(-1, -1, -1))
    {
    }

    int numParams()
    {
        return numParams_;
    }

    void reset(std::span<const Frame *const> frames, const KeyFrame &kframe, DenseLinearProblemx &problem, Solverx<float> &solver)
    {
        best_depths_ = get_depths(kframe.mesh());

        local_poses_.clear();
        best_exps_.clear();
        for (auto frame : frames)
        {
            local_poses_.push_back(kframe.global_pose_to_local(frame->global_pose()));
            best_exps_.push_back(frame->local_exposure());
        }

        // triangles_ = get_indices(kframe.mesh());
        edges_ = get_edges(kframe.mesh());
        numDepths_ = kframe.mesh().vertex_count();
        numParams_ = numDepths_ + 2 * frames.size();

        depths_ = best_depths_;
        exps_ = best_exps_;

        problem = DenseLinearProblemx(numParams_);
        solver = Solverx<float>(numParams_);
    }

    float regu_error() const
    {
        return regu_depth(depths_, edges_);
    }

    void regu_jacobian(DenseLinearProblemx &problem)
    {
        regu_depth_jacobian(depths_, edges_, problem);
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

    Error compute_error(std::span<const Frame *const> frames, const KeyFrame &kframe, const Camera &cam, int in_lvl, int out_lvl)
    {
        Error total;
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            imagerenderer_.Render(kframe.mesh(), local_poses_[i], frames[i]->local_exposure(), cam, in_lvl, out_lvl, kframe.image(), image_texture_);
            residualreducer_.reduce(out_lvl, image_texture_, frames[i]->image(), total);
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
                                   frames[frame_idx]->didxy(),
                                   image_texture_,
                                   jdepth_texture_,
                                   jexp_texture_,
                                   pids_texture_);
            hgmapreducer_.reduce(out_lvl,
                                 frame_idx,
                                 frames.size(),
                                 kframe.mesh().vertex_count(),
                                 jdepth_texture_,
                                 jexp_texture_,
                                 pids_texture_,
                                 image_texture_,
                                 frames[frame_idx]->image(),
                                 kframe.mesh(),
                                 total);
        }
    }

private:
    JDepthExpRenderer jdepthrenderer_;
    HGDepthExpReducerCPU hgmapreducer_;

    Texture<Vec3<float>> jdepth_texture_;
    Texture<Vec3<float>> jexp_texture_;
    Texture<Vec3<PidType>> pids_texture_;

    std::vector<float> best_depths_;
    std::vector<Vec2f> best_exps_;

    std::vector<float> depths_;
    std::vector<Vec2f> exps_;

    std::vector<SE3f> local_poses_;
    // std::vector<Vec3<int>> triangles_;
    std::vector<Vec2<int>> edges_;

    int numDepths_;
    int numParams_;
};
