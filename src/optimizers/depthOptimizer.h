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

class DepthOptimizer : public BaseOptimizer<DepthOptimizer,
                                            Matx<float>,
                                            Vecx<float>,
                                            Error<float>,
                                            DenseLinearProblemx<float>,
                                            Solverx<float>>
{
public:
    using Base = BaseOptimizer<DepthOptimizer,
                               Matx<float>,
                               Vecx<float>,
                               Error<float>,
                               DenseLinearProblemx<float>,
                               Solverx<float>>;

    DepthOptimizer(int w, int h, bool printlog = false)
        : Base(printlog),
          jdepth_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          jexp_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          pids_texture_(w, h, Vec3<PidType>(-1, -1, -1)),
          res_texture_(w, h, 0.0)
    {
    }

    int numParams() const
    {
        return numParams_;
    }

    void reset(std::span<const Frame *const> frames, const KeyFrame &kframe, DenseLinearProblemx<float> &problem, Solverx<float> &solver)
    {
        best_depths_ = get_depths(kframe.mesh());

        local_poses_.clear();
        for (auto &frame : frames)
            local_poses_.push_back(kframe.global_pose_to_local(frame->global_pose()));

        // triangles_ = get_indices(kframe.mesh());
        edges_ = get_edges(kframe.mesh());
        numDepths_ = kframe.mesh().vertex_count();
        numParams_ = numDepths_;

        depths_ = best_depths_;

        problem = DenseLinearProblemx<float>(numParams_);
        solver = Solverx<float>(numParams_);
    }

    float regu_error() const
    {
        return regu_depth(depths_, edges_);
    }

    void regu_jacobian(DenseLinearProblemx<float> &problem) const
    {
        regu_depth_jacobian(depths_, edges_, problem);
    }

    void apply_inc(std::span<Frame *const> frames, KeyFrame &kframe, const Vecx<float> &inc)
    {
        depths_.clear();
        for (size_t i = 0; i < numDepths_; i++)
        {
            float new_depth = fromParamToDepth(fromDepthToParam(best_depths_[i]) + inc(i));
            if (new_depth < RenderConstants::NEAR_PLANE)
                new_depth = RenderConstants::NEAR_PLANE;
            if (new_depth > RenderConstants::FAR_PLANE)
                new_depth = RenderConstants::FAR_PLANE;

            depths_.push_back(new_depth);
        }
        set_depths(kframe.mesh(), depths_);
    }

    void update_params(std::span<Frame *const> frames, KeyFrame &kframe)
    {
        set_depths(kframe.mesh(), best_depths_);
    }

    void update_best_params()
    {
        best_depths_ = depths_;
    }

    void restore_best_params(std::span<Frame *const> frames, KeyFrame &kframe)
    {
        depths_ = best_depths_;
        set_depths(kframe.mesh(), best_depths_);
    }

    Error<float> compute_error(std::span<const Frame *const> frames, const KeyFrame &kframe, const Cameraf &cam, int in_lvl, int out_lvl)
    {
        Error<float> total;
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            if (frames[i]->id() == kframe.id())
                continue;
            residualrenderer_.Render(kframe.mesh(),
                                     local_poses_[i],
                                     frames[i]->local_exposure(),
                                     cam, in_lvl, out_lvl,
                                     kframe.image(), frames[i]->image(),
                                     res_texture_);
            residualreducer_.reduce(out_lvl, res_texture_, total);
        }
        return total;
    }

    void compute_problem(std::span<const Frame *const> frames, const KeyFrame &kframe, const Cameraf &cam, int in_lvl, int out_lvl, DenseLinearProblemx<float> &total)
    {
        for (std::size_t frame_idx = 0; frame_idx < frames.size(); frame_idx++)
        {
            if (frames[frame_idx]->id() == kframe.id())
                continue;
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
                                 pids_texture_,
                                 res_texture_,
                                 kframe.mesh(), total);
        }
    }

private:
    JDepthExpRenderer jdepthrenderer_;
    HGDepthReducerCPU<float> hgmapreducer_;
    ResidualRenderer residualrenderer_;
    ResidualReducerCPU<float> residualreducer_;

    Texture<Vec3<float>> jdepth_texture_;
    Texture<Vec3<float>> jexp_texture_;
    Texture<Vec3<PidType>> pids_texture_;
    Texture<float> res_texture_;

    std::vector<float> best_depths_;
    std::vector<float> depths_;

    std::vector<SE3f> local_poses_;
    // std::vector<Vec3<int>> triangles_;
    std::vector<Vec2<int>> edges_;

    int numDepths_;
    int numParams_;
};
