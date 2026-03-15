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

class PoseExpDepthOptimizer : public BaseOptimizer<PoseExpDepthOptimizer,
                                                   Matx<float>,
                                                   Vecx<float>,
                                                   Error<float>,
                                                   DenseLinearProblemx<float>,
                                                   Solverx<float>>
{
public:
    using Base = BaseOptimizer<PoseExpDepthOptimizer,
                               Matx<float>,
                               Vecx<float>,
                               Error<float>,
                               DenseLinearProblemx<float>,
                               Solverx<float>>;
    PoseExpDepthOptimizer(int w, int h, bool printlog = false)
        : Base(printlog),
          jtra_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          jrot_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
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
        best_poses_.clear();
        best_exps_.clear();
        for (int i = 0; i < frames.size(); i++)
        {
            best_poses_.push_back(kframe.global_pose_to_local(frames[i]->global_pose()));
            best_exps_.push_back(frames[i]->local_exposure());
        }
        // triangles_ = get_indices(kframe.mesh());
        edges_ = get_edges(kframe.mesh());
        numDepths_ = kframe.mesh().vertex_count();
        numParams_ = numDepths_ + 8 * frames.size();

        depths_ = best_depths_;
        poses_ = best_poses_;
        exps_ = best_exps_;

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
        //depths_.clear();
        //poses_.clear();
        //exps_.clear();

        for (size_t i = 0; i < numDepths_; i++)
        {
            float new_depth = fromParamToDepth(fromDepthToParam(best_depths_[i]) + inc(i));
            if (new_depth < RenderConstants::NEAR_PLANE)
                new_depth = RenderConstants::NEAR_PLANE;
            if (new_depth > RenderConstants::FAR_PLANE)
                new_depth = RenderConstants::FAR_PLANE;

            depths_[i] = new_depth;
        }

        for (size_t i = 0; i < frames.size(); i++)
        {
            Vec6f pose_inc(inc(numDepths_ + i * 8 + 0),
                                 inc(numDepths_ + i * 8 + 1),
                                 inc(numDepths_ + i * 8 + 2),
                                 inc(numDepths_ + i * 8 + 3),
                                 inc(numDepths_ + i * 8 + 4),
                                 inc(numDepths_ + i * 8 + 5));
            SE3f new_pose = best_poses_[i] * SE3f::exp(pose_inc);
            poses_[i] = new_pose;

            Vec2f exp_inc(inc(numDepths_ + i * 8 + 6),
                                inc(numDepths_ + i * 8 + 7));
            Vec2f new_exp = best_exps_[i] + exp_inc;
            exps_[i] = new_exp;
        }

        set_depths(kframe.mesh(), depths_);

        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i]->global_pose() = kframe.local_pose_to_global(poses_[i]);
            frames[i]->local_exposure() = exps_[i];
        }
    }

    void update_params(std::span<Frame *const> frames, KeyFrame &kframe)
    {
        set_depths(kframe.mesh(), best_depths_);

        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i]->global_pose() = kframe.local_pose_to_global(best_poses_[i]);
            frames[i]->local_exposure() = best_exps_[i];
        }
    }

    void update_best_params()
    {
        best_depths_ = depths_;
        best_poses_ = poses_;
        best_exps_ = exps_;
    }

    void restore_best_params(std::span<Frame *const> frames, KeyFrame &kframe)
    {
        depths_ = best_depths_;
        poses_ = best_poses_;
        exps_ = best_exps_;

        set_depths(kframe.mesh(), best_depths_);

        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i]->global_pose() = kframe.local_pose_to_global(best_poses_[i]);
            frames[i]->local_exposure() = best_exps_[i];
        }
    }

    Error<float> compute_error(std::span<const Frame *const> frames, const KeyFrame &kframe, const Cameraf &cam, int in_lvl, int out_lvl)
    {
        Error<float> total;
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            residualrenderer_.Render(kframe.mesh(), poses_[i], exps_[i], cam, in_lvl, out_lvl, kframe.image(), frames[i]->image(), res_texture_);
            residualreducer_.reduce(out_lvl, res_texture_, total);
        }
        return total;
    }

    void compute_problem(std::span<const Frame *const> frames, const KeyFrame &kframe, const Cameraf &cam, int in_lvl, int out_lvl, DenseLinearProblemx<float> &total)
    {
        for (std::size_t frame_idx = 0; frame_idx < frames.size(); frame_idx++)
        {
            jposedepthrenderer_.Render(kframe.mesh(),
                                       best_poses_[frame_idx],
                                       best_exps_[frame_idx],
                                       cam,
                                       in_lvl, out_lvl,
                                       kframe.image(),
                                       frames[frame_idx]->image(),
                                       frames[frame_idx]->didxy(),
                                       jtra_texture_,
                                       jrot_texture_,
                                       jexp_texture_,
                                       jdepth_texture_,
                                       pids_texture_,
                                       res_texture_);
            hgposedepthreducer_.reduce(out_lvl,
                                       frame_idx, frames.size(), numDepths_,
                                       jtra_texture_,
                                       jrot_texture_,
                                       jexp_texture_,
                                       jdepth_texture_,
                                       pids_texture_,
                                       res_texture_,
                                       kframe.mesh(),
                                       total);
        }
    }

private:
    JPoseExpDepthRenderer jposedepthrenderer_;
    HGPoseExpDepthReducerCPU<float> hgposedepthreducer_;
    ResidualRenderer residualrenderer_;
    ResidualReducerCPU<float> residualreducer_;

    Texture<Vec3f> jtra_texture_;
    Texture<Vec3f> jrot_texture_;
    Texture<Vec3f> jdepth_texture_;
    Texture<Vec3f> jexp_texture_;
    Texture<Vec3<PidType>> pids_texture_;
    Texture<float> res_texture_;

    std::vector<float> best_depths_;
    std::vector<SE3f> best_poses_;
    std::vector<Vec2f> best_exps_;

    std::vector<float> depths_;
    std::vector<SE3f> poses_;
    std::vector<Vec2f> exps_;

    // std::vector<Vec3<int>> triangles_;
    std::vector<Vec2i> edges_;

    int numDepths_;
    int numParams_;
};
