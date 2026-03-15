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

class PoseOptimizer : public BaseOptimizer<PoseOptimizer,
                                           Mat6<float>,
                                           Vec6<float>,
                                           Error<float>,
                                           DenseLinearProblem<float, 6>,
                                           Solver<float, 6>>
{
public:
    using Base = BaseOptimizer<PoseOptimizer,
                               Mat6<float>,
                               Vec6<float>,
                               Error<float>,
                               DenseLinearProblem<float, 6>,
                               Solver<float, 6>>;
    PoseOptimizer(int w, int h, bool printlog = false)
        : Base(printlog),
          jtra_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          jrot_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          jexp_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          res_texture_(w, h, 0.0)
    {
    }

    int numParams()
    {
        return numParams_;
    }

    void reset(std::span<const Frame *const> frames, const KeyFrame &kframe, DenseLinearProblem<float, 6> &problem, Solver<float, 6> &solver)
    {
        // scale_ = mean_depth(kframe.mesh());
        best_local_poses_.clear();
        for (int i = 0; i < frames.size(); i++)
        {
            SE3f local_pose = kframe.global_pose_to_local(frames[i]->global_pose());
            best_local_poses_.push_back(local_pose);
        }
        local_poses_ = best_local_poses_;
        numParams_ = 6 * frames.size();
    }

    float regu_error() const
    {
        return 0; // regu_depth(depths_, edges_);
    }

    void regu_jacobian(DenseLinearProblem<float, 6> &problem) const
    {
        // regu_depth_jacobian(depths_, edges_, problem);
    }

    void apply_inc(std::span<Frame *const> frames, KeyFrame &kframe, const Vec6<float> &inc)
    {
        //local_poses_.clear();

        for (size_t i = 0; i < frames.size(); i++)
        {
            Vec6<float> pose_inc(inc(i * 6 + 0),
                                 inc(i * 6 + 1),
                                 inc(i * 6 + 2),
                                 inc(i * 6 + 3),
                                 inc(i * 6 + 4),
                                 inc(i * 6 + 5));
            SE3f new_pose = best_local_poses_[i] * SE3f::exp(pose_inc);
            local_poses_[i] = new_pose;
        }

        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i]->global_pose() = kframe.local_pose_to_global(local_poses_[i]);
        }
    }

    void update_params(std::span<Frame *const> frames, KeyFrame &kframe)
    {
        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i]->global_pose() = kframe.local_pose_to_global(best_local_poses_[i]);
        }
    }

    void update_best_params()
    {
        best_local_poses_ = local_poses_;
    }

    void restore_best_params(std::span<Frame *const> frames, KeyFrame &kframe)
    {
        local_poses_ = best_local_poses_;

        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i]->global_pose() = kframe.local_pose_to_global(best_local_poses_[i]);
        }
    }

    Error<float> compute_error(std::span<const Frame *const> frames, const KeyFrame &kframe, const Cameraf &cam, int in_lvl, int out_lvl)
    {
        Error<float> total;
        for (std::size_t frame_idx = 0; frame_idx < frames.size(); frame_idx++)
        {
            residualrenderer_.Render(kframe.mesh(), local_poses_[frame_idx], frames[frame_idx]->local_exposure(), cam, in_lvl, out_lvl, kframe.image(), frames[frame_idx]->image(), res_texture_);
            residualreducer_.reduce(out_lvl, res_texture_, total);
        }
        return total;
    }

    void compute_problem(std::span<const Frame *const> frames, const KeyFrame &kframe, Cameraf &cam, int in_lvl, int out_lvl, DenseLinearProblem<float, 6> &total)
    {
        for (std::size_t frame_idx = 0; frame_idx < frames.size(); frame_idx++)
        {
            jposedepthrenderer_.Render(kframe.mesh(),
                                       local_poses_[frame_idx],
                                       frames[frame_idx]->local_exposure(),
                                       cam,
                                       in_lvl, out_lvl,
                                       kframe.image(),
                                       frames[frame_idx]->image(),
                                       frames[frame_idx]->didxy(),
                                       jtra_texture_,
                                       jrot_texture_,
                                       jexp_texture_,
                                       res_texture_);
            hgposedepthreducer_.reduce(out_lvl,
                                       jtra_texture_,
                                       jrot_texture_,
                                       res_texture_,
                                       total);
        }
    }

private:
    JPoseExpRenderer jposedepthrenderer_;
    HGPoseReducerCPU<float> hgposedepthreducer_;
    ResidualRenderer residualrenderer_;
    ResidualReducerCPU<float> residualreducer_;

    Texture<Vec3f> jtra_texture_;
    Texture<Vec3f> jrot_texture_;
    Texture<Vec3f> jexp_texture_;
    Texture<float> res_texture_;

    std::vector<SE3f> best_local_poses_;
    std::vector<SE3f> local_poses_;

    int numParams_;
};
