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

class PoseExpOptimizer : public BaseOptimizer<PoseExpOptimizer,
                                              Mat<float, 8, 8>,
                                              Vec<float, 8>,
                                              Error<float>,
                                              DenseLinearProblem<float, 8>,
                                              Solver<float, 8>>
{
public:
    using Base = BaseOptimizer<PoseExpOptimizer,
                               Mat<float, 8, 8>,
                               Vec<float, 8>,
                               Error<float>,
                               DenseLinearProblem<float, 8>,
                               Solver<float, 8>>;
    PoseExpOptimizer(int w, int h, bool printlog = false)
        : Base(printlog),
          jtra_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          jrot_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          jexp_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          res_texture_(w, h, 0.0)
    {
    }

    int numParams() const
    {
        return numParams_;
    }

    void reset(std::span<const Frame *const> frames, const KeyFrame &kframe, DenseLinearProblem<float, 8> &problem, Solver<float, 8> &solver)
    {
        init_poses_.clear();
        init_exps_.clear();
        for (int i = 0; i < frames.size(); i++)
        {
            init_poses_.push_back(kframe.global_pose_to_local(frames[i]->global_pose()));
            init_exps_.push_back(frames[i]->local_exposure());
        }

        best_poses_ = init_poses_;
        best_exps_ = init_exps_;

        poses_ = best_poses_;
        exps_ = best_exps_;
        numParams_ = 8 * frames.size();
    }

    float regu_error() const
    {
        return 0; // regu_depth(depths_, edges_);
    }

    float prior_error() const
    {
        return 0; // regu_depth(depths_, edges_);
    }

    void regu_jacobian(DenseLinearProblem<float, 8> &problem) const
    {
        // regu_depth_jacobian(depths_, edges_, problem);
    }

    void prior_jacobian(DenseLinearProblem<float, 8> &problem) const
    {
        // regu_depth_jacobian(depths_, edges_, problem);
    }

    void apply_inc(std::span<Frame *const> frames, KeyFrame &kframe, const Vec<float, 8> &inc)
    {
        // poses_.clear();
        // exps_.clear();

        for (size_t i = 0; i < frames.size(); i++)
        {
            Vec6<float> pose_inc(inc(i * 8 + 0),
                                 inc(i * 8 + 1),
                                 inc(i * 8 + 2),
                                 inc(i * 8 + 3),
                                 inc(i * 8 + 4),
                                 inc(i * 8 + 5));
            SE3f new_pose = best_poses_[i] * SE3f::exp(pose_inc);
            poses_[i] = new_pose;

            Vec2<float> exp_inc(inc(i * 8 + 6),
                                inc(i * 8 + 7));
            Vec2<float> new_exp = best_exps_[i] + exp_inc;
            exps_[i] = new_exp;
        }

        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i]->global_pose() = kframe.local_pose_to_global(poses_[i]);
            frames[i]->local_exposure() = exps_[i];
        }
    }

    void update_params(std::span<Frame *const> frames, KeyFrame &kframe)
    {
        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i]->global_pose() = kframe.local_pose_to_global(best_poses_[i]);
            frames[i]->local_exposure() = best_exps_[i];
        }
    }

    void update_best_params()
    {
        best_poses_ = poses_;
        best_exps_ = exps_;
    }

    void restore_best_params(std::span<Frame *const> frames, KeyFrame &kframe)
    {
        poses_ = best_poses_;
        exps_ = best_exps_;

        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i]->global_pose() = kframe.local_pose_to_global(best_poses_[i]);
            frames[i]->local_exposure() = best_exps_[i];
        }
    }

    Error<float> compute_error(std::span<const Frame *const> frames, const KeyFrame &kframe, const Cameraf &cam, int in_lvl, int out_lvl)
    {
        Error<float> total;
        for (std::size_t frame_idx = 0; frame_idx < frames.size(); frame_idx++)
        {
            residualrenderer_.Render(kframe.mesh(), poses_[frame_idx], exps_[frame_idx], cam, in_lvl, out_lvl, kframe.image(), frames[frame_idx]->image(), res_texture_);
            residualreducer_.reduce(out_lvl, res_texture_, total);
        }
        return total;
    }

    void compute_problem(std::span<const Frame *const> frames, const KeyFrame &kframe, const Cameraf &cam, int in_lvl, int out_lvl, DenseLinearProblem<float, 8> &total)
    {
        for (std::size_t frame_idx = 0; frame_idx < frames.size(); frame_idx++)
        {
            jposedepthrenderer_.Render(kframe.mesh(),
                                       poses_[frame_idx],
                                       exps_[frame_idx],
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
                                       jexp_texture_,
                                       res_texture_,
                                       total);
        }
    }

private:
    JPoseExpRenderer jposedepthrenderer_;
    HGPoseExpReducerCPU<float> hgposedepthreducer_;
    ResidualRenderer residualrenderer_;
    ResidualReducerCPU<float> residualreducer_;

    Texture<Vec3f> jtra_texture_;
    Texture<Vec3f> jrot_texture_;
    Texture<Vec3f> jexp_texture_;
    Texture<float> res_texture_;

    std::vector<SE3f> init_poses_;
    std::vector<Vec2f> init_exps_;

    std::vector<SE3f> best_poses_;
    std::vector<Vec2f> best_exps_;

    std::vector<SE3f> poses_;
    std::vector<Vec2f> exps_;

    int numParams_;
};
