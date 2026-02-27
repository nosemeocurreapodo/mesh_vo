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
                                              DenseLinearProblem<6>,
                                              Solver<float, 6>>
{
public:
    using Base = BaseOptimizer<PoseOptimizer,
                               Mat6<float>,
                               Vec6<float>,
                               DenseLinearProblem<6>,
                               Solver<float, 6>>;
    PoseOptimizer(int w, int h,bool printlog = false)
        : Base(w, h, printlog),
          jtra_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          jrot_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
          jexp_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0))
    {
    }

    int numParams()
    {
        return numParams_;
    }

    void reset(std::span<const Frame* const> frames, const KeyFrame &kframe, DenseLinearProblem<6> &problem, Solver<float, 6> &solver)
    {
        best_poses_.clear();
        for (int i = 0; i < frames.size(); i++)
        {
            best_poses_.push_back(frames[i]->local_pose());
        }
        numParams_ = 6 * frames.size();
    }

    float regu_error() const
    {
        return 0;//regu_depth(depths_, edges_);
    }

    void regu_jacobian(DenseLinearProblem<6> &problem) const
    {
        //regu_depth_jacobian(depths_, edges_, problem);
    }

    void update_params(std::span<Frame* const> frames, KeyFrame &kframe, const Vec6<float> &inc)
    {
        poses_.clear();
        
        for (size_t i = 0; i < frames.size(); i++)
        {
            Vec6<float> pose_inc(inc(i * 6 + 0),
                                 inc(i * 6 + 1),
                                 inc(i * 6 + 2),
                                 inc(i * 6 + 3),
                                 inc(i * 6 + 4),
                                 inc( i * 6 + 5));
            SE3f new_pose = best_poses_[i] * SE3f::exp(pose_inc);
            poses_.push_back(new_pose);
        }

        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i]->local_pose() = poses_[i];
        }
    }

    void update_best_params()
    {
        best_poses_ = poses_;
    }

    void restore_best_params(std::span<Frame* const> frames, KeyFrame &kframe)
    {
        poses_ = best_poses_;

        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i]->local_pose() = best_poses_[i];
        }
    }

    void compute_problem(std::span<const Frame* const> frames, const KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl, DenseLinearProblem<6> &total)
    {
        for (std::size_t frame_idx = 0; frame_idx < frames.size(); frame_idx++)
        {
            jposedepthrenderer_.Render(kframe.mesh(),
                                       frames[frame_idx]->local_pose(),
                                       frames[frame_idx]->local_exposure(),
                                       cam,
                                       in_lvl, out_lvl,
                                       kframe.image(),
                                       kframe.didxy(),
                                       image_texture_,
                                       jtra_texture_,
                                       jrot_texture_,
                                       jexp_texture_);
            hgposedepthreducer_.reduce(out_lvl,
                                       jtra_texture_,
                                       jrot_texture_,
                                       image_texture_,
                                       frames[frame_idx]->image(),
                                       total);
        }
    }

private:
    JPoseExpRenderer jposedepthrenderer_;
    HGPoseReducerCPU hgposedepthreducer_;

    Texture<Vec3<float>> jtra_texture_;
    Texture<Vec3<float>> jrot_texture_;
    Texture<Vec3<float>> jexp_texture_;

    std::vector<SE3f> best_poses_;

    std::vector<SE3f> poses_;

    int numParams_;
};
