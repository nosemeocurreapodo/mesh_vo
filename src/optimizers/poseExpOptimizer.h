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
                                              Matx<float>,
                                              Vecx<float>,
                                              DenseLinearProblem<8>,
                                              Solver<float, 8>>
{
public:
    using Base = BaseOptimizer<PoseExpOptimizer,
                               Matx<float>,
                               Vecx<float>,
                               DenseLinearProblem<8>,
                               Solver<float, 8>>;
    PoseExpOptimizer(bool printlog = false)
        : Base(printlog),
          jtra_texture_(1, 1, Vec3<float>(0.0, 0.0, 0.0)),
          jrot_texture_(1, 1, Vec3<float>(0.0, 0.0, 0.0)),
          jexp_texture_(1, 1, Vec3<float>(0.0, 0.0, 0.0))
    {
    }

    int numParams()
    {
        return numParams_;
    }

    void init(const std::vector<Frame> &frames, const KeyFrame &kframe, DenseLinearProblem<8> &problem, Solver<float, 8> &solver)
    {
        best_poses_.clear();
        best_exps_.clear();
        for (int i = 0; i < frames.size(); i++)
        {
            best_poses_.push_back(frames[i].local_pose());
            best_exps_.push_back(frames[i].local_exposure());
        }
        numParams_ = 8 * frames.size();
    }

    float regu_error() const
    {
        return 0;//regu_depth(depths_, edges_);
    }

    void regu_jacobian(DenseLinearProblemx &problem)
    {
        //regu_depth_jacobian(depths_, edges_, problem);
    }

    void update_params(std::vector<Frame> &frames, KeyFrame &kframe, const Vecx<float> &inc)
    {
        for (size_t i = 0; i < frames.size(); i++)
        {
            Vec6<float> pose_inc(inc(i * 8 + 0),
                                 inc(i * 8 + 1),
                                 inc(i * 8 + 2),
                                 inc(i * 8 + 3),
                                 inc(i * 8 + 4),
                                 inc( i * 8 + 5));
            SE3f new_pose = best_poses_[i] * SE3f::exp(pose_inc);
            poses_.push_back(new_pose);

            Vec2<float> exp_inc(inc(i * 8 + 6),
                                inc(i * 8 + 7));
            Vec2<float> new_exp = best_exps_[i] + exp_inc;
            exps_.push_back(new_exp);
        }

        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i].local_pose() = poses_[i];
            frames[i].local_exposure() = exps_[i];
        }
    }

    void update_best_params()
    {
        best_poses_ = poses_;
        best_exps_ = exps_;
    }

    void restore_best_params(std::vector<Frame> &frames, KeyFrame &kframe)
    {
        poses_ = best_poses_;
        exps_ = best_exps_;

        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i].local_pose() = best_poses_[i];
            frames[i].local_exposure() = best_exps_[i];
        }
    }

    void compute_problem(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl, DenseLinearProblem<8> &total)
    {
        for (std::size_t frame_idx = 0; frame_idx < frames.size(); frame_idx++)
        {
            jposedepthrenderer_.Render(kframe.mesh(),
                                       frames[frame_idx].local_pose(),
                                       frames[frame_idx].local_exposure(),
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
                                       jexp_texture_,
                                       image_texture_,
                                       frames[frame_idx].image(),
                                       total);
        }
    }

private:
    JPoseExpRenderer jposedepthrenderer_;
    HGPoseExpReducerCPU hgposedepthreducer_;

    Texture<Vec3<float>> jtra_texture_;
    Texture<Vec3<float>> jrot_texture_;
    Texture<Vec3<float>> jexp_texture_;

    std::vector<SE3f> best_poses_;
    std::vector<Vec2f> best_exps_;

    std::vector<SE3f> poses_;
    std::vector<Vec2f> exps_;

    int numParams_;
};
