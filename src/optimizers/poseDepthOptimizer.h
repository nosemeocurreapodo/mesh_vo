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

class PoseDepthOptimizer : public BaseOptimizer<PoseDepthOptimizer,
                                                Matx<float>,
                                                Vecx<float>,
                                                DenseLinearProblemx,
                                                Solverx<float>>
{
public:
    using Base = BaseOptimizer<PoseDepthOptimizer,
                               Matx<float>,
                               Vecx<float>,
                               DenseLinearProblemx,
                               Solverx<float>>;
    PoseDepthOptimizer(bool printlog = false)
        : Base(printlog),
          jtra_texture_(1, 1, Vec3<float>(0.0, 0.0, 0.0)),
          jrot_texture_(1, 1, Vec3<float>(0.0, 0.0, 0.0)),
          jdepth_texture_(1, 1, Vec3<float>(0.0, 0.0, 0.0)),
          jexp_texture_(1, 1, Vec3<float>(0.0, 0.0, 0.0)),
          pids_texture_(1, 1, Vec3<PidType>(-1, -1, -1))
    {
    }

    int numParams()
    {
        return numParams_;
    }

    void init(const std::vector<Frame> &frames, const KeyFrame &kframe, DenseLinearProblemx &problem, Solverx<float> &solver)
    {
        best_depths_ = get_depths(kframe.mesh());
        best_poses_.clear();
        for (int i = 0; i < frames.size(); i++)
            best_poses_.push_back(frames[i].local_pose());
        // triangles_ = get_indices(kframe.mesh());
        edges_ = get_edges(kframe.mesh());
        numDepths_ = kframe.mesh().vertex_count();
        numParams_ = numDepths_ + 6 * frames.size();

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

    void update_params(std::vector<Frame> &frames, KeyFrame &kframe, const Vecx<float> &inc)
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

        for (size_t i = 0; i < frames.size(); i++)
        {
            Vec6<float> pose_inc(inc(numDepths_ + i * 6 + 0),
                                 inc(numDepths_ + i * 6 + 1),
                                 inc(numDepths_ + i * 6 + 2),
                                 inc(numDepths_ + i * 6 + 3),
                                 inc(numDepths_ + i * 6 + 4),
                                 inc(numDepths_ + i * 6 + 5));
            SE3f new_pose = best_poses_[i] * SE3f::exp(pose_inc);
            poses_.push_back(new_pose);
        }

        set_depths(kframe.mesh(), depths_);

        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i].local_pose() = poses_[i];
        }
    }

    void update_best_params()
    {
        best_depths_ = depths_;
        best_poses_ = poses_;
    }

    void restore_best_params(std::vector<Frame> &frames, KeyFrame &kframe)
    {
        depths_ = best_depths_;
        poses_ = best_poses_;

        set_depths(kframe.mesh(), best_depths_);

        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i].local_pose() = best_poses_[i];
        }
    }

    void compute_problem(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl, DenseLinearProblemx &total)
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
                                       jexp_texture_,
                                       jdepth_texture_,
                                       pids_texture_);
            hgposedepthreducer_.reduce(out_lvl,
                                       frame_idx, frames.size(), numDepths_,
                                       jtra_texture_,
                                       jrot_texture_,
                                       jdepth_texture_,
                                       pids_texture_,
                                       image_texture_,
                                       frames[frame_idx].image(),
                                       kframe.mesh(),
                                       total);
        }
    }

private:
    JPoseExpDepthRenderer jposedepthrenderer_;
    HGPoseDepthReducerCPU hgposedepthreducer_;

    Texture<Vec3<float>> jtra_texture_;
    Texture<Vec3<float>> jrot_texture_;
    Texture<Vec3<float>> jdepth_texture_;
    Texture<Vec3<float>> jexp_texture_;
    Texture<Vec3<PidType>> pids_texture_;

    std::vector<float> best_depths_;
    std::vector<SE3f> best_poses_;

    std::vector<float> depths_;
    std::vector<SE3f> poses_;

    // std::vector<Vec3<int>> triangles_;
    std::vector<Vec2<int>> edges_;

    int numDepths_;
    int numParams_;
};
