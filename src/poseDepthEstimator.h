#pragma once

#include "common/types.h"
#include "optimizers/poseDepthOptimizer.h"
#include "optimizers/poseExpDepthOptimizer.h"

class PoseDepthEstimator
{
public:
    PoseDepthEstimator(int w, int h, bool log)
        : optimizer(w, h, log)
    {
    }

    // void guess(Frame &frame)
    //{
    //     frame.local_pose() = last_local_move * last_local_pose;
    //     frame.local_exposure() = last_local_exp;
    // }

    void init(std::span<Frame *const> frames, KeyFrame &kframe, Camera &cam)
    {
        optimizer.init(frames, kframe, cam, 1, 1);
    }

    bool converged() const
    {
        return optimizer.converged();
    }

    void step(std::span<Frame *const> frames, KeyFrame &kframe, Camera &cam)
    {
        optimizer.step(frames, kframe, cam, 1, 1);
    }

    void estimate(std::span<Frame *const> frames, KeyFrame &kframe, Camera &cam)
    {
        // for (auto frame : frames)
        //     assert(frame.keyframe_id() == kframe.id());

        for (int lvl = mesh_vo::mapping_ini_lvl; lvl >= mesh_vo::mapping_fin_lvl; lvl--)
        {
            optimizer.init(frames, kframe, cam, lvl, lvl);
            while (true)
            {
                optimizer.step(frames, kframe, cam, lvl, lvl);
                if (optimizer.converged())
                    break;
            }
            // optimizer.update(frames, kframe, cam);
        }
        // float md = mean_depth(kframe.mesh());
        // kframe.scale_mesh(md / mesh_vo::mapping_mean_depth);
    }

    void update_keyframe(const Frame &new_frame, const Texture<float> &new_depth, KeyFrame &kframe, const Camera &cam)
    {
        // float md = mean_depth(kframe.mesh());
        // kframe.scale_mesh(md / mesh_vo::mapping_mean_depth);

        SE3f global_pose = new_frame.global_pose();
        float global_scale = kframe.global_scale();

        Mesh mesh = CreateMesh<Mesh>(new_depth.MapRead(0).data(),
                                     cam,
                                     new_depth.width(0),
                                     new_depth.height(0),
                                     mesh_vo::mesh_width,
                                     mesh_vo::mapping_mean_depth);

        kframe.image() = new_frame.image();
        kframe.mesh() = std::move(mesh);
        kframe.global_pose() = global_pose;
        kframe.global_scale() = global_scale;
        kframe.id() = new_frame.id();
    }

private:
    PoseExpDepthOptimizer optimizer;
};