#pragma once

#include "common/types.h"
#include "optimizers/depthOptimizer.h"

class DepthEstimator
{
public:
    DepthEstimator(int w, int h, bool log)
        : optimizer(w, h, log),
          depth_texture(w, h, -1.0)
    {
    }

    // void guess(Frame &frame)
    //{
    //     frame.local_pose() = last_local_move * last_local_pose;
    //     frame.local_exposure() = last_local_exp;
    // }

    void estimate(std::span<Frame* const> frames, KeyFrame &kframe, Camera &cam)
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
        }
    }

    void changeKeyframe(Frame &new_kframe, std::span<Frame* const> old_frames, KeyFrame &old_kframe, const Camera &cam)
    {
        SE3f global_pose = old_kframe.localPoseToGlobal(new_kframe.local_pose());
        float global_scale = old_kframe.getGlobalScale();

        depth_renderer.Render(old_kframe.mesh(),
                              new_kframe.local_pose(),
                              cam,
                              0,
                              depth_texture);

        Mesh mesh = CreateMesh<Mesh>(depth_texture.MapRead(0).data(),
                                     cam,
                                     depth_texture.width(0),
                                     depth_texture.height(0),
                                     mesh_vo::mesh_width);
        // CreateFlatMesh(mesh_vo::mapping_mean_depth * 0.5, mesh_vo::mapping_mean_depth * 1.5, cam_, mesh_vo::mesh_width, ver_buff_, idx_buff_, true, true, true);
        //   CreateSphereMesh(mesh_vo::mapping_mean_depth, cam_, mesh_vo::mesh_width, pos_buff_, tex_buff_, wei_buff_, idx_buff_);

        old_kframe = KeyFrame(new_kframe,
                              std::move(mesh),
                              global_pose,
                              global_scale);

        SE3f reference_pose = new_kframe.local_pose().inverse();

        for (Frame* f: old_frames)
        {
            f->local_pose() = f->local_pose() * reference_pose;
            // frames[k].local_pose() = kframe->globalPoseToLocal(gt_global_poses[k]);
            f->keyframe_id() = new_kframe.id();
        }
    }

private:
    DepthOptimizer optimizer;
    DepthRenderer depth_renderer;
    Texture<float> depth_texture;
};