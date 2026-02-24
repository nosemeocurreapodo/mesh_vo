#pragma once

#include "common/types.h"
#include "optimizers/poseOptimizer.h"

class PoseEstimator
{
public:
    PoseEstimator(int w, int h, bool log)
        : optimizer(w, h, log)
    {
        last_local_exp = Vec2f(0.0, 0.0);
    }

    void guess(Frame &frame)
    {
        frame.local_pose() = last_local_move * last_local_pose;
        frame.local_exposure() = last_local_exp;
    }

    void update_lastpose(SE3f new_last_pose)
    {
        last_local_pose = new_last_pose;
    }

    void estimate(Frame &frame, KeyFrame &kframe, Camera &cam)
    {
        assert(frame.keyframe_id() == kframe.id());

        for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_fin_lvl; lvl--)
        {
            optimizer.init(frame, kframe, cam, lvl, lvl);
            while (true)
            {
                optimizer.step(frame, kframe, cam, lvl, lvl);
                if (optimizer.converged())
                    break;
            }
        }
        last_local_move = frame.local_pose() * last_local_pose.inverse();
        last_local_pose = frame.local_pose();
        last_local_exp = frame.local_exposure();
    }

    void changeKeyframe(const Frame &new_kframe, KeyFrame &old_kframe, const Camera &cam)
    {
        /*
        assert(frame.keyframe_id() == kframe.keyframe_id());

        SE3f ref = kframe.local_pose().inverse();
        SE3f new_pose = frame.local_pose() * ref;
        frame.local_pose() = new_pose;
        frame.keyframe_id() = kframe.id();
        frame.local_exposure() = Vec2f(0.0, 0.0);

        last_local_pose = SE3f();
        last_local_move = SE3f();
        last_local_exp = Vec2f(0.0, 0.0);
        */
    }

private:
    PoseOptimizer optimizer;
    SE3f last_local_pose;
    SE3f last_local_move;
    Vec2f last_local_exp;
};