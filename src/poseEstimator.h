#pragma once

#include "common/types.h"
#include "optimizers/poseOptimizer.h"
#include "optimizers/poseExpOptimizer.h"

class PoseEstimator
{
public:
    PoseEstimator(int w, int h, bool log)
        : optimizer(w, h, log)
    {
        last_global_pose = SE3d();
        last_local_exp = Vec2f(0.0, 0.0);
        last_pose_lambda = Mat6f::Zero();
        last_exp_lambda = Vec2f(0.0, 0.0);
    }

    void init_global_pose(SE3d &global_pose)
    {
        last_global_pose = global_pose;
    }

    void guess(Frame &frame, KeyFrame &kframe)
    {
        SE3d guess_global_pose = last_global_move * last_global_pose;
        frame.global_pose() = guess_global_pose;
        frame.local_exposure() = last_local_exp;
        frame.pose_lambda() = last_pose_lambda; // Mat6f::Zero();
        frame.exp_lambda() = last_exp_lambda;   // Vec2f(0.0, 0.0);
    }

    void estimate(Frame &frame, KeyFrame &kframe, Cameraf &cam)
    {
        for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_fin_lvl; lvl--)
        {
            optimizer.init(&frame, kframe, cam, lvl, lvl);
            while (true)
            {
                optimizer.step(&frame, kframe, cam, lvl, lvl);
                if (optimizer.converged())
                    break;
            }
            optimizer.update(&frame, kframe, cam);
        }
        SE3d new_global_pose = frame.global_pose();
        last_global_move = new_global_pose * last_global_pose.inverse();
        last_global_pose = new_global_pose;
        last_local_exp = frame.local_exposure();
        last_pose_lambda = frame.pose_lambda();
        last_exp_lambda = frame.exp_lambda();
    }

private:
    PoseOptimizer optimizer;
    SE3d last_global_pose;
    SE3d last_global_move;
    Vec2f last_local_exp;
    Mat6f last_pose_lambda;
    Vec2f last_exp_lambda;
};