#pragma once

#include "common/types.h"

class PoseEstimator
{
public:

    PoseEstimator()
    {

    }

        static void optimize(Frame &frame, Keyframe &kframe, Camera &cam)
    {
        for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_ini_lvl; lvl--)
        {
            optimizer.init(frame, kframe, cam, lvl);
            while (true)
            {
                optimizer.step(frame, kframe, cam, lvl);
                if (optimizer.converged())
                    break;
            }
        }
    }

private:

        PoseOptimizer optimizer(w_, h_, true);

}