#pragma once

#include "params.h"
#include "core/camera.h"
#include "core/types.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "backends/cpu/renderercpu.h"
#include "common/reducer.h"
#include "common/DenseLinearProblem.h"
// #include "cpu/OpenCVDebug.h"

class BaseOptimizer
{
public:
    BaseOptimizer(int w, int h) : r_texture_(w, h, 0.0)
    {
    }

    // virtual void init(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl) = 0;
    // virtual void step(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl) = 0;

    bool converged()
    {
        return reached_convergence_;
    }

protected:
    Error computeError(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl)
    {
        //imagerenderer_.Render(kframe.mesh(), frame.local_pose() * kframe.frame().local_pose().inverse(), cam, lvl, lvl, kframe.frame().image(), e_texture_);
        //return errorreducer_.reduce(lvl, frame.image(), e_texture_);

        residualrenderer_.Render(kframe.mesh(), frame.local_pose(), cam, lvl, lvl, kframe.frame().image(), frame.image(), r_texture_);
        return residualreducer_.reduce(lvl, r_texture_);
    }

    //ImageRendererCPU imagerenderer_;
    //ErrorReducerCPU errorreducer_;

    ResidualRendererCPU residualrenderer_;
    ResidualReducerCPU residualreducer_;

    TextureCPU<float> r_texture_;

    bool reached_convergence_;
};
