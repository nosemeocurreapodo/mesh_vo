#pragma once

#include "params.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "common/reducer.h"
#include "common/DenseLinearProblem.h"
#include "mpdr/backends/cpu/renderercpu.h"
// #include "cpu/OpenCVDebug.h"

class BaseOptimizer
{
public:
    BaseOptimizer(int w, int h) : image_texture_(w, h, 0.0)
    {
    }

    // virtual void init(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl) = 0;
    // virtual void step(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl) = 0;

    bool converged()
    {
        return reached_convergence_;
    }

protected:
    void compute_error_(Frame &frame, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl, Error &total)
    {
        // imagerenderer_.Render(kframe.mesh(), frame.local_pose() * kframe.frame().local_pose().inverse(), cam, lvl, lvl, kframe.frame().image(), e_texture_);
        // return errorreducer_.reduce(lvl, frame.image(), e_texture_);

        imagerenderer_.Render(kframe.mesh(), frame.local_pose(), frame.local_exposure(), cam, in_lvl, out_lvl, kframe.image(), image_texture_);
        residualreducer_.reduce(out_lvl, image_texture_, frame.image(), total);
    }

    ImageRenderer imagerenderer_;
    ResidualReducerCPU residualreducer_;

    Texture<ImageType> image_texture_;

    bool reached_convergence_;
};
