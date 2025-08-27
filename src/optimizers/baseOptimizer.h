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
    BaseOptimizer(int w, int h) : image_buffer_(w, h, 0.0)
    {
    }

    //virtual void init(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl) = 0;
    //virtual void step(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl) = 0;

    bool converged()
    {
        return reached_convergence_;
    }

protected:
    Error computeError(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl)
    {
        imagerenderer_.Render(kframe.mesh(), frame.local_pose() * kframe.frame().local_pose().inverse(), cam, kframe.frame().image(), image_buffer_, lvl, lvl);
        return errorreducer_.reduce(lvl, frame.image(), image_buffer_);
    }

    // void plotDebug(keyFrame &kframe, std::vector<Frame> &frames, cameraType &cam, std::string window_name);

    // TextureCPU<imageType> image_buffer;
    // TextureCPU<float> depth_buffer;
    // TextureCPU<errorType> error_buffer;
    // TextureCPU<float> weight_buffer;

    // renderCPU renderer;
    // reduceCPU reducer;

    ImageRendererCPU imagerenderer_;
    ErrorReducerCPU errorreducer_;

    TextureCPU<float> image_buffer_;

    bool reached_convergence_;
};
