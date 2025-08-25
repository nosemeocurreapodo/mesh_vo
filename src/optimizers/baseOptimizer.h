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
//#include "cpu/OpenCVDebug.h"

class BaseOptimizer
{
public:
    BaseOptimizer(int width, int height);

    virtual void init(Frame &frame, KeyFrame &kframe, CameraType &cam, int lvl) = 0;
    virtual void step(Frame &frame, KeyFrame &kframe, CameraType &cam, int lvl) = 0;

    bool converged();

protected:
    //void plotDebug(keyFrame &kframe, std::vector<Frame> &frames, cameraType &cam, std::string window_name);

    //TextureCPU<imageType> image_buffer;
    //TextureCPU<float> depth_buffer;
    //TextureCPU<errorType> error_buffer;
    //TextureCPU<float> weight_buffer;

    //renderCPU renderer;
    //reduceCPU reducer;

    bool reachedConvergence;
};
