#pragma once

#include "params.h"
#include "core/camera.h"
#include "common/types.h"
#include "common/Error.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/keyFrameCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/reduceCPU.h"
#include "common/DenseLinearProblem.h"
#include "cpu/OpenCVDebug.h"

class baseOptimizerCPU
{
public:
    baseOptimizerCPU(int width, int height);

    virtual void init(FrameCPU &frame, KeyFrameCPU &kframe, CameraType &cam, int lvl) = 0;
    virtual void step(FrameCPU &frame, KeyFrameCPU &kframe, CameraType &cam, int lvl) = 0;

    bool converged();

protected:
    void plotDebug(keyFrameCPU &kframe, std::vector<frameCPU> &frames, cameraType &cam, std::string window_name);

    TextureCPU<imageType> image_buffer;
    TextureCPU<float> depth_buffer;
    TextureCPU<errorType> error_buffer;
    TextureCPU<float> weight_buffer;

    //renderCPU renderer;
    //reduceCPU reducer;

    bool reachedConvergence;
};
