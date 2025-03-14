#pragma once

#include "params.h"
#include "common/camera.h"
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
    bool converged();

protected:
    void plotDebug(keyFrameCPU &kframe, std::vector<frameCPU> &frames, cameraType &cam, std::string window_name);
    Error computeError(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl);

    dataMipMapCPU<imageType> image_buffer;
    dataMipMapCPU<float> depth_buffer;
    dataMipMapCPU<errorType> error_buffer;
    dataMipMapCPU<float> weight_buffer;

    renderCPU renderer;
    reduceCPU reducer;

    bool reachedConvergence;
};
