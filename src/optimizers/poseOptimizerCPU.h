#pragma once

#include "params.h"
#include "common/camera.h"
#include "common/types.h"
#include "common/Error.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/reduceCPU.h"
#include "common/DenseLinearProblem.h"
#include "optimizers/baseOptimizerCPU.h"
#include "cpu/OpenCVDebug.h"

class poseOptimizerCPU : public baseOptimizerCPU
{
public:
    poseOptimizerCPU(int width, int height);

    void optimize(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam);

private:
    DenseLinearProblem computeProblem(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl);

    dataMipMapCPU<vec6f> j_buffer;
    mat6f invCovariance;
};
