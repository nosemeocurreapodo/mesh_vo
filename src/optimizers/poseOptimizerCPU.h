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

class poseOptimizerCPU : public baseOptimizerCPU
{
public:
    poseOptimizerCPU(int width, int height, bool _printLog = false);

    void init(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl);
    void step(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl);
    std::vector<dataCPU<float>> getDebugData(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl);

private:
    DenseLinearProblem computeProblem(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl);

    dataMipMapCPU<jposeType> j_buffer;
    mat6f invCovariance;

    vec6f init_pose;
    mat6f init_invcovariance;
    mat6f init_invcovariancesqrt;
    float init_error;

    bool printLog;
};
