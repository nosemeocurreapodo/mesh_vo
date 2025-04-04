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

class poseVelOptimizerCPU : public baseOptimizerCPU
{
public:
    poseVelOptimizerCPU(int width, int height);

    void init(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl);
    void step(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl);
    std::vector<dataCPU<float>> getDebugData(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl);

private:
    DenseLinearProblem computeProblem(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl);

    dataMipMapCPU<jposeType> jpose_buffer;
    dataMipMapCPU<jvelType> jvel_buffer;
    matxf invCovariance;

    vecxf init_pose;
    matxf init_invcovariance;
    matxf init_invcovariancesqrt;
    float init_error;
};
