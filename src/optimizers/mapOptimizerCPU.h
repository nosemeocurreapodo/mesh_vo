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
#include "optimizers/baseOptimizerCPU.h"
#include "cpu/OpenCVDebug.h"

class mapOptimizerCPU : public baseOptimizerCPU
{
public:
    mapOptimizerCPU(int width, int height, bool _printLog = false);
    
    void init(std::vector<frameCPU> &frames, keyFrameCPU &kframe, cameraType &cam, int lvl);
    void step(std::vector<frameCPU> &frames, keyFrameCPU &kframe, cameraType &cam, int lvl);
    std::vector<dataCPU<float>> getDebugData(std::vector<frameCPU> &frames, keyFrameCPU &kframe, cameraType &cam, int lvl);

private:
    DenseLinearProblem computeProblem(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl);

    dataMipMapCPU<jmapType> jmap_buffer;
    dataMipMapCPU<idsType> pId_buffer;

    matxf invCovariance;

    vecxf init_params;
    matxf init_invcovariance;
    matxf init_invcovariancesqrt;
    float init_error;

    bool printLog;
};
