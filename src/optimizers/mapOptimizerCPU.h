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
    mapOptimizerCPU(int width, int height);
    void optimize(std::vector<frameCPU> &frames, keyFrameCPU &kframe, camera &cam);

private:
    DenseLinearProblem computeProblem(frameCPU &frame, keyFrameCPU &kframe, camera &cam, int lvl);

    dataMipMapCPU<jmapType> j_buffer;
    dataMipMapCPU<idsType> pId_buffer;

    matxf invCovariance;
};
