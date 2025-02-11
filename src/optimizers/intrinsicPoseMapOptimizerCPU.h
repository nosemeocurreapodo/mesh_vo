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

class intrinsicPoseMapOptimizerCPU : public baseOptimizerCPU
{
public:
    intrinsicPoseMapOptimizerCPU(camera &_cam);

    void optimize(std::vector<frameCPU> &frames, keyFrameCPU &kframe);

private:
    DenseLinearProblem computeProblem(frameCPU &frame, keyFrameCPU &kframe, int frameId, int numFrames, int lvl);

    dataMipMapCPU<vec4f> jintrinsic_buffer;
    dataMipMapCPU<vec6f> jpose_buffer;
    dataMipMapCPU<jmapType> jmap_buffer;
    dataMipMapCPU<idsType> pId_buffer;

    matxf invCovariance;
};
