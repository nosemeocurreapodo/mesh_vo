#pragma once

#include <iostream>
#include "params.h"
#include "core/camera.h"
#include "core/types.h"
#include "common/types.h"
#include "optimizers/baseOptimizer.h"
#include "common/reducer.h"

class MapOptimizer : public BaseOptimizer
{
public:
    MapOptimizer(int w, int h, bool _printLog = false);
    
    void init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int lvl);
    void step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int lvl);
    //std::vector<dataCPU<float>> getDebugData(std::vector<frameCPU> &frames, keyFrameCPU &kframe, cameraType &cam, int lvl);

private:
    DenseLinearProblem computeProblem_(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl);

    JMapRendererCPU jmaprenderer_;
    HGMapReducerCPU hgmapreducer_;

    TextureCPU<Vec3> jmap_texture_;
    TextureCPU<Vec3> pids_texture_;

    Matx invCovariance;

    Vecx init_params;
    Matx init_invcovariance;
    Matx init_invcovariancesqrt;
    float init_error;

    bool printLog;
};
