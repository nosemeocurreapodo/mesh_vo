#pragma once

#include "params.h"
#include "core/camera.h"

class PoseOptimizer : public BaseOptimizer
{
public:
    PoseOptimizer(int width, int height, bool _printLog = false);

private:
    DenseLinearProblem computeProblem(Frame &frame, KeyFrame &kframe, CameraType &cam, int lvl);

    JPoseRendererCPU jposerenderer;
    ImageRendererCPU imagerenderer;

    HGPoseReducerCPU hgposereducer;

    TextureCPU<jposeType> j_buffer;
    TextureCPU<jposeType> i_buffer;

    Mat6 invCovariance;

    Vec6 init_pose;
    Mat6 init_invcovariance;
    Mat6 init_invcovariancesqrt;
    float init_error;

    bool printLog;
};
