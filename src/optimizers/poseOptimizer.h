#pragma once

#include "params.h"
#include "core/camera.h"
#include "core/types.h"
#include "common/types.h"
#include "optimizers/baseOptimizer.h"
#include "backends/cpu/renderercpu.h"
#include "common/reducer.h"

class PoseOptimizer : public BaseOptimizer
{
public:
    PoseOptimizer(int width, int height, bool _printLog = false);

private:
    DenseLinearProblem computeProblem(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl);

    JposeRendererCPU jposerenderer_;
    ImageRendererCPU imagerenderer_;

    ErrorReducerCPU errprreducer_;
    HGPoseReducerCPU hgposereducer_;

    TextureCPU<JposeType> j_buffer_;
    TextureCPU<ImageType> i_buffer_;

    Mat6 invCovariance_;

    Vec6 init_pose_;
    Mat6 init_invcovariance_;
    Mat6 init_invcovariancesqrt_;
    float init_error_;

    bool printLog_;
};
