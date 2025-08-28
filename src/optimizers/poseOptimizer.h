#pragma once

#include <iostream>
#include "params.h"
#include "core/camera.h"
#include "core/types.h"
#include "common/types.h"
#include "optimizers/baseOptimizer.h"
#include "common/reducer.h"

class PoseOptimizer : public BaseOptimizer
{
public:
    PoseOptimizer(int w, int h, bool print_log = false);

    void init(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl);
    void step(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl);

private:
    DenseLinearProblem computeProblem(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl);

    JposeRendererCPU2 jposerenderer_;
    HGPoseReducerCPU2 hgposereducer_;

    TextureCPU<Vec3> jtra_texture_;
    TextureCPU<Vec3> jrot_texture_;

    Mat6 inv_covariance_;

    Vec6 init_pose_;
    Mat6 init_invcovariance_;
    Mat6 init_invcovariancesqrt_;
    float init_error_;

    bool print_log_;
};
