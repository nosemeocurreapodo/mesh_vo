#pragma once

#include <iostream>
#include "params.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "common/DenseLinearProblem.h"
#include "optimizers/baseOptimizer.h"
#include "common/reducer.h"

class PoseOptimizer : public BaseOptimizer
{
public:
    PoseOptimizer(int w, int h, bool print_log = false);

    void init(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl);
    void step(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl);

private:
    DenseLinearProblem<6> compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl);

    JPoseRenderer jposerenderer_;
    HGPoseReducerCPU hgposereducer_;

    Texture<Vec3f> jtra_texture_;
    Texture<Vec3f> jrot_texture_;

    Mat6f inv_covariance_;

    Vec6f init_pose_;
    Mat6f init_invcovariance_;
    Mat6f init_invcovariancesqrt_;
    float init_error_;

    bool print_log_;
};
