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

    void init(Frame &frame, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl);
    void step(Frame &frame, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl);

private:
    void compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl, DenseLinearProblem<6> &problem);

    JPoseExpRenderer jposerenderer_;
    HGPoseReducerCPU hgposereducer_;

    Texture<Vec3f> jtra_texture_;
    Texture<Vec3f> jrot_texture_;
    Texture<Vec3f> jexp_texture_;

    Mat6f inv_covariance_;

    SE3f init_pose_;
    float init_error_;

    SE3f pose_;
    float error_;

    Mat6f init_invcovariance_;
    Mat6f init_invcovariancesqrt_;

    Solver<float, 6> solver_;

    bool print_log_;
};
