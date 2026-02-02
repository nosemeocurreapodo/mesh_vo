#pragma once

#include <iostream>
#include <vector>
#include "params.h"
#include "core/types.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "common/DenseLinearProblem.h"
#include "common/depthParam.h"
#include "common/reducer.h"
#include "optimizers/baseOptimizer.h"
#include "utils/tictoc.h"

class DepthOptimizer : public BaseOptimizer
{
public:
    DepthOptimizer(int w, int h, bool _printLog = false);

    void init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl);
    void step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl);

private:
    void compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int frame_id, int num_frames, int num_vertices, int in_lvl, int out_lvl, DenseLinearProblemx &total);

    JDepthExpRenderer jmaprenderer_;
    HGDepthReducerCPU hgmapreducer_;

    Texture<Vec3<float>> jdepth_texture_;
    Texture<Vec3<float>> jexp_texture_;
    Texture<Vec3<PidType>> pids_texture_;

    Matxf invCovariance_;

    std::vector<float> init_depths_;
    Vecxf init_params_;
    float init_error_;

    std::vector<float> depths_;
    Vecxf params_;
    float error_;

    std::vector<Vec3<int>> triangles_;
    std::vector<Vec2<int>> edges_;

    Matxf init_invcovariance_;
    Matxf init_invcovariancesqrt_;

    DenseLinearProblemx problem_;
    Solverx<float> solver_;

    bool printLog_;
    tic_toc timer_;
};
