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

class PoseDepthOptimizer : public BaseOptimizer
{
public:
    PoseDepthOptimizer(int w, int h, bool _printLog = false);

    void init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl);
    void step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl);

private:
    void compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int frame_id, int num_frames, int num_vertices, int in_lvl, int out_lvl, DenseLinearProblemx &problem);

    JPoseExpDepthRenderer jposemaprenderer_;
    HGPoseDepthReducerCPU hgposemapreducer_;

    Texture<Vec3<float>> jtra_texture_;
    Texture<Vec3<float>> jrot_texture_;
    Texture<Vec3<float>> jexp_texture_;
    Texture<Vec3<float>> jdepth_texture_;
    Texture<Vec3<PidType>> pids_texture_;

    Matxf invCovariance;

    std::vector<float> init_depths_;
    std::vector<SE3f> init_poses_;
    float init_error_;

    std::vector<float> depths_;
    std::vector<SE3f> poses_;
    float error_;

    std::vector<Vec3<int>> triangles_;
    std::vector<Vec2<int>> edges_;

    Matxf init_invcovariance_;
    Matxf init_invcovariancesqrt_;

    DenseLinearProblemx problem_;
    Solverx<float> solver_;

    bool printLog_;
};
