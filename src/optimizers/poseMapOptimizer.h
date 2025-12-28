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

class PoseMapOptimizer : public BaseOptimizer
{
public:
    PoseMapOptimizer(int w, int h, bool _printLog = false);

    void init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl);
    void step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl);

private:
    DenseLinearProblemx compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int frame_id, int num_frames, int num_vertices, int in_lvl, int out_lvl);

    JPoseVelExpMapRenderer jposemaprenderer_;
    HGPoseMapReducerCPU hgposemapreducer_;

    Texture<Vec3<float>> jtra_texture_;
    Texture<Vec3<float>> jrot_texture_;
    Texture<Vec3<float>> jtravel_texture_;
    Texture<Vec3<float>> jrotvel_texture_;
    Texture<Vec3<float>> jexp_texture_;
    Texture<Vec3<float>> jmap_texture_;
    Texture<Vec3<PidType>> pids_texture_;

    Matxf invCovariance;

    std::vector<float> init_positions;
    std::vector<int> init_indices;
    std::vector<SE3f> init_poses;
    float init_error;

    std::vector<float> positions;
    std::vector<int> indices;
    std::vector<SE3f> poses;
    float error;

    Matxf init_invcovariance;
    Matxf init_invcovariancesqrt;

    bool printLog;
};
