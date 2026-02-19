#pragma once

#include <iostream>
#include <vector>
#include "params.h"
#include "mpdr/common/types.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "common/DenseLinearProblem.h"
#include "common/depthParam.h"
#include "common/reducer.h"
#include "optimizers/baseOptimizer.h"
#include "utils/tictoc.h"

template <typename T>
T area3d(const Vec3<T> &v0, const Vec3<T> &v1, const Vec3<T> &v2)
{
    Vec3<T> u = v1 - v0;
    Vec3<T> v = v2 - v0;
    return fabs(u.cross(v))/2.0;
}

class VertexOptimizer : public BaseOptimizer
{
public:
    VertexOptimizer(int w, int h, bool _printLog = false);

    void init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl);
    void step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl);

private:
    void compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int frame_id, int num_frames, int num_vertices, int in_lvl, int out_lvl, DenseLinearProblemx &total);

    JVertexExpRenderer jvertexrenderer_;
    HGVertexReducerCPU hgmapreducer_;

    Texture<Vec3<float>> jv0_texture_;
    Texture<Vec3<float>> jv1_texture_;
    Texture<Vec3<float>> jv2_texture_;

    Texture<Vec3<float>> jexp_texture_;
    Texture<Vec3<PidType>> pids_texture_;

    Matxf invCovariance_;

    std::vector<Vec3<float>> init_vertices_;
    std::vector<Vec3<int>> init_triangles_;
    Vecxf init_params_;
    float init_error_;

    std::vector<Vec3<float>> vertices_;
    std::vector<Vec3<int>> triangles_;
    Vecxf params_;
    float error_;

    Matxf init_invcovariance_;
    Matxf init_invcovariancesqrt_;

    DenseLinearProblemx problem_;
    Solverx<float> solver_;

    bool printLog_;
    tic_toc timer_;
};
