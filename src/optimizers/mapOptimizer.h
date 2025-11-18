#pragma once

#include <iostream>
#include "params.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "common/DenseLinearProblem.h"
#include "common/depthParam.h"
#include "common/reducer.h"
#include "optimizers/baseOptimizer.h"

class MapOptimizer : public BaseOptimizer
{
public:
    MapOptimizer(int w, int h, bool _printLog = false);

    void init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int lvl);
    void step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int lvl);

private:
    DenseLinearProblemx compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl);

    float error_regu_(const Mesh &mesh)
    {
        float regu_error = 0.0;
        auto pos_map = mesh.MapReadPositions();
        auto ids_map = mesh.MapReadIndices();
        for (size_t i = 0; i < ids_map.size(); i += 3)
        {
            Vec3i id(ids_map[i + 0],
                     ids_map[i + 1],
                     ids_map[i + 2]);
            Vec3f depth(pos_map[id(0) * 3 + 2],
                        pos_map[id(1) * 3 + 2],
                        pos_map[id(2) * 3 + 2]);
            float r1 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(1));
            float r2 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(2));
            float r3 = fromDepthToParam(depth(1)) - fromDepthToParam(depth(2));
            regu_error += r1 * r1 + r2 * r2 + r3 * r3;
        }
        return regu_error;
    }

    JMapRenderer jmaprenderer_;
    HGMapReducerCPU hgmapreducer_;

    Texture<Vec3f> jmap_texture_;
    Texture<Vec3f> pids_texture_;

    Matxf invCovariance;

    Matxf init_params;
    Matxf init_invcovariance;
    Matxf init_invcovariancesqrt;
    float init_error;

    bool printLog;
};
