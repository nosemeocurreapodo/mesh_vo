#pragma once

#include <iostream>
#include "params.h"
#include "core/camera.h"
#include "core/types.h"
#include "common/types.h"
#include "optimizers/baseOptimizer.h"
#include "common/reducer.h"
#include "backends/cpu/meshcpu.h"

class MapOptimizer : public BaseOptimizer
{
public:
    MapOptimizer(int w, int h, bool _printLog = false);

    void init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int lvl);
    void step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int lvl);

private:
    DenseLinearProblem compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl);

    float error_regu_(const MeshCPU &mesh)
    {
        float regu_error = 0.0;
        auto pos_map = mesh.MapReadPositions();
        auto ids_map = mesh.MapReadIndices();
        for (size_t i = 0; i < ids_map.size(); i += 3)
        {
            Vec3i id(ids_map[i + 0],
                     ids_map[i + 1],
                     ids_map[i + 2]);
            Vec3 depth(pos_map[id(0) * 3 + 2],
                       pos_map[id(1) * 3 + 2],
                       pos_map[id(2) * 3 + 2]);
            float r1 = depth(0) - depth(1);
            float r2 = depth(0) - depth(2);
            float r3 = depth(1) - depth(2);
            regu_error += r1 * r1 + r2 * r2 + r3 * r3;
        }
        return regu_error / mesh.vertex_count();
    }

    JMapRendererCPU jmaprenderer_;
    HGMapReducerCPU hgmapreducer_;

    TextureCPU<Vec3> jmap_texture_;
    TextureCPU<Vec3> pids_texture_;

    Matx invCovariance;

    Vecx init_params;
    Matx init_invcovariance;
    Matx init_invcovariancesqrt;
    float init_error;

    bool printLog;
};
