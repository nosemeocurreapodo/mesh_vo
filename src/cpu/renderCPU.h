#pragma once

#include <Eigen/Core>

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGMapped.h"
#include "common/Error.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/MeshCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/IndexThreadReduce.h"
#include "params.h"

class renderCPU
{
public:
    renderCPU()
    {

    }

    void renderIdepth(MeshCPU &mesh, camera &cam, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderImage(MeshCPU &mesh, camera &cam, dataCPU<float> &image, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderDebug(MeshCPU &mesh, camera &cam, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    //void renderInvVar(MeshCPU &mesh, camera &cam,  frameCPU &frame, dataCPU<float> &buffer, int lvl);
    void renderError(MeshCPU &mesh, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<float> &buffer, int lvl);
};
