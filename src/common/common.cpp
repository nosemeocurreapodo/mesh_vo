#include "common/common.h"

Eigen::Vector3f fromRayIdepthToVertex(Eigen::Vector3f rayIdepth)
{
    Eigen::Vector3f kf_tri_ray;
    kf_tri_ray(0) = rayIdepth[v_index][0];
    kf_tri_ray(1) = rayIdepth[v_index][1];
    kf_tri_ray(2) = 1.0;
    float kf_tri_idepth = rayIdepth[v_index][2];

    return kf_tri_ray / kf_tri_idepth;
}