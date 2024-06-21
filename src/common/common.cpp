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

Eigen::Vector3f fromVertexToRayIdepth(Eigen::Vector3f vertex)
{
    Eigen::Vector3f kf_tri_ray = vertex/vertex(2);
    kf_tri_ray(2) = 1.0/vertex(2);

    return kf_tri_ray;
}

Eigen::Vector3f arrayToEigen(std::array<float, 3> point)
{
    return Eigen::Vector3f(point[0], point[1], point[2]);
}

std::array<float, 3> eigenToArray(Eigen::Vector3f point)
{
    std::array<float, 3> array;
    array[0] = point(0);
    array[1] = point(1);
    array[2] = point(2);
    return array;
}