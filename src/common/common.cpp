#include "common/common.h"

#include <Eigen/Core>

Eigen::Vector3f rayIdepthToCartesian(Eigen::Vector3f rayIdepth)
{
    Eigen::Vector3f kf_tri_ray;
    kf_tri_ray(0) = rayIdepth(0);
    kf_tri_ray(1) = rayIdepth(1);
    kf_tri_ray(2) = 1.0;
    float kf_tri_idepth = rayIdepth(2);

    return kf_tri_ray / kf_tri_idepth;
}

Eigen::Vector3f cartesianToRayIdepth(Eigen::Vector3f vertex)
{
    Eigen::Vector3f kf_tri_ray = vertex / vertex(2);
    kf_tri_ray(2) = 1.0 / vertex(2);

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

bool isTriangleEqual(std::array<unsigned int, 3> tri_indices_1, std::array<unsigned int, 3> tri_indices_2)
{
    bool isIndicePresent[3];
    for (int tri_indice = 0; tri_indice < 3; tri_indice++)
    {
        isIndicePresent[tri_indice] = false;
        if (tri_indices_1[tri_indice] == tri_indices_2[0] || tri_indices_1[tri_indice] == tri_indices_2[1] || tri_indices_1[tri_indice] == tri_indices_2[2])
            isIndicePresent[tri_indice] = true;
    }
    if (isIndicePresent[0] && isIndicePresent[1] && isIndicePresent[2])
        return true;
    return false;
}

bool isEdgeEqual(std::array<unsigned int, 2> edge_indices_1, std::array<unsigned int, 2> edge_indices_2)
{
    bool isIndicePresent[2];
    for (int edge_indice = 0; edge_indice < 2; edge_indice++)
    {
        isIndicePresent[edge_indice] = false;
        if (edge_indices_1[edge_indice] == edge_indices_2[0] || edge_indices_1[edge_indice] == edge_indices_2[1])
            isIndicePresent[edge_indice] = true;
    }
    if (isIndicePresent[0] && isIndicePresent[1])
        return true;
    return false;
}
