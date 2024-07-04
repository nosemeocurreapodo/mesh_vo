#pragma once

#include <Eigen/Core>

Eigen::Vector3f rayIdepthToCartesian(Eigen::Vector3f rayIdepth);
Eigen::Vector3f cartesianToRayIdepth(Eigen::Vector3f vertex);

Eigen::Vector3f arrayToEigen(std::array<float, 3> point);
std::array<float, 3> eigenToArray(Eigen::Vector3f point);

bool isTriangleEqual(std::array<unsigned int, 3> tri_indices_1, std::array<unsigned int, 3> tri_indices_2);
bool isEdgeEqual(std::array<unsigned int, 2> edge_indices_1, std::array<unsigned int, 2> edge_indices_2);

