#pragma once

#include <Eigen/Core>

Eigen::Vector3f fromRayIdepthToVertex(Eigen::Vector3f rayIdepth);
Eigen::Vector3f fromVertexToRayIdepth(Eigen::Vector3f vertex);
Eigen::Vector3f arrayToEigen(std::array<float, 3> point);
std::array<float, 3> eigenToArray(Eigen::Vector3f point);
