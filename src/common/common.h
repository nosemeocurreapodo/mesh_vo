#pragma once

#include <Eigen/Core>

Eigen::Vector3f fromRayIdepthToVertex(Eigen::Vector3f rayIdepth);
Eigen::Vector3f fromVertexToRayIdepth(Eigen::Vector3f vertex);