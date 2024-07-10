#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "PointSetCPU.h"
#include "cpu/Triangle3D.h"
#include "common/common.h"
#include "params.h"

class MeshCPU : public PointSetCPU
{
public:
    MeshCPU() : PointSetCPU()
    {
        last_triangle_id = 0;
    };

    MeshCPU(const MeshCPU &other) : PointSetCPU(other)
    {
        triangles = other.triangles;
        last_triangle_id = other.last_triangle_id;
    }

    MeshCPU &operator=(const MeshCPU &other)
    {
        if (this != &other)
        {
            PointSetCPU::operator=(other);
            triangles = other.triangles;
            last_triangle_id = other.last_triangle_id;
        }
        return *this;
    }

    void clear()
    {
        PointSetCPU::clear();
        triangles.clear();
        last_triangle_id = 0;
    }

    void clearTriangles()
    {
        triangles.clear();
    }

    void setTriangles(std::map<unsigned int, std::array<unsigned int, 3>> &new_tris)
    {
        triangles = new_tris;
    }

    void removeTriangle(unsigned int id)
    {
        triangles.erase(id);
    }

    std::array<unsigned int, 3> getTriangleIndices(unsigned int id);
    unsigned int addTriangle(std::array<unsigned int, 3> &tri);
    void setTriangleIndices(std::array<unsigned int, 3> &tri, unsigned int id);
    Polygon getCartesianTriangle(unsigned int id);
    std::vector<unsigned int> getTrianglesIds();

    std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> computeEdgeFront();

    bool isTrianglePresent(std::array<unsigned int, 3> &tri);

    void removePointsWithoutTriangles();
    void removeTrianglesWithoutPoints();

private:
    std::map<unsigned int, std::array<unsigned int, 3>> triangles;

    int last_triangle_id;
};
