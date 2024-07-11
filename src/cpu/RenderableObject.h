#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "cpu/PointSet.h"
#include "cpu/Polygon.h"
#include "common/common.h"
#include "common/DelaunayTriangulation.h"
#include "params.h"

class RenderableObject
{
public:
    RenderableObject()
    {

    };

    RenderableObject(const RenderableObject &other)
    {

    }

    RenderableObject &operator=(const RenderableObject &other)
    {
        if (this != &other)
        {

        }
        return *this;
    }

    void clear()
    {

    }

    void clearPolygons()
    {

    }

    void setTriangles(std::map<unsigned int, std::array<unsigned int, 3>> &new_tris)
    {
        triangles = new_tris;
    }

    void removeTriangle(unsigned int id)
    {
        triangles.erase(id);
    }

    std::array<unsigned int, 3> getTriangleIndices(unsigned int id)
    {
        if (!triangles.count(id))
            throw std::out_of_range("getTriangleIndices invalid id");
        return triangles[id];
    }

    unsigned int addTriangle(std::array<unsigned int, 3> &tri)
    {
        last_triangle_id++;
        if (triangles.count(last_triangle_id))
            throw std::out_of_range("addTriangle id already exist");
        triangles[last_triangle_id] = tri;
        return last_triangle_id;
    }

    void setTriangleIndices(std::array<unsigned int, 3> &tri, unsigned int id)
    {
        if (!triangles.count(id))
            throw std::out_of_range("setTriangleIndices invalid id");
        triangles[id] = tri;
    }

    PolygonFlat getPolygon(unsigned int id)
    {
        // always return triangle in cartesian
        std::array<unsigned int, 3> tri = getTriangleIndices(id);
        PolygonFlat t(getVertice(tri[0]), getVertice(tri[1]), getVertice(tri[2]));
        return t;
    }

    std::vector<unsigned int> getPolygonIds()
    {
        std::vector<unsigned int> keys;
        for (auto it = triangles.begin(); it != triangles.end(); ++it)
        {
            keys.push_back(it->first);
        }
        return keys;
    }


    bool isPolygonPresent(std::array<unsigned int, 3> &tri)
    {
        for (auto it = triangles.begin(); it != triangles.end(); ++it)
        {
            std::array<unsigned int, 3> tri2 = it->second;

            if (isTriangleEqual(tri, tri2))
                return true;
        }
        return false;
    }

private:

};
