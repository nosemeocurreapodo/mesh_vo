#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "cpu/PointSet.h"
#include "cpu/Polygon.h"
#include "common/common.h"
#include "common/DelaunayTriangulation.h"
#include "params.h"

class Mesh : public PointSet
{
public:
    Mesh() : PointSet()
    {
        last_triangle_id = 0;
    };

    Mesh(const Mesh &other) : PointSet(other)
    {
        triangles = other.triangles;
        last_triangle_id = other.last_triangle_id;
    }

    Mesh &operator=(const Mesh &other)
    {
        if (this != &other)
        {
            PointSet::operator=(other);
            triangles = other.triangles;
            last_triangle_id = other.last_triangle_id;
        }
        return *this;
    }

    void clear()
    {
        PointSet::clear();
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
        PolygonFlat t(getVertice(tri[0]), getVertice(tri[1]), getVertice(tri[2]),
                      getTexCoord(tri[0]), getTexCoord(tri[1]), getTexCoord(tri[2]));
        return t;
    }

    std::vector<unsigned int> getTrianglesIds()
    {
        std::vector<unsigned int> keys;
        for (auto it = triangles.begin(); it != triangles.end(); ++it)
        {
            keys.push_back(it->first);
        }
        return keys;
    }

    std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> computeEdgeFront()
    {
        std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> edgeFront;
        for (auto it = triangles.begin(); it != triangles.end(); ++it)
        {
            auto triIndices = it->second;

            // Triangle3D tri3D = getTriangle3D(it->first);
            // if (tri3D.isBackFace())
            //     continue;

            std::array<unsigned int, 2> edges[3];
            edges[0] = {triIndices[0], triIndices[1]};
            edges[1] = {triIndices[1], triIndices[2]};
            edges[2] = {triIndices[2], triIndices[0]};

            for (int i = 0; i < 3; i++)
            {
                int edge_index = -1;
                for (int j = 0; j < edgeFront.size(); j++)
                {
                    std::array<unsigned int, 2> ef = edgeFront[j].first;
                    unsigned int t_id = edgeFront[j].second;
                    if (isEdgeEqual(edges[i], ef))
                    {
                        edge_index = j;
                        break;
                    }
                }
                if (edge_index >= 0)
                    edgeFront.erase(edgeFront.begin() + edge_index);
                else
                    edgeFront.push_back({edges[i], it->first});
            }
        }
        return edgeFront;
    }

    bool isTrianglePresent(std::array<unsigned int, 3> &tri)
    {
        for (auto it = triangles.begin(); it != triangles.end(); ++it)
        {
            std::array<unsigned int, 3> tri2 = it->second;

            if (isTriangleEqual(tri, tri2))
                return true;
        }
        return false;
    }

    void removePointsWithoutTriangles()
    {
        std::vector<unsigned int> vertsIds = getVerticesIds();
        for (auto it = vertsIds.begin(); it != vertsIds.end(); it++)
        {
            bool remove = true;
            for (auto t_it = triangles.begin(); t_it != triangles.end(); t_it++)
            {
                if (*it == t_it->second[0] || *it == t_it->second[1] || *it == t_it->second[2])
                {
                    remove = false;
                    break;
                }
            }
            if (remove)
                removeVertice(*it);
        }
    }

    void removeTrianglesWithoutPoints()
    {
        std::vector<unsigned int> trisIds = getTrianglesIds();
        for (auto it = trisIds.begin(); it != trisIds.end(); it++)
        {
            std::array<unsigned int, 3> vertIds = getTriangleIndices(*it);
            if (!vertices.count(vertIds[0]) || !vertices.count(vertIds[1]) || !vertices.count(vertIds[2]))
            {
                triangles.erase(*it);
            }
        }
    }

    void buildTriangles(camera &cam)
    {
        projectToCamera(cam);
        DelaunayTriangulation triangulation;
        triangulation.loadPoints(texcoords);
        triangulation.triangulate();
        std::map<unsigned int, std::array<unsigned int, 3>> tris = triangulation.getTriangles();
        clearTriangles();
        setTriangles(tris);
    }
    /*
    void removeOcluded(camera &cam)
    {
        projectToCamera(cam);
        std::vector<unsigned int> trisIds = getTrianglesIds();
        for (auto it = trisIds.begin(); it != trisIds.end(); it++)
        {
            // remove triangle if:
            // 1-is backface
            // 2-the points are behind the camera
            // 2-all vertices lays outside the image
            // afterwards,
            // remove vertices without triangles
            Polygon pol = getPolygon(*it);
            if (pol.vertices[0](2) <= 0.0 || pol.vertices[1](2) <= 0.0 || pol.vertices[2](2) <= 0.0)
            {
                removeTriangle(*it);
                continue;
            }
            if (pol.getArea() < 1.0)
            {
                removeTriangle(*it);
                continue;
            }
            if (!cam.isPixVisible(pol.vertices[0]) && !cam.isPixVisible(pol.vertices[1]) && !cam.isPixVisible(pol.vertices[2]))
            {
                removeTriangle(*it);
                continue;
            }
        }
        removePointsWithoutTriangles();
    }
    */

private:
    std::map<unsigned int, std::array<unsigned int, 3>> triangles;
    int last_triangle_id;
};
