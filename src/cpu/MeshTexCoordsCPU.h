#pragma once

#include "MeshCPU.h"
#include "common/camera.h"
#include "common/common.h"
#include "cpu/Triangle2D.h"
#include "params.h"
#include "common/DelaunayTriangulation.h"

class MeshTexCoordsCPU : public MeshCPU
{
public:

    MeshTexCoordsCPU() : MeshCPU()
    {

    }

    MeshTexCoordsCPU(const MeshTexCoordsCPU &other) : MeshCPU(other)
    {
        texcoords = other.texcoords;
    }

    MeshTexCoordsCPU &operator=(const MeshTexCoordsCPU &other)
    {
        if (this != &other)
        {
            MeshCPU::operator=(other);   // Call base class assignment operator
            texcoords = other.texcoords; // Copy texcoord member
        }
        return *this;
    }

    void clear()
    {
        MeshCPU::clear();
        texcoords.clear();
    }

    Eigen::Vector2f getTexCoord(unsigned int id)
    {
        if (!texcoords.count(id))
            throw std::out_of_range("getTexCoord invalid id");
        return texcoords[id];
    }

    Triangle2D getTexCoordTriangle(unsigned int id)
    {
        std::array<unsigned int, 3> tri = getTriangleIndices(id);
        Triangle2D t(getTexCoord(tri[0]), getTexCoord(tri[1]), getTexCoord(tri[2]));
        return t;
    }

    void computeTexCoords(camera &cam)
    {
        texcoords.clear();
        std::vector<unsigned int> ids = getVerticesIds();
        for (auto id : ids)
        {
            Eigen::Vector3f ray;
            if (representation == rayIdepth)
            {
                ray = getVertice(id);
                ray(2) = 1.0;
            }
            if (representation == cartesian)
                ray = getVertice(id) / getVertice(id)(2);

            Eigen::Vector2f pix = cam.rayToPix(ray);

            texcoords[id] = pix;
        }
    }

    std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> getSortedEdgeFront(std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> &edgeFront, Eigen::Vector2f &pix)
    {
        std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> sortedEdgeFront;

        auto auxEdgeFront = edgeFront;

        while (auxEdgeFront.size() > 0)
        {
            size_t closest;
            float closest_distance = std::numeric_limits<float>::max();
            for (auto it = auxEdgeFront.begin(); it != auxEdgeFront.end(); it++)
            {
                Triangle2D tri = getTexCoordTriangle(it->second);
                std::array<unsigned int, 2> edge = it->first;
                Eigen::Vector2f e1 = getTexCoord(edge[0]);
                Eigen::Vector2f e2 = getTexCoord(edge[1]);
                float distance = std::numeric_limits<float>::max();
                if (tri.getArea() > 10)
                    distance = ((e1 + e2) / 2.0 - pix).norm();
                if (distance <= closest_distance)
                {
                    closest_distance = distance;
                    closest = it - auxEdgeFront.begin();
                }
            }
            sortedEdgeFront.push_back(auxEdgeFront[closest]);
            auxEdgeFront.erase(auxEdgeFront.begin() + closest);
        }

        return sortedEdgeFront;
    }

    std::vector<unsigned int> getSortedTexCoordTriangles(Eigen::Vector2f &pix)
    {
        std::vector<unsigned int> sortedTriangles;
        std::vector<unsigned int> trisIds = getTrianglesIds();

        while (trisIds.size() > 0)
        {
            size_t closestIndex;
            float closest_distance = std::numeric_limits<float>::max();
            for (auto it = trisIds.begin(); it != trisIds.end(); it++)
            {
                Triangle2D tri = getTexCoordTriangle(*it);
                float distance = std::numeric_limits<float>::max();
                if (tri.getArea() > 10)
                    distance = (tri.getMean() - pix).norm();
                if (distance <= closest_distance)
                {
                    closest_distance = distance;
                    closestIndex = it - trisIds.begin();
                }
            }
            sortedTriangles.push_back(trisIds[closestIndex]);
            trisIds.erase(trisIds.begin() + closestIndex);
        }

        return sortedTriangles;
    }

    unsigned int getClosestTexCoordTriangle(Eigen::Vector2f &pix)
    {
        float min_distance = std::numeric_limits<float>::max();
        unsigned int min_id = 0;
        std::vector<unsigned int> ids = getTrianglesIds();
        for (auto id : ids)
        {
            Triangle2D tri2D = getTexCoordTriangle(id);
            if (tri2D.getArea() < 10)
                continue;
            float distance1 = (tri2D.vertices[0] - pix).norm();
            float distance2 = (tri2D.vertices[1] - pix).norm();
            float distance3 = (tri2D.vertices[2] - pix).norm();

            float distance = std::min(distance1, std::min(distance2, distance3));
            if (distance < min_distance)
            {
                min_distance = distance;
                min_id = id;
            }
        }
        return min_id;
    }

    void buildTriangles(camera &cam)
    {
        computeTexCoords(cam);
        DelaunayTriangulation triangulation;
        triangulation.loadPoints(texcoords);
        triangulation.triangulate();
        std::map<unsigned int, std::array<unsigned int, 3>> tris = triangulation.getTriangles();
        clearTriangles();
        setTriangles(tris);
    }

    void removeOcluded(camera &cam)
    {
        computeTexCoords(cam);
        std::vector<unsigned int> trisIds = getTrianglesIds();
        for (auto it = trisIds.begin(); it != trisIds.end(); it++)
        {
            // remove triangle if:
            // 1-is backface
            // 2-the points are behind the camera
            // 2-all vertices lays outside the image
            // afterwards,
            // remove vertices without triangles
            Polygon tri3D = getCartesianTriangle(*it);
            if (tri3D.vertices[0](2) <= 0.0 || tri3D.vertices[1](2) <= 0.0 || tri3D.vertices[2](2) <= 0.0)
            {
                removeTriangle(*it);
                continue;
            }
            Triangle2D tri2D = getTexCoordTriangle(*it);
            if (tri2D.getArea() < 1.0)
            {
                removeTriangle(*it);
                continue;
            }
            if (!cam.isPixVisible(tri2D.vertices[0]) && !cam.isPixVisible(tri2D.vertices[1]) && !cam.isPixVisible(tri2D.vertices[2]))
            {
                removeTriangle(*it);
                continue;
            }
        }
        removePointsWithoutTriangles();
    }

    void devideBigTriangles(camera &cam)
    {
        computeTexCoords(cam);

        float max_area = 2 * (float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1));

        std::vector<unsigned int> trisIds = getTrianglesIds();
        for (auto it = trisIds.begin(); it != trisIds.end(); it++)
        {
            Triangle2D tri2D = getTexCoordTriangle(*it);
            if (!cam.isPixVisible(tri2D.vertices[0]) && !cam.isPixVisible(tri2D.vertices[1]) && !cam.isPixVisible(tri2D.vertices[2]))
            {
                continue;
            }
            if (std::fabs(tri2D.getArea()) > max_area)
            {
                Eigen::Vector2f pix = tri2D.getMean();
                Eigen::Vector3f ray = cam.pixToRay(pix);

                Polygon tri3D = getCartesianTriangle(*it);
                float depth = tri3D.getDepth(ray);
                Eigen::Vector3f point = ray * depth;

                addVertice(point);
            }
        }
    }

    void extrapolateMesh(camera &cam, dataCPU<float> &mask, int lvl)
    {
        computeTexCoords(cam);

        float step_x = 0.25 * float(cam.width) / (MESH_WIDTH - 1);
        float step_y = 0.25 * float(cam.height) / (MESH_HEIGHT - 1);

        std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> edgeFront = computeEdgeFront();

        for (int y = -step_y * 1; y < cam.height + step_y * 1; y += step_y)
        {
            for (int x = -step_x * 1; x < cam.width + step_x * 1; x += step_x)
            {
                Eigen::Vector2f pix(x, y);

                if (cam.isPixVisible(pix) && mask.get(y, x, lvl) != mask.nodata)
                    continue;

                // unsigned int triId = getClosestTexCoordTriangle(pix);

                float depth = 0.0;
                int count = 0;
                Eigen::Vector3f ray = cam.pixToRay(pix);

                auto sortedEdgeFront = getSortedEdgeFront(edgeFront, pix);
                for (auto edge : sortedEdgeFront)
                {
                    Polygon tri3D = getCartesianTriangle(edge.second);
                    float ndepth = tri3D.getDepth(ray);
                    if (ndepth <= 0.0)
                        continue;
                    depth += ndepth;
                    count++;
                    if (count > 3)
                        break;
                }

                /*
                std::vector<unsigned int> trisIds = getSortedTexCoordTriangles(pix);
                for (auto triId : trisIds)
                {
                    Triangle3D tri3D = getCartesianTriangle(triId);
                    float ndepth = tri3D.getDepth(ray);
                    if (ndepth <= 0.0)
                        continue;
                    depth += ndepth;
                    count++;
                    if (count > 3)
                        break;
                }
                */

                if (count == 0)
                    continue;

                depth = depth / count;
                Eigen::Vector3f new_vertice = ray * depth;

                unsigned int id = addVertice(new_vertice);
            }
        }

        buildTriangles(cam);
    }

private:
    std::map<unsigned int, Eigen::Vector2f> texcoords;
};
