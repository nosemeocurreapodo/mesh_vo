#pragma once

#include "common/camera.h"
#include "common/common.h"
#include "cpu/Triangle2D.h"
#include "cpu/Triangle3D.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "params.h"
#include "common/DelaunayTriangulation.h"

class MeshCPU
{
public:
    MeshCPU();

    void init(frameCPU &frame, dataCPU<float> &idepth, camera &cam, int lvl);

    Eigen::Vector3f getVertice(unsigned int id);
    Eigen::Vector2f &getTexCoord(unsigned int id);
    std::array<unsigned int, 3> getTriangleIndices(unsigned int id);
    Triangle2D getTriangle2D(unsigned int id);
    Triangle3D getTriangle3D(unsigned int id);
    MeshCPU getCopy();
    std::vector<unsigned int> getVerticesIds();
    std::vector<unsigned int> getTrianglesIds();

    unsigned int addVertice(Eigen::Vector3f &vert);
    unsigned int addVertice(Eigen::Vector3f &vert, Eigen::Vector2f &tex);
    unsigned int addTriangle(std::array<unsigned int, 3> &tri);
    void setVerticeIdepth(float idepth, unsigned int id);

    void toRayIdepth();
    void toVertex();
    void transform(Sophus::SE3f pose);
    void computeTexCoords(camera &cam);
    void computeNormalizedTexCoords(camera &cam);

    std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> computeEdgeFront();
    std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> getSortedEdgeFront(Eigen::Vector2f &pix);
    std::vector<unsigned int> getSortedTriangles(Eigen::Vector2f &pix);

    bool isTrianglePresent(std::array<unsigned int, 3> &tri);

    void buildTriangles(camera &cam)
    {
        triangles.clear();
        computeTexCoords(cam);
        DelaunayTriangulation triangulation;
        triangulation.loadPoints(texcoords);
        triangulation.triangulate();
        triangles = triangulation.getTriangles();
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
                vertices.erase(*it);
        }
    }

    void removeOcluded(camera &cam)
    {
        std::vector<unsigned int> trisIds = getTrianglesIds();
        for (auto it = trisIds.begin(); it != trisIds.end(); it++)
        {
            // remove triangle if:
            // 1-is backface
            // 2-any vertex lays outside the image
            // afterwards, remove any vertex witouh triangles
            Triangle2D tri = getTriangle2D(*it);
            if (tri.getArea() < 0.0)
            {
                triangles.erase(*it);
                continue;
            }

            std::array<unsigned int, 3> vertsIds = getTriangleIndices(*it);
            for (int i = 0; i < 3; i++)
            {
                Eigen::Vector3f ray = vertices[vertsIds[i]] / vertices[vertsIds[i]](2);
                Eigen::Vector2f pix = cam.rayToPix(ray);
                if (!cam.isPixVisible(pix))
                {
                    triangles.erase(*it);
                    break;
                }
            }
        }

        removePointsWithoutTriangles();
    }

private:
    std::map<unsigned int, Eigen::Vector3f> vertices;
    std::map<unsigned int, Eigen::Vector2f> texcoords;
    std::map<unsigned int, std::array<unsigned int, 3>> triangles;

    bool isRayIdepth;
};
