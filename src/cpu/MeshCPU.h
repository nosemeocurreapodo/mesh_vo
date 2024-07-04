#pragma once

#include "common/camera.h"
#include "common/common.h"
#include "cpu/Triangle2D.h"
#include "cpu/Triangle3D.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "params.h"
#include "common/DelaunayTriangulation.h"

enum MeshVerticeRepresentation
{
    cartesian,
    rayIdepth
};

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
    float getVerticeIdepth(unsigned int id);

    void transform(Sophus::SE3f pose);
    void computeTexCoords(camera &cam);
    void computeNormalizedTexCoords(camera &cam);

    std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> computeEdgeFront();
    std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> getSortedEdgeFront(Eigen::Vector2f &pix);
    std::vector<unsigned int> getSortedTriangles(Eigen::Vector2f &pix);

    bool isTrianglePresent(std::array<unsigned int, 3> &tri);

    unsigned int getClosestTriangle(Eigen::Vector2f &pix)
    {
        float min_distance = std::numeric_limits<float>::max();
        unsigned int min_id = 0;
        for (auto tri : triangles)
        {
            Triangle2D tri2D = getTriangle2D(tri.first);
            float distance1 = (tri2D.vertices[0] - pix).norm();
            float distance2 = (tri2D.vertices[1] - pix).norm();
            float distance3 = (tri2D.vertices[2] - pix).norm();

            float distance = std::min(distance1, std::min(distance2, distance3));
            if (distance < min_distance)
            {
                min_distance = distance;
                min_id = tri.first;
            }
        }
        return min_id;
    }

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

    void removeTrianglesWithoutPoints()
    {
        std::vector<unsigned int> trisIds = getTrianglesIds();
        for (auto it = trisIds.begin(); it != trisIds.end(); it++)
        {
            std::array<unsigned int, 3> vertIds = getTriangleIndices(*it);
            if (!vertices[vertIds[0]].count() || !vertices[vertIds[1]].count() || !vertices[vertIds[2]].count())
            {
                triangles.erase(*it);
            }
        }
    }

    void removeOcludedTriangles(camera &cam)
    {
        float min_area = (float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) / 2.0;
        float max_area = min_area * 4.0;
        float min_angle = M_PI / 8;

        std::vector<unsigned int> vetsToRemove;

        computeTexCoords(cam);
        std::vector<unsigned int> trisIds = getTrianglesIds();
        for (auto it = trisIds.begin(); it != trisIds.end(); it++)
        {
            // remove triangle if:
            // 1-is backface
            // 2-the points are behind the camera
            // 2-all vertices lays outside the image
            // afterwards, 
            // remove vertices
            // remove any vertex witouh triangles
            // remove any triangle without vertices
            //std::array<unsigned int, 3> vertsIds = getTriangleIndices(*it);
            Triangle3D tri3D = getTriangle3D(*it);
            if (tri3D.vertices[0](2) <= 0.0 || tri3D.vertices[1](2) <= 0.0 || tri3D.vertices[2](2) <= 0.0)
            {
                //vetsToRemove.push_back(vertsIds[0]);
                //vetsToRemove.push_back(vertsIds[1]);
                //vetsToRemove.push_back(vertsIds[2]);

                triangles.erase(*it);
                continue;
            }
            Triangle2D tri2D = getTriangle2D(*it);
            if (tri2D.getArea() < min_area || tri2D.getArea() > max_area)
            {
                //vetsToRemove.push_back(vertsIds[0]);
                //vetsToRemove.push_back(vertsIds[1]);
                //vetsToRemove.push_back(vertsIds[2]);

                triangles.erase(*it);
                continue;
            }
            /*
            std::array<float, 3> angles = tri2D.getAngles();
            if (float(angles[0]) < min_angle || float(angles[1]) < min_angle || float(angles[2]) < min_angle)
            {
                vetsToRemove.push_back(vertsIds[0]);
                vetsToRemove.push_back(vertsIds[1]);
                vetsToRemove.push_back(vertsIds[2]);

                //triangles.erase(*it);
                continue;
            }
            */
            if (!cam.isPixVisible(tri2D.vertices[0]) && !cam.isPixVisible(tri2D.vertices[1]) && !cam.isPixVisible(tri2D.vertices[2]))
            {
                //vetsToRemove.push_back(vertsIds[0]);
                //vetsToRemove.push_back(vertsIds[1]);
                //vetsToRemove.push_back(vertsIds[2]);

                triangles.erase(*it);
                continue;
            }
        }
        /*
        for(auto vert : vetsToRemove)
        {
            if(vertices[vert].count())
                vertices.erase(vert);
        }
        */
        removePointsWithoutTriangles();
        buildTriangles(cam);
        //removeTrianglesWithoutPoints();
    }

    MeshVerticeRepresentation representation;

private:
    void toRayIdepth();
    void toCartesian();

    std::map<unsigned int, Eigen::Vector3f> vertices;
    std::map<unsigned int, Eigen::Vector2f> texcoords;
    std::map<unsigned int, std::array<unsigned int, 3>> triangles;
    int last_v_id;
    int last_t_id;
};
