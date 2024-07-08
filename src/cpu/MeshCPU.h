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

    void clear()
    {
        vertices.clear();
        texcoords.clear();
        triangles.clear();

        last_v_id = 0;
        last_t_id = 0;
    }

    MeshCPU getCopy();

    Eigen::Vector3f getVertice(unsigned int id);
    Eigen::Vector2f getTexCoord(unsigned int id);
    std::array<unsigned int, 3> getTriangleIndices(unsigned int id);
    Triangle2D getTexCoordTriangle(unsigned int id);
    Triangle3D getCartesianTriangle(unsigned int id);
    std::vector<unsigned int> getVerticesIds();
    std::vector<unsigned int> getTrianglesIds();

    unsigned int addVertice(Eigen::Vector3f &vert);
    unsigned int addTriangle(std::array<unsigned int, 3> &tri);
    void setVerticeDepth(float depth, unsigned int id);
    float getVerticeDepth(unsigned int id);

    void transform(Sophus::SE3f pose);
    void computeTexCoords(camera &cam);
    void computeNormalizedTexCoords(camera &cam);

    std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> computeEdgeFront();
    std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> getSortedEdgeFront(std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> &edgeFront, Eigen::Vector2f &pix);
    std::vector<unsigned int> getSortedTexCoordTriangles(Eigen::Vector2f &pix);

    bool isTrianglePresent(std::array<unsigned int, 3> &tri);

    unsigned int getClosestTexCoordTriangle(Eigen::Vector2f &pix);
    void buildTriangles(camera &cam);
    void removePointsWithoutTriangles();
    void removeTrianglesWithoutPoints();
    void removeOcludedTriangles(camera &cam);

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
                    Triangle3D tri3D = getCartesianTriangle(edge.second);
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
