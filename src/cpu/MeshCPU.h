#pragma once

#include "common/camera.h"
#include "cpu/VerticeCPU.h"
#include "cpu/TriangleCPU.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "params.h"

class MeshCPU
{
public:
    MeshCPU();

    void init(frameCPU &frame, dataCPU<float> &idepth, camera &cam, int lvl);
    void initr(frameCPU &frame, dataCPU<float> &idepth, camera &cam, int lvl);

    void setVerticeIdepth(float idepth, unsigned int id)
    {
        if (isRayIdepth)
            vertices[id].position(2) = idepth;
        else
        {
            Eigen::Vector3f pos = fromVertexToRayIdepth(vertices[id].position);
            pos(2) = idepth;
            vertices[id].position = fromRayIdepthToVertex(pos);
        }
    }

    VerticeCPU getVertice(unsigned int id)
    {
        return vertices[id];
    }

    std::array<unsigned int, 3> getTriangle(unsigned int id)
    {
        return triangles[id];
    }

    TriangleCPU getTriangleStructure(unsigned int id)
    {
        std::array<unsigned int, 3> tri = triangles[id];
        TriangleCPU t(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]);
        return t;
    }

    void addVertice(VerticeCPU &vert, unsigned int id)
    {
        vertices[id] = vert;
    }

    void addTriangle(std::array<unsigned int, 3> &tri, unsigned int id)
    {
        triangles[id] = tri;
    }

    MeshCPU getCopy()
    {
        MeshCPU meshCopy;

        meshCopy.vertices = vertices;
        meshCopy.triangles = triangles;
        meshCopy.isRayIdepth = isRayIdepth;

        return meshCopy;
    }

    std::vector<unsigned int> getVerticesIds()
    {
        std::vector<unsigned int> keys;
        for (std::map<unsigned int, VerticeCPU>::iterator it = vertices.begin(); it != vertices.end(); ++it)
        {
            keys.push_back(it->first);
        }
        return keys;
    }

    std::vector<unsigned int> getTrianglesIds()
    {
        std::vector<unsigned int> keys;
        for (std::map<unsigned int, std::array<unsigned int, 3>>::iterator it = triangles.begin(); it != triangles.end(); ++it)
        {
            keys.push_back(it->first);
        }
        return keys;
    }

    void toRayIdepth();
    void toVertex();
    void transform(Sophus::SE3f &pose);

    void computeTexCoords(camera &cam, int lvl);
    void computeTexCoords(Sophus::SE3f &pose, camera &cam, int lvl);

    bool isTrianglePresent(std::array<unsigned int, 3> &tri);

    void buildTriangles(camera &cam, int lvl);

    unsigned int getClosestVerticeId(Eigen::Vector2f &pix);
    unsigned int getClosestVerticeId(Eigen::Vector3f &v);

    unsigned int getClosestTriangleId(Eigen::Vector3f &pos);
    unsigned int getClosestTriangleId(Eigen::Vector2f &pix);

    std::map<unsigned int, VerticeCPU> vertices;
    std::map<unsigned int, std::array<unsigned int, 3>> triangles;

    bool isRayIdepth;
    
private:

};
