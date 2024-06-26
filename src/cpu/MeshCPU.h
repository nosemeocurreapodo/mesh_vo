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

    int getVerticesSize()
    {
        return (int)vertices.size();
    }

    int getTrianglesSize()
    {
        return (int)triangles.size();
    }

    void addVertice(VerticeCPU &vert)
    {
        vertices.push_back(vert);
        buildTriangles(triangles);
    }

    void addTriangle(TriangleCPU &tri)
    {
        triangles.push_back(tri);
        buildTriangles(triangles);
    }

    std::vector<float> getVerticesIdepths()
    {
        std::vector<float> idepths;
        for(int i = 0; i < (int)vertices.size(); i++)
        {
            if(isRayIdepth)
                idepths.push_back(vertices[i].position(2));
            else
                idepths.push_back(1.0/vertices[i].position(2));
        }
        return idepths;
    }

    void setVerticesIdepths(std::vector<float> &idepths)
    {
        for(int i = 0; i < (int)vertices.size(); i++)
        {
            if(isRayIdepth)
                vertices[i].position(2) = idepths[i];
            else
            {
                vertices[i].position = (vertices[i].position/vertices[i].position(2))/idepths[i];
            }
        } 
    }

    MeshCPU getCopy();

    void toRayIdepth();
    void toVertex();
    void transform(Sophus::SE3f &pose);
    void computeTexCoords(camera &cam, int lvl);

    VerticeCPU getVertexFromId(unsigned int id);
    TriangleCPU getTriangleFromId(unsigned int id);

    bool isTrianglePresent(TriangleCPU &tri);

    void removeVerticeByIndex(int vertice_index);
    void removeVerticeById(int vertice_id);
    void removeTriangleByIndex(int triangle_index);
    void removeTriangleById(int triangle_id);

    void buildTriangles(camera &cam, int lvl);
    void buildTriangles(std::vector<TriangleCPU> &tris);

    TriangleCPU getClosestTriangle(Eigen::Vector3f &pos);
    TriangleCPU getClosestTriangle(Eigen::Vector2f &pix);

    VerticeCPU getClosestVertice(Eigen::Vector2f &pix);
    VerticeCPU getClosestVertice(Eigen::Vector3f &v);

    int getClosestVerticeIndex(Eigen::Vector2f &pix);
    int getClosestVerticeIndex(Eigen::Vector3f &v);

    int getClosestTriangleIndex(Eigen::Vector3f &pos);
    int getClosestTriangleIndex(Eigen::Vector2f &pix);

    int getVertexIndexFromId(unsigned int id);
    int getTriangleIndexFromId(unsigned int id);
    int getVertexIdFromIndex(unsigned int in);
    int getTriangleIdFromIndex(unsigned int in);

    VerticeCPU getVertexFromIndex(unsigned int in);
    TriangleCPU getTriangleFromIndex(unsigned int in);

private:
    // scene
    // Triangles contains basically pointers to the corresponding 3 vertices
    std::vector<TriangleCPU> triangles;
    // the vertices, actual data of the scene
    // the vector container can change the pointer if we add or remove elements
    // so we have to be carefull
    std::vector<VerticeCPU> vertices;

    bool isRayIdepth;
};
