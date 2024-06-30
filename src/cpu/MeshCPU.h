#pragma once

#include "common/camera.h"
#include "common/common.h"
#include "cpu/Triangle2D.h"
#include "cpu/Triangle3D.h"
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
            vertices[id](2) = idepth;
        else
        {
            Eigen::Vector3f pos = fromVertexToRayIdepth(vertices[id]);
            pos(2) = idepth;
            vertices[id] = fromRayIdepthToVertex(pos);
        }
    }

    Eigen::Vector3f getVertice(unsigned int id)
    {
        return vertices[id];
    }

    Eigen::Vector2f& getTexCoord(unsigned int id)
    {
        return texcoords[id];
    }

    std::array<unsigned int, 3> getTriangleIndices(unsigned int id)
    {
        return triangles[id];
    }

    Triangle2D getTriangle2D(unsigned int id)
    {
        std::array<unsigned int, 3> tri = triangles[id];
        Triangle2D t(texcoords[tri[0]], texcoords[tri[1]], texcoords[tri[2]]);
        return t;
    }

    Triangle3D getTriangle3D(unsigned int id)
    {
        std::array<unsigned int, 3> tri = triangles[id];
        Triangle3D t(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]);
        return t;
    }

    unsigned int addVertice(Eigen::Vector3f &vert)
    {
        unsigned int v_id = 0;
        for(auto vert : vertices)
        {
            if(vert.first > v_id)
                v_id = vert.first;
        }
        v_id++;
        vertices[v_id] = vert;
        return v_id;
    }

    unsigned int addVertice(Eigen::Vector3f &vert, Eigen::Vector2f &tex)
    {
        unsigned int v_id = 0;
        for(auto vert : vertices)
        {
            if(vert.first > v_id)
                v_id = vert.first;
        }
        v_id++;
        vertices[v_id] = vert;
        texcoords[v_id] = tex;
        return v_id;
    }

    unsigned int addTriangle(std::array<unsigned int, 3> &tri)
    {
        unsigned int t_id = 0;
        for(auto tri : triangles)
        {
            if(tri.first > t_id)
                t_id = tri.first;
        }
        t_id++;
        triangles[t_id] = tri;
        return t_id;
    }

    /*
    void addTriangle(std::array<unsigned int, 3> &tri, unsigned int id)
    {
        triangles[id] = tri;
    }
    */

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
        for (auto it = vertices.begin(); it != vertices.end(); ++it)
        {
            keys.push_back(it->first);
        }
        return keys;
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

    void toRayIdepth();
    void toVertex();
    void transform(Sophus::SE3f &pose);

    bool isTrianglePresent(std::array<unsigned int, 3> &tri);

    unsigned int getClosestVerticeId(Eigen::Vector2f &pix);
    unsigned int getClosestVerticeId(Eigen::Vector3f &v);

    unsigned int getClosestTriangleId(Eigen::Vector3f &pos);
    unsigned int getClosestTriangleId(Eigen::Vector2f &pix);

    void computeTexCoords(camera &cam, int lvl)
    {
        texcoords.clear();
        for (auto it = vertices.begin(); it != vertices.end(); ++it)
        {
            Eigen::Vector3f ray;
            if (isRayIdepth)
                ray = it->second;
            else
                ray = it->second / it->second(2);

            Eigen::Vector2f pix;
            pix(0) = cam.fx[lvl] * ray(0) + cam.cx[lvl];
            pix(1) = cam.fy[lvl] * ray(1) + cam.cy[lvl];

            texcoords[it->first] = pix;
        }
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

private:
    std::map<unsigned int, Eigen::Vector3f> vertices;
    std::map<unsigned int, Eigen::Vector2f> texcoords;
    std::map<unsigned int, std::array<unsigned int, 3>> triangles;

    bool isRayIdepth;
};
