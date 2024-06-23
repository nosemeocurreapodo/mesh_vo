#pragma once

#include "common/Vertex.h"
#include "common/Triangle.h"
#include "common/camera.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "params.h"

class Mesh
{
public:
    Mesh(){

    };

    void init(frameCPU &frame, dataCPU<float> &idepth, camera &cam, int lvl)
    {
        vertices.clear();
        triangles.clear();

        // preallocate scene vertices to zero
        for (int y = 0; y < MESH_HEIGHT; y++)
        {
            for (int x = 0; x < MESH_WIDTH; x++)
            {
                Eigen::Vector2f pix;
                pix[0] = cam.width[lvl] * float(x) / MESH_WIDTH;
                pix[1] = cam.height[lvl] * float(y) / MESH_HEIGHT;
                Eigen::Vector3f ray;
                ray(0) = cam.fxinv[lvl] * pix[0] + cam.cxinv[lvl];
                ray(1) = cam.fyinv[lvl] * pix[1] + cam.cyinv[lvl];
                ray(2) = 1.0;
                float id = idepth.get(pix[1], pix[0], lvl);
                Eigen::Vector3f point = ray / id;

                Vertex vertex(point, pix, vertices.size());

                vertices.push_back(vertex);
            }
        }

        // init scene indices
        for (int y = 0; y < MESH_HEIGHT; y++)
        {
            for (int x = 0; x < MESH_WIDTH; x++)
            {
                if (x > 0 && y > 0)
                {
                    // if (((x % 2 == 0)))
                    //  if(((x % 2 == 0) && (y % 2 == 0)) || ((x % 2 != 0) && (y % 2 != 0)))
                    if (rand() > 0.5 * RAND_MAX)
                    {
                        unsigned int v11 = x - 1 + y * (MESH_WIDTH);
                        unsigned int v12 = x + (y - 1) * (MESH_WIDTH);
                        unsigned int v13 = x - 1 + (y - 1) * (MESH_WIDTH);

                        Triangle tri1(vertices[v11], vertices[v12], vertices[v13], triangles.size());
                        triangles.push_back(tri1);

                        vertices[v11].triangles.push_back(&tri1);
                        vertices[v12].triangles.push_back(&tri1);
                        vertices[v13].triangles.push_back(&tri1);

                        unsigned int v21 = x + y * (MESH_WIDTH);
                        unsigned int v22 = x + (y - 1) * (MESH_WIDTH);
                        unsigned int v23 = x - 1 + y * (MESH_WIDTH);

                        Triangle tri2(vertices[v21], vertices[v22], vertices[v23], triangles.size());
                        triangles.push_back(tri2);

                        vertices[v21].triangles.push_back(&tri2);
                        vertices[v22].triangles.push_back(&tri2);
                        vertices[v23].triangles.push_back(&tri2);
                    }
                    else
                    {
                        unsigned int v11 = x + y * (MESH_WIDTH);
                        unsigned int v12 = x - 1 + (y - 1) * (MESH_WIDTH);
                        unsigned int v13 = x - 1 + y * (MESH_WIDTH);

                        Triangle tri1(vertices[v11], vertices[v12], vertices[v13], triangles.size());
                        triangles.push_back(tri1);

                        vertices[v11].triangles.push_back(&tri1);
                        vertices[v12].triangles.push_back(&tri1);
                        vertices[v13].triangles.push_back(&tri1);

                        unsigned int v21 = x + y * (MESH_WIDTH);
                        unsigned int v22 = x + (y - 1) * (MESH_WIDTH);
                        unsigned int v23 = x - 1 + (y + 1) * (MESH_WIDTH);

                        Triangle tri2(vertices[v21], vertices[v22], vertices[v23], triangles.size());
                        triangles.push_back(tri2);

                        vertices[v21].triangles.push_back(&tri2);
                        vertices[v22].triangles.push_back(&tri2);
                        vertices[v23].triangles.push_back(&tri2);
                    }
                }
            }
        }
    }

    unsigned int getVertexIndexById(unsigned int id)
    {
        for (size_t i = 0; i < vertices.size(); i++)
        {
            if (id == vertices[i].id)
                return (unsigned int)i;
        }
    }

    unsigned int getTriangleIndexById(unsigned int id)
    {
        for (size_t i = 0; i < vertices.size(); i++)
        {
            if (id == triangles[i].id)
                return (unsigned int)i;
        }
    }

    Mesh getCopy()
    {
        Mesh meshCopy;
        for (size_t i = 0; i < vertices.size(); i++)
        {
            Vertex ver = vertices[i];
            Eigen::Vector3f pos = ver.position;
            Eigen::Vector2f tex = ver.texcoord;
            unsigned int id = ver.id;
            Vertex new_ver(pos, tex, id);
            meshCopy.vertices.push_back(new_ver);
        }

        for (size_t i = 0; triangles.size(); i++)
        {
            Triangle tri = triangles[i];
            std::array<Vertex *, 3> tri_vertices = tri.vertices;

            unsigned int v1_in = meshCopy.getVertexIndexById(tri_vertices[0]->id);
            unsigned int v2_in = meshCopy.getVertexIndexById(tri_vertices[1]->id);
            unsigned int v3_in = meshCopy.getVertexIndexById(tri_vertices[2]->id);

            unsigned int id = tri.id;

            Triangle new_tri(vertices[v1_in], vertices[v2_in], vertices[v3_in], id);
            meshCopy.triangles.push_back(new_tri);

            vertices[v1_in].triangles.push_back(&new_tri);
            vertices[v1_in].triangles.push_back(&new_tri);
            vertices[v1_in].triangles.push_back(&new_tri);
        }

        return meshCopy;
    }

    bool rayHitsMesh(Eigen::Vector3f &ray)
    {
    }

    unsigned int getClosesVertexIndex(std::array<float, 3> point)
    {
        Eigen::Vector3f point_e = arrayToEigen(point);
        for (size_t i = 0; i < vertices.size(); i++)
        {
            Eigen::Vector3f vertex = arrayToEigen(vertices[i]);
            if (isRayIdepth)
                vertex = fromRayIdepthToVertex(vertex);
            float distance = (vertex - point_e).norm();
        }
    }

    void transform(Sophus::SE3f &pose)
    {

        for (size_t i = 0; i < vertices.size(); i++)
        {
            Eigen::Vector3f pos;
            if (isRayIdepth)
                pos = fromRayIdepthToVertex(vertices[i].position);
            else
                pos = vertices[i].position;
            pos = pose * pos;

            if (isRayIdepth)
                vertices[i].position = fromVertexToRayIdepth(pos);
            else
                vertices[i].position = pos;
        }
    }

    void project(camera &cam, int lvl)
    {
        for (size_t i = 0; i < vertices.size(); i++)
        {
            Eigen::Vector3f ray;
            if (isRayIdepth)
                ray = vertices[i].position;
            else
                ray = vertices[i].position / vertices[i].position(2);

            ray(2) = 1.0;

            Eigen::Vector2f pix;
            pix(0) = cam.fx[lvl] * ray(0) + cam.cx[lvl];
            pix(1) = cam.fy[lvl] * ray(1) + cam.cy[lvl];

            vertices[i].texcoord = pix;
        }
    }

    Mesh getObservedMesh(Sophus::SE3f &pose, camera &cam)
    {
        int lvl = 0;

        Mesh obsMesh = getCopy();
        obsMesh.transform(pose);

        std::vector<unsigned int> vertices_to_remove;
        std::vector<unsigned int> triangles_to_remove;
        for (size_t i = 0; i < obsMesh.vertices.size(); i++)
        {
            bool remove = false;

            Vertex vert = obsMesh.vertices[i];

            if (vert.position[2] <= 0.0)
            {
                remove = true;
            }

            Eigen::Vector2f p = cam.project(vert.position, lvl);
            if (!cam.isPixVisible(p, lvl))
            {
                remove = true;
            }

            if (remove)
            {
                vertices_to_remove.push_back(i);

                std::vector<Triangle *> tris = vert.triangles;

                for (size_t j = 0; j < tris.size(); j++)
                {
                    Triangle *tri = tris[j];
                    triangles_to_remove.push_back(tri->id);
                }
            }
        }

        for (size_t i = 0; i < vertices_to_remove.size(); i++)
        {
            obsMesh.vertices.erase(vertices_to_remove[i]);
        }

        for (size_t i = 0; i < triangles_to_remove.size(); i++)
        {
            obsMesh.triangles.erase(triangles_to_remove[i]);
        }

        return obsMesh;
    }

    // scene
    // the vertices, actual data of the scene
    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;

    bool isRayIdepth;

private:
};
