#pragma once

#include "common/Vertex.h"
#include "common/Triangle.h"
#include "params.h"

class Mesh
{
public:
    Mesh(){

    };

    void init()
    {
        vertices.clear();
        triangles.clear();

        // preallocate scene vertices to zero
        for (int y = 0; y < MESH_HEIGHT; y++)
        {
            for (int x = 0; x < MESH_WIDTH; x++)
            {
                Eigen::Vector3f data;
                unsigned int vertex_index = vertices.size();
                Vertex vertex(data, vertex_index);

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
                        std::array<unsigned int, 3> v_id1;
                        v_id1[0] = x - 1 + y * (MESH_WIDTH);
                        v_id1[1] = x + (y - 1) * (MESH_WIDTH);
                        v_id1[2] = x - 1 + (y - 1) * (MESH_WIDTH);

                        unsigned int triangle_index1 = triangles.size();
                        Triangle tri1(v_id1, triangle_index1);
                        triangles.push_back(tri1);

                        vertices[v_id1[0]].addTriangle(triangle_index1);
                        vertices[v_id1[1]].addTriangle(triangle_index1);
                        vertices[v_id1[2]].addTriangle(triangle_index1);

                        std::array<unsigned int, 3> v_id2;
                        v_id2[0] = x + y * (MESH_WIDTH);
                        v_id2[1] = x + (y - 1) * (MESH_WIDTH);
                        v_id2[2] = x - 1 + y * (MESH_WIDTH);

                        unsigned int triangle_index2 = triangles.size();
                        Triangle tri2(v_id2, triangle_index2);
                        triangles.push_back(tri2);

                        vertices[v_id2[0]].addTriangle(triangle_index2);
                        vertices[v_id2[1]].addTriangle(triangle_index2);
                        vertices[v_id2[2]].addTriangle(triangle_index2);
                    }
                    else
                    {
                        std::array<unsigned int, 3> v_id1;
                        v_id1[0] = x + y * (MESH_WIDTH);
                        v_id1[1] = x - 1 + (y - 1) * (MESH_WIDTH);
                        v_id1[2] = x - 1 + y * (MESH_WIDTH);

                        unsigned int triangle_index1 = triangles.size();
                        Triangle tri1(v_id1, triangle_index1);
                        triangles.push_back(tri1);

                        vertices[v_id1[0]].addTriangle(triangle_index1);
                        vertices[v_id1[1]].addTriangle(triangle_index1);
                        vertices[v_id1[2]].addTriangle(triangle_index1);

                        std::array<unsigned int, 3> v_id2;
                        v_id2[0] = x + y * (MESH_WIDTH);
                        v_id2[1] = x + (y - 1) * (MESH_WIDTH);
                        v_id2[2] = x - 1 + (y + 1) * (MESH_WIDTH);

                        unsigned int triangle_index2 = triangles.size();
                        Triangle tri2(v_id2, triangle_index2);
                        triangles.push_back(tri2);

                        vertices[v_id2[0]].addTriangle(triangle_index2);
                        vertices[v_id2[1]].addTriangle(triangle_index2);
                        vertices[v_id2[2]].addTriangle(triangle_index2);
                    }
                }
            }
        }
    }

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
                pix[0] = cam.width[lvl] * float(x)/MESH_WIDTH;
                pix[1] = cam.height[lvl] * float(y)/MESH_HEIGHT;
                Eigen::Vector3f ray;
                ray(0) = cam.fxinv[lvl] * pix[0] + cam.cxinv[lvl];
                ray(1) = cam.fyinv[lvl] * pix[1] + cam.cyinv[lvl];
                ray(2) = 1.0;
                float id = idepth.get(pix[1], pix[0], lvl);
                Eigen::Vector3f point = ray / id;
                
                unsigned int vertex_index = vertices.size();
                Vertex vertex(point, vertex_index);

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
                        std::array<unsigned int, 3> v_id1;
                        v_id1[0] = x - 1 + y * (MESH_WIDTH);
                        v_id1[1] = x + (y - 1) * (MESH_WIDTH);
                        v_id1[2] = x - 1 + (y - 1) * (MESH_WIDTH);

                        unsigned int triangle_index1 = triangles.size();
                        Triangle tri1(v_id1, triangle_index1);
                        triangles.push_back(tri1);

                        vertices[v_id1[0]].addTriangle(triangle_index1);
                        vertices[v_id1[1]].addTriangle(triangle_index1);
                        vertices[v_id1[2]].addTriangle(triangle_index1);

                        std::array<unsigned int, 3> v_id2;
                        v_id2[0] = x + y * (MESH_WIDTH);
                        v_id2[1] = x + (y - 1) * (MESH_WIDTH);
                        v_id2[2] = x - 1 + y * (MESH_WIDTH);

                        unsigned int triangle_index2 = triangles.size();
                        Triangle tri2(v_id2, triangle_index2);
                        triangles.push_back(tri2);

                        vertices[v_id2[0]].addTriangle(triangle_index2);
                        vertices[v_id2[1]].addTriangle(triangle_index2);
                        vertices[v_id2[2]].addTriangle(triangle_index2);
                    }
                    else
                    {
                        std::array<unsigned int, 3> v_id1;
                        v_id1[0] = x + y * (MESH_WIDTH);
                        v_id1[1] = x - 1 + (y - 1) * (MESH_WIDTH);
                        v_id1[2] = x - 1 + y * (MESH_WIDTH);

                        unsigned int triangle_index1 = triangles.size();
                        Triangle tri1(v_id1, triangle_index1);
                        triangles.push_back(tri1);

                        vertices[v_id1[0]].addTriangle(triangle_index1);
                        vertices[v_id1[1]].addTriangle(triangle_index1);
                        vertices[v_id1[2]].addTriangle(triangle_index1);

                        std::array<unsigned int, 3> v_id2;
                        v_id2[0] = x + y * (MESH_WIDTH);
                        v_id2[1] = x + (y - 1) * (MESH_WIDTH);
                        v_id2[2] = x - 1 + (y + 1) * (MESH_WIDTH);

                        unsigned int triangle_index2 = triangles.size();
                        Triangle tri2(v_id2, triangle_index2);
                        triangles.push_back(tri2);

                        vertices[v_id2[0]].addTriangle(triangle_index2);
                        vertices[v_id2[1]].addTriangle(triangle_index2);
                        vertices[v_id2[2]].addTriangle(triangle_index2);
                    }
                }
            }
        }
    }

    Mesh getCopy()
    {
        Mesh meshCopy;
        meshCopy.vertices = vertices;
        meshCopy.triangles = triangles;
        meshCopy.isRayIdepth = isRayIdepth;

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

    Mesh getTransformed(Sophus::SE3f &pose)
    {
        Mesh meshTransformed = getCopy();

        for (size_t i = 0; i < vertices.size(); i++)
        {
            meshTransformed.vertices[i] = pose * meshTransformed.vertices[i];
        }

        return meshTransformed;
    }

    Mesh getProjected(camera &cam, int lvl)
    {
        Mesh meshProjected = getCopy();

        for (size_t i = 0; i < vertices.size(); i++)
        {
            meshProjected.vertices[i] = cam.project(vertices[i], lvl)
        }

        return meshProjected;
    }

    Mesh getObservedMesh(Sophus::SE3f &pose, camera &cam)
    {
        int lvl = 0;

        Mesh obsMesh = getTransformed(pose);

        std::vector<unsigned int> vertices_to_remove;
        std::vector<unsigned int> triangle_to_remove;
        for (size_t = 0; i < obsMesh.vertices.size(); i++)
        {
            bool remove = false;

            Vertex vert = obsMesh.vertices[i];

            if (vert[2] <= 0.0)
            {
                remove = true;
            }

            Eigen::Vector2f p = cam.project(vert());
            if (!cam.isPixVisible(p))
            {
                remove = true;
            }

            if (remove)
            {
                vertices_to_remove.push_back(i);

                std::vector<unsigned int> tris = vert.triangles_indices;

                for (int j = 0; j < 3; j++)
                {
                    triangles_to_remove.push_back(tris[j]);
                }
            }
        }

        for (size_t = 0; i < vertices_to_remove.size(); i++)
        {
            obsMesh.vertices.erase(vertices_to_remove[i]);
        }

        for (size_t = 0; i < triangle_to_remove.size(); i++)
        {
            obsMesh.triangles.erase(triangle_to_remove[i]);
        }

        return obsMesh;
    }

private:
    // scene
    // the vertices, actual data of the scene
    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;

    bool isRayIdepth;
};
