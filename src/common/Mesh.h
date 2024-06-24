#pragma once

#include "common/Vertice.h"
#include "common/Triangle.h"
#include "common/camera.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "params.h"

class Mesh
{
public:
    Mesh()
    {
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
                pix[0] = cam.width[lvl] * float(x) / float(MESH_WIDTH - 1);
                pix[1] = cam.height[lvl] * float(y) / float(MESH_HEIGHT - 1);
                Eigen::Vector3f ray;
                ray(0) = cam.fxinv[lvl] * pix[0] + cam.cxinv[lvl];
                ray(1) = cam.fyinv[lvl] * pix[1] + cam.cyinv[lvl];
                ray(2) = 1.0;
                float id = idepth.get(pix[1], pix[0], lvl);
                if(id <= 0.0)
                    id = 0.5;

                Eigen::Vector3f point = ray / id;
                Vertice vertex(point, pix, vertices.size());

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
                    //if (rand() > 0.5 * RAND_MAX)
                    if(true)
                    {
                        unsigned int v11 = x - 1 + y * (MESH_WIDTH);
                        unsigned int v12 = x + (y - 1) * (MESH_WIDTH);
                        unsigned int v13 = x - 1 + (y - 1) * (MESH_WIDTH);

                        unsigned int t1_index = triangles.size();
                        Triangle tri1(vertices[v11], vertices[v12], vertices[v13], t1_index);
                        triangles.push_back(tri1);

                        //Triangle* tri11 = &triangles[t1_index];
                        //vertices[v11].triangles.push_back(tri11);
                        //vertices[v12].triangles.push_back(tri11);
                        //vertices[v13].triangles.push_back(tri11);

                        unsigned int v21 = x + y * (MESH_WIDTH);
                        unsigned int v22 = x + (y - 1) * (MESH_WIDTH);
                        unsigned int v23 = x - 1 + y * (MESH_WIDTH);

                        unsigned int t2_index = triangles.size();
                        Triangle tri2(vertices[v21], vertices[v22], vertices[v23], t2_index);
                        triangles.push_back(tri2);

                        //Triangle* tri22 = &triangles[t2_index];
                        //vertices[v21].triangles.push_back(tri22);
                        //vertices[v22].triangles.push_back(tri22);
                        //vertices[v23].triangles.push_back(tri22);
                    }
                    else
                    {
                        unsigned int v11 = x + y * (MESH_WIDTH);
                        unsigned int v12 = x - 1 + (y - 1) * (MESH_WIDTH);
                        unsigned int v13 = x - 1 + y * (MESH_WIDTH);

                        unsigned int t1_index = triangles.size();
                        Triangle tri1(vertices[v11], vertices[v12], vertices[v13], t1_index);
                        triangles.push_back(tri1);

                        //vertices[v11].triangles.push_back(&triangles[t1_index]);
                        //vertices[v12].triangles.push_back(&triangles[t1_index]);
                        //vertices[v13].triangles.push_back(&triangles[t1_index]);

                        unsigned int v21 = x + y * (MESH_WIDTH);
                        unsigned int v22 = x + (y - 1) * (MESH_WIDTH);
                        unsigned int v23 = x - 1 + (y - 1) * (MESH_WIDTH);

                        unsigned int t2_index = triangles.size();
                        Triangle tri2(vertices[v21], vertices[v22], vertices[v23], t2_index);
                        triangles.push_back(tri2);

                        //vertices[v21].triangles.push_back(&triangles[t2_index]);
                        //vertices[v22].triangles.push_back(&triangles[t2_index]);
                        //vertices[v23].triangles.push_back(&triangles[t2_index]);
                    }
                }
            }
        }
    }

    int getVertexIndexFromId(unsigned int id)
    {
        for (int i = 0; i < (int)vertices.size(); i++)
        {
            if (id == vertices[i].id)
                return i;
        }
        return -1;
    }

    int getTriangleIndexFromId(unsigned int id)
    {
        for (int i = 0; i < (int)triangles.size(); i++)
        {
            if (id == triangles[i].id)
                return i;
        }
        return -1;
    }

    /*
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
        */

    void transform(Sophus::SE3f &pose)
    {

        for (int i = 0; i < (int)vertices.size(); i++)
        {
            Eigen::Vector3f pos = vertices[i].position;
            pos = pose * pos;
            vertices[i].position = pos;
        }
    }

    void computeTexCoords(camera &cam, int lvl)
    {
        for (int i = 0; i < (int)vertices.size(); i++)
        {
            Eigen::Vector3f ray = vertices[i].position / vertices[i].position(2);

            Eigen::Vector2f pix;
            pix(0) = cam.fx[lvl] * ray(0) + cam.cx[lvl];
            pix(1) = cam.fy[lvl] * ray(1) + cam.cy[lvl];

            vertices[i].texcoord = pix;
        }
    }

    Mesh getCopy()
    {
        Mesh meshCopy;
        for (int i = 0; i < (int)vertices.size(); i++)
        {
            Vertice new_ver(vertices[i].position, vertices[i].texcoord, vertices[i].id);
            meshCopy.vertices.push_back(new_ver);
        }

        for (int i = 0; i < (int)triangles.size(); i++)
        {
            Triangle tri = triangles[i];

            unsigned int v1_in = meshCopy.getVertexIndexFromId(tri.vertices[0]->id);
            unsigned int v2_in = meshCopy.getVertexIndexFromId(tri.vertices[1]->id);
            unsigned int v3_in = meshCopy.getVertexIndexFromId(tri.vertices[2]->id);

            //should not happen
            if(v1_in < 0 || v2_in < 0 || v3_in < 0)
                continue;

            unsigned int id = tri.id;

            Triangle new_tri(meshCopy.vertices[v1_in], meshCopy.vertices[v2_in], meshCopy.vertices[v3_in], id);
            meshCopy.triangles.push_back(new_tri);

            //vertices[v1_in].triangles.push_back(&new_tri);
            //vertices[v1_in].triangles.push_back(&new_tri);
            //vertices[v1_in].triangles.push_back(&new_tri);
        }

        return meshCopy;
    }

    Mesh geObservedMesh(Sophus::SE3f &pose, camera &cam)
    {
        int lvl = 0;

        Mesh frameMesh = getCopy();
        frameMesh.transform(pose);
        frameMesh.computeTexCoords(cam, lvl);

        Mesh observedMesh;

        for (int i = 0; i < (int)frameMesh.vertices.size(); i++)
        {
            Vertice vert = frameMesh.vertices[i];
            if(vert.position(2) <= 0)
                continue;
            if(!cam.isPixVisible(vert.texcoord, lvl))
                continue;
            observedMesh.vertices.push_back(Vertice(vert.position, vert.texcoord, vert.id));
        }

        for (int i = 0; i < (int)frameMesh.triangles.size(); i++)
        {
            Triangle tri = triangles[i];

            unsigned int v1_in = observedMesh.getVertexIndexFromId(tri.vertices[0]->id);
            unsigned int v2_in = observedMesh.getVertexIndexFromId(tri.vertices[1]->id);
            unsigned int v3_in = observedMesh.getVertexIndexFromId(tri.vertices[2]->id);

            if(v1_in < 0 || v2_in < 0 || v3_in < 0)
                continue;

            unsigned int id = tri.id;

            Triangle new_tri(observedMesh.vertices[v1_in], observedMesh.vertices[v2_in], observedMesh.vertices[v3_in], id);
            observedMesh.triangles.push_back(new_tri);

            //vertices[v1_in].triangles.push_back(&new_tri);
            //vertices[v1_in].triangles.push_back(&new_tri);
            //vertices[v1_in].triangles.push_back(&new_tri);
        }

        return observedMesh;
    }

    // scene
    // the vertices, actual data of the scene
    std::vector<Vertice> vertices;
    std::vector<Triangle> triangles;

    bool isRayIdepth;

private:
};
