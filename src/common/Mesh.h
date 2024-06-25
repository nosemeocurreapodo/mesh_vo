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
        isRayIdepth = false;
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
                if (id <= 0.0)
                    id = 0.5;

                Eigen::Vector3f point;
                if (isRayIdepth)
                    point = Eigen::Vector3f(ray(0), ray(1), id);
                else
                    point = ray / id;
                Vertice vertex(point, pix, vertices.size());

                vertices.push_back(vertex);
            }
        }

        buildTriangles(cam, lvl);
        return;

        // init scene indices
        for (int y = 0; y < MESH_HEIGHT; y++)
        {
            for (int x = 0; x < MESH_WIDTH; x++)
            {
                if (x > 0 && y > 0)
                {
                    // if (((x % 2 == 0)))
                    //  if(((x % 2 == 0) && (y % 2 == 0)) || ((x % 2 != 0) && (y % 2 != 0)))
                    // if (rand() > 0.5 * RAND_MAX)
                    if (true)
                    {
                        unsigned int v11 = x - 1 + y * (MESH_WIDTH);
                        unsigned int v12 = x + (y - 1) * (MESH_WIDTH);
                        unsigned int v13 = x - 1 + (y - 1) * (MESH_WIDTH);

                        unsigned int t1_index = triangles.size();
                        Triangle tri1(vertices[v11], vertices[v12], vertices[v13], t1_index);
                        triangles.push_back(tri1);

                        // Triangle* tri11 = &triangles[t1_index];
                        // vertices[v11].triangles.push_back(tri11);
                        // vertices[v12].triangles.push_back(tri11);
                        // vertices[v13].triangles.push_back(tri11);

                        unsigned int v21 = x + y * (MESH_WIDTH);
                        unsigned int v22 = x + (y - 1) * (MESH_WIDTH);
                        unsigned int v23 = x - 1 + y * (MESH_WIDTH);

                        unsigned int t2_index = triangles.size();
                        Triangle tri2(vertices[v21], vertices[v22], vertices[v23], t2_index);
                        triangles.push_back(tri2);

                        // Triangle* tri22 = &triangles[t2_index];
                        // vertices[v21].triangles.push_back(tri22);
                        // vertices[v22].triangles.push_back(tri22);
                        // vertices[v23].triangles.push_back(tri22);
                    }
                    else
                    {
                        unsigned int v11 = x + y * (MESH_WIDTH);
                        unsigned int v12 = x - 1 + (y - 1) * (MESH_WIDTH);
                        unsigned int v13 = x - 1 + y * (MESH_WIDTH);

                        unsigned int t1_index = triangles.size();
                        Triangle tri1(vertices[v11], vertices[v12], vertices[v13], t1_index);
                        triangles.push_back(tri1);

                        // vertices[v11].triangles.push_back(&triangles[t1_index]);
                        // vertices[v12].triangles.push_back(&triangles[t1_index]);
                        // vertices[v13].triangles.push_back(&triangles[t1_index]);

                        unsigned int v21 = x + y * (MESH_WIDTH);
                        unsigned int v22 = x + (y - 1) * (MESH_WIDTH);
                        unsigned int v23 = x - 1 + (y - 1) * (MESH_WIDTH);

                        unsigned int t2_index = triangles.size();
                        Triangle tri2(vertices[v21], vertices[v22], vertices[v23], t2_index);
                        triangles.push_back(tri2);

                        // vertices[v21].triangles.push_back(&triangles[t2_index]);
                        // vertices[v22].triangles.push_back(&triangles[t2_index]);
                        // vertices[v23].triangles.push_back(&triangles[t2_index]);
                    }
                }
            }
        }
    }

    void initr(frameCPU &frame, dataCPU<float> &idepth, camera &cam, int lvl)
    {
        vertices.clear();
        triangles.clear();

        for (int y = 0; y < 2; y++)
        {
            for (int x = 0; x < 2; x++)
            {
                Eigen::Vector2f pix;
                pix[0] = x * (cam.width[lvl] - 1);
                pix[1] = y * (cam.height[lvl] - 1);
                Eigen::Vector3f ray;
                ray(0) = cam.fxinv[lvl] * pix[0] + cam.cxinv[lvl];
                ray(1) = cam.fyinv[lvl] * pix[1] + cam.cyinv[lvl];
                ray(2) = 1.0;
                float id = idepth.get(pix[1], pix[0], lvl);
                if (id <= 0.0)
                    id = 0.5;

                Eigen::Vector3f point;
                if (isRayIdepth)
                    point = Eigen::Vector3f(ray(0), ray(1), id);
                else
                    point = ray / id;
                Vertice vertex(point, pix, vertices.size());

                vertices.push_back(vertex);
            }
        }

        // preallocate scene vertices to zero
        for (int i = 0; i < MESH_HEIGHT * MESH_HEIGHT - 4; i++)
        {
            Eigen::Vector2f pix;
            pix[0] = rand() % cam.width[lvl];
            pix[1] = rand() % cam.height[lvl];
            Eigen::Vector3f ray;
            ray(0) = cam.fxinv[lvl] * pix[0] + cam.cxinv[lvl];
            ray(1) = cam.fyinv[lvl] * pix[1] + cam.cyinv[lvl];
            ray(2) = 1.0;
            float id = idepth.get(pix[1], pix[0], lvl);
            if (id <= 0.0)
                id = 0.5;

            Eigen::Vector3f point;
            if (isRayIdepth)
                point = Eigen::Vector3f(ray(0), ray(1), id);
            else
                point = ray / id;
            Vertice vertex(point, pix, vertices.size());

            vertices.push_back(vertex);
        }

        buildTriangles(cam, lvl);
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

    void toRayIdepth()
    {
        if (!isRayIdepth)
        {
            for (int i = 0; i < (int)vertices.size(); i++)
            {
                vertices[i].position = fromVertexToRayIdepth(vertices[i].position);
            }
            isRayIdepth = true;
        }
    }

    void toVertex()
    {
        if (isRayIdepth)
        {
            for (int i = 0; i < (int)vertices.size(); i++)
            {
                vertices[i].position = fromRayIdepthToVertex(vertices[i].position);
            }
            isRayIdepth = false;
        }
    }

    void transform(Sophus::SE3f &pose)
    {
        for (int i = 0; i < (int)vertices.size(); i++)
        {
            Eigen::Vector3f pos = vertices[i].position;
            if (isRayIdepth)
                pos = fromRayIdepthToVertex(pos);
            pos = pose * pos;
            if (isRayIdepth)
                pos = fromVertexToRayIdepth(pos);
            vertices[i].position = pos;
        }
    }

    void computeTexCoords(camera &cam, int lvl)
    {
        for (int i = 0; i < (int)vertices.size(); i++)
        {
            Eigen::Vector3f ray;
            if (isRayIdepth)
                ray = vertices[i].position;
            else
                ray = vertices[i].position / vertices[i].position(2);

            Eigen::Vector2f pix;
            pix(0) = cam.fx[lvl] * ray(0) + cam.cx[lvl];
            pix(1) = cam.fy[lvl] * ray(1) + cam.cy[lvl];

            vertices[i].texcoord = pix;
        }
    }

    bool isTrianglePresent(Triangle &tri)
    {
        for (int i = 0; i < (int)triangles.size(); i++)
        {
            Triangle tri2 = triangles[i];
            bool isVertThere[3];
            isVertThere[0] = false;
            isVertThere[1] = false;
            isVertThere[2] = false;
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    if (tri.vertices[j]->id == tri2.vertices[k]->id)
                        isVertThere[j] = true;
                }
            }
            if (isVertThere[0] == true && isVertThere[1] == true && isVertThere[2] == true)
                return true;
        }
        return false;
    }

    void buildTriangles(camera &cam, int lvl)
    {
        toVertex();

        triangles.clear();

        dataCPU<int> triIdImage(-1);

        for (int y = 0; y < cam.height[lvl]; y++)
        {
            for (int x = 0; x < cam.width[lvl]; x++)
            {
                Eigen::Vector2f pix(x, y);
                if (triIdImage.get(pix(1), pix(0), lvl) != triIdImage.nodata)
                    continue;

                std::vector<Vertice> verts;

                // search closest
                verts = vertices;
                Vertice closest_vertice = getClosestVertice(verts, pix);
                int closest_vertice_index = getVertexIndexFromId(closest_vertice.id);

                // search second closest
                verts = vertices;
                verts.erase(verts.begin() + closest_vertice_index);
                
                Vertice second_closest_vertice = getClosestVertice(verts, pix);
                int second_closest_vertice_index = getVertexIndexFromId(second_closest_vertice.id);

                //search third closest
                verts = vertices;
                if (second_closest_vertice_index > closest_vertice_index)
                {
                    verts.erase(verts.begin() + second_closest_vertice_index);
                    verts.erase(verts.begin() + closest_vertice_index);
                }
                else
                {
                    verts.erase(verts.begin() + closest_vertice_index);
                    verts.erase(verts.begin() + second_closest_vertice_index);
                }

                Vertice third_closest_vertice = getClosestVertice(verts, pix);
                int third_closest_vertice_index = getVertexIndexFromId(third_closest_vertice.id);

                Triangle tri(vertices[closest_vertice_index], vertices[second_closest_vertice_index], vertices[third_closest_vertice_index], triangles.size());
                if (isTrianglePresent(tri))
                    continue;

                Eigen::Vector3f ray = cam.toRay(pix, lvl);
                tri.arrageClockwise(ray);
                tri.computeTinv();

                tri.computeBarycentric(pix);
                if(!tri.isBarycentricOk())
                    continue;

                // rasterize triangle, so we know which pixels are already taken by a triangle
                bool otherTriangle = false;
                std::array<Eigen::Vector2f, 2> minmax = tri.getMinMax();
                for (int py = minmax[0](1); py <= minmax[1](1); py++)
                {
                    if(otherTriangle)
                        break;
                    for (int px = minmax[0](0); px <= minmax[1](0); px++)
                    {
                        Eigen::Vector2f ppix = Eigen::Vector2f(px, py);
                        if (!cam.isPixVisible(ppix, lvl))
                            continue;

                        tri.computeBarycentric(ppix);
                        if (!tri.isBarycentricOk())
                            continue;

                        if(triIdImage.get(ppix(1), ppix(0), lvl) != triIdImage.nodata)
                        {
                            otherTriangle = true;
                            break;
                        }
                    }
                }

                if(otherTriangle)
                    continue;

                triangles.push_back(tri);

                // rasterize triangle, so we know which pixels are already taken by a triangle

                for (int py = minmax[0](1); py <= minmax[1](1); py++)
                {
                    for (int px = minmax[0](0); px <= minmax[1](0); px++)
                    {
                        Eigen::Vector2f ppix = Eigen::Vector2f(px, py);
                        if (!cam.isPixVisible(ppix, lvl))
                            continue;

                        tri.computeBarycentric(ppix);
                        if (!tri.isBarycentricOk())
                            continue;

                        triIdImage.set(tri.id, ppix(1), ppix(0), lvl);
                    }
                }
            }
        }

        return;

        for (int i = 0; i < (int)vertices.size(); i++)
        {
            Vertice vert = vertices[i];

            /*
            Eigen::Rotation2Df rot(2.0*3.14*1.0/8.0);
            Eigen::Vector2f shifts[8];
            shifts[0] = Eigen::Rotation2Df(2.0*3.14*1.0/16.0)*Eigen::Vector2f(3, 0);
            shifts[1] = rot*shifts[0];
            shifts[2] = rot*shifts[1];
            shifts[3] = rot*shifts[2];
            shifts[4] = rot*shifts[3];
            shifts[5] = rot*shifts[4];
            shifts[6] = rot*shifts[5];
            shifts[7] = rot*shifts[6];
            */

            float pixel_shift = 3.0;
            Eigen::Vector2f shifts[8];
            shifts[0] = Eigen::Vector2f(pixel_shift, 1.0);
            shifts[1] = Eigen::Vector2f(pixel_shift, pixel_shift);
            shifts[2] = Eigen::Vector2f(1.0, pixel_shift);
            shifts[3] = Eigen::Vector2f(-pixel_shift, pixel_shift);
            shifts[4] = Eigen::Vector2f(-pixel_shift, 1.0);
            shifts[5] = Eigen::Vector2f(-pixel_shift, -pixel_shift);
            shifts[6] = Eigen::Vector2f(1.0, -pixel_shift);
            shifts[7] = Eigen::Vector2f(pixel_shift, -pixel_shift);

            // check for triangles moving slighly in the 4 directions
            for (int j = 0; j < 8; j++)
            {
                Vertice vert_shifted = vert;
                vert_shifted.texcoord += shifts[j];
                if (!cam.isPixVisible(vert_shifted.texcoord, lvl))
                    continue;

                if (triIdImage.get(int(vert_shifted.texcoord(1)), int(vert_shifted.texcoord(0)), lvl) != triIdImage.nodata)
                    continue;

                // search closest
                std::vector<Vertice> verts = vertices;

                verts.erase(verts.begin() + i);
                Vertice closest_vertice = getClosestVertice(verts, vert_shifted.texcoord);
                int closest_vertice_index = getVertexIndexFromId(closest_vertice.id);

                // search second closest
                verts = vertices;
                if (closest_vertice_index > i)
                {
                    verts.erase(verts.begin() + closest_vertice_index);
                    verts.erase(verts.begin() + i);
                }
                else
                {
                    verts.erase(verts.begin() + i);
                    verts.erase(verts.begin() + closest_vertice_index);
                }

                Vertice second_closest_vertice = getClosestVertice(verts, vert_shifted.texcoord);
                int second_closest_vertice_index = getVertexIndexFromId(second_closest_vertice.id);

                Triangle tri(vertices[i], vertices[closest_vertice_index], vertices[second_closest_vertice_index], triangles.size());
                if (isTrianglePresent(tri))
                    continue;

                Eigen::Vector3f ray = cam.toRay(vert_shifted.texcoord, lvl);
                tri.arrageClockwise(ray);
                triangles.push_back(tri);

                // rasterize triangle, so we know which pixels are already taken by a triangle
                tri.computeTinv();

                std::array<Eigen::Vector2f, 2> minmax = tri.getMinMax();

                for (int y = minmax[0](1); y <= minmax[1](1); y++)
                {
                    for (int x = minmax[0](0); x <= minmax[1](0); x++)
                    {
                        Eigen::Vector2f pix = Eigen::Vector2f(x, y);
                        if (!cam.isPixVisible(pix, lvl))
                            continue;

                        tri.computeBarycentric(pix);
                        if (!tri.isBarycentricOk())
                            continue;

                        triIdImage.set(tri.id, pix(1), pix(0), lvl);
                    }
                }
            }
        }
    }

    void buildTriangles(std::vector<Triangle> &tris)
    {
        std::vector<Triangle> new_triangles;
        for (int i = 0; i < (int)tris.size(); i++)
        {
            Triangle tri = tris[i];

            unsigned int v1_in = getVertexIndexFromId(tri.vertices[0]->id);
            unsigned int v2_in = getVertexIndexFromId(tri.vertices[1]->id);
            unsigned int v3_in = getVertexIndexFromId(tri.vertices[2]->id);

            if (v1_in < 0 || v2_in < 0 || v3_in < 0)
                continue;

            unsigned int id = tri.id;

            Triangle new_tri(vertices[v1_in], vertices[v2_in], vertices[v3_in], id);
            new_triangles.push_back(new_tri);

            // vertices[v1_in].triangles.push_back(&new_tri);
            // vertices[v1_in].triangles.push_back(&new_tri);
            // vertices[v1_in].triangles.push_back(&new_tri);
        }
        triangles = new_triangles;
    }

    Mesh getCopy()
    {
        Mesh meshCopy;

        meshCopy.vertices = vertices;
        meshCopy.buildTriangles(triangles);
        meshCopy.isRayIdepth = isRayIdepth;

        return meshCopy;
    }

    Mesh getObservedMesh(Sophus::SE3f &pose, camera &cam)
    {
        int lvl = 0;

        Mesh frameMesh = getCopy();
        frameMesh.transform(pose);
        frameMesh.computeTexCoords(cam, lvl);

        Mesh observedMesh;

        for (int i = 0; i < (int)frameMesh.vertices.size(); i++)
        {
            Vertice vert = frameMesh.vertices[i];
            if (vert.position(2) <= 0)
                continue;
            if (!cam.isPixVisible(vert.texcoord, lvl))
                continue;
            observedMesh.vertices.push_back(Vertice(vert.position, vert.texcoord, vert.id));
        }

        observedMesh.buildTriangles(frameMesh.triangles);
        observedMesh.isRayIdepth = frameMesh.isRayIdepth;

        return observedMesh;
    }

    Mesh completeMesh(Sophus::SE3f &pose, dataCPU<float> &idepth, camera &cam, int lvl)
    {
        Mesh frameMesh = getCopy();
        frameMesh.transform(pose);
        frameMesh.computeTexCoords(cam, lvl);

        int vertexToAdd = int(idepth.getPercentNoData(lvl) * MESH_HEIGHT * MESH_WIDTH);

        for (int v = 0; v < vertexToAdd; v++)
        {
            while (true)
            {
                int x = rand() % cam.width[lvl];
                int y = rand() % cam.height[lvl];

                Eigen::Vector2f pix(x, y);

                float id = idepth.get(pix(1), pix(0), lvl);
                if (id < 0)
                {
                    // get closest triangle
                    float closest_distance = std::numeric_limits<float>::max();
                    int closest_index = -1;
                    for (int t = 0; t < (int)frameMesh.triangles.size(); t++)
                    {
                        Triangle tri = frameMesh.triangles[t];
                        Eigen::Vector2f tri_mean = tri.getMeanTexCoord();
                        float distance = (tri_mean - pix).norm();
                        if (distance < closest_distance)
                        {
                            closest_distance = distance;
                            closest_index = t;
                        }
                    }

                    // compute depth extrapolating from triangle
                    Triangle closest_tri = frameMesh.triangles[closest_index];
                    Eigen::Vector3f ray = cam.toRay(pix, lvl);
                    float pix_depth = closest_tri.vertices[0]->position.dot(closest_tri.getNormal()) / ray.dot(closest_tri.getNormal());

                    // get the 2 closest points in the triangle
                    std::vector<unsigned int> closest_points = {0, 1, 2};
                    float farthest_distance = 0;
                    int farthest_index = -1;
                    for (int t = 0; t < 3; t++)
                    {
                        float distance = (pix - closest_tri.vertices[t]->texcoord).norm();
                        if (distance > farthest_distance)
                        {
                            farthest_distance = distance;
                            farthest_index = t;
                        }
                    }
                    closest_points.erase(closest_points.begin() + farthest_index);

                    Eigen::Vector3f point = ray * pix_depth;
                    Vertice new_vertice(point, pix, (unsigned int)frameMesh.vertices.size());
                    frameMesh.vertices.push_back(new_vertice);

                    Triangle new_triangle(frameMesh.vertices[closest_points[0]], frameMesh.vertices[closest_points[1]], new_vertice, frameMesh.triangles.size());

                    // sceneMesh.vertices.push_back(vertex);
                    // sceneMesh.triangle.push_back()

                    break;
                }
            }
        }
    }

    // scene
    // Triangles contains basically pointers to the corresponding 3 vertices
    std::vector<Triangle> triangles;
    // the vertices, actual data of the scene
    // the vector container can change the pointer if we add or remove elements
    // so we have to be carefull
    std::vector<Vertice> vertices;

    bool isRayIdepth;

private:
};
