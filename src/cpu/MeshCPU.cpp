#include "cpu/MeshCPU.h"
#include "common/DelaunayTriangulation.h"

MeshCPU::MeshCPU()
{
    isRayIdepth = false;
};

void MeshCPU::init(frameCPU &frame, dataCPU<float> &idepth, camera &cam, int lvl)
{
    vertices.clear();
    texcoords.clear();
    triangles.clear();

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

            vertices[vertices.size()] = point;
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
                // if (rand() > 0.5 * RAND_MAX)
                if (true)
                {
                    std::array<unsigned int, 3> tri1;
                    tri1[1] = x - 1 + y * (MESH_WIDTH);
                    tri1[0] = x + (y - 1) * (MESH_WIDTH);
                    tri1[2] = x - 1 + (y - 1) * (MESH_WIDTH);

                    triangles[triangles.size()] = tri1;

                    std::array<unsigned int, 3> tri2;
                    tri2[1] = x + y * (MESH_WIDTH);
                    tri2[0] = x + (y - 1) * (MESH_WIDTH);
                    tri2[2] = x - 1 + y * (MESH_WIDTH);

                    triangles[triangles.size()] = tri2;
                }
                else
                {
                    std::array<unsigned int, 3> tri1;
                    tri1[0] = x + y * (MESH_WIDTH);
                    tri1[1] = x - 1 + (y - 1) * (MESH_WIDTH);
                    tri1[2] = x - 1 + y * (MESH_WIDTH);

                    triangles[triangles.size()] = tri1;

                    std::array<unsigned int, 3> tri2;
                    tri2[0] = x + y * (MESH_WIDTH);
                    tri2[1] = x + (y - 1) * (MESH_WIDTH);
                    tri2[2] = x - 1 + (y - 1) * (MESH_WIDTH);

                    triangles[triangles.size()] = tri2;
                }
            }
        }
    }
}

void MeshCPU::initr(frameCPU &frame, dataCPU<float> &idepth, camera &cam, int lvl)
{
    vertices.clear();
    texcoords.clear();
    triangles.clear();

    for (int y = 0; y < MESH_HEIGHT; y++)
    {
        for (int x = 0; x < MESH_WIDTH; x++)
        {
            Eigen::Vector2f pix;
            pix[0] = float(x)/(MESH_WIDTH-1)*cam.width[lvl];
            pix[1] = float(y)/(MESH_HEIGHT-1)*cam.height[lvl];
            //pix[0] = rand() % cam.width[lvl];
            //pix[1] = rand() % cam.height[lvl];
            //pix[0] = rand() % cam.width[lvl] * 0.75 + cam.width[lvl] * 0.125;
            //pix[1] = rand() % cam.height[lvl] * 0.75 + cam.height[lvl] * 0.125;
            //pix[0] = (float(x)/(MESH_WIDTH-1)) * cam.width[lvl] / 2.0 + cam.width[lvl] / 4.0;
            //pix[1] = (float(y)/(MESH_HEIGHT-1)) * cam.height[lvl] / 2.0 + cam.height[lvl] / 4.0;
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

            vertices[vertices.size()] = point;
        }
    }
    
    computeTexCoords(cam, lvl);
    DelaunayTriangulation triangulation;
    triangulation.loadPoints(texcoords);
    triangulation.triangulate();
    triangles = triangulation.getTriangles();
}

void MeshCPU::toRayIdepth()
{
    if (!isRayIdepth)
    {
        for (auto it = vertices.begin(); it != vertices.end(); ++it)
        {
            it->second = fromVertexToRayIdepth(it->second);
            // key.push_back(it->first);
            // value.push_back(it->second);
            // std::cout << "Key: " << it->first << std::endl;
            // std::cout << "Value: " << it->second << std::endl;
        }

        isRayIdepth = true;
    }
}

void MeshCPU::toVertex()
{
    if (isRayIdepth)
    {
        for (auto it = vertices.begin(); it != vertices.end(); ++it)
        {
            it->second = fromRayIdepthToVertex(it->second);
            // key.push_back(it->first);
            // value.push_back(it->second);
            // std::cout << "Key: " << it->first << std::endl;
            // std::cout << "Value: " << it->second << std::endl;
        }
        isRayIdepth = false;
    }
}

void MeshCPU::transform(Sophus::SE3f pose)
{
    for (auto it = vertices.begin(); it != vertices.end(); ++it)
    {
        Eigen::Vector3f pos = it->second;
        if (isRayIdepth)
            pos = fromRayIdepthToVertex(pos);
        pos = pose * pos;
        if (isRayIdepth)
            pos = fromVertexToRayIdepth(pos);
        it->second = pos;
    }
}

bool MeshCPU::isTrianglePresent(std::array<unsigned int, 3> &tri)
{
    for (auto it = triangles.begin(); it != triangles.end(); ++it)
    {
        std::array<unsigned int, 3> tri2 = it->second;

        if (isTriangleEqual(tri, tri2))
            return true;
    }
    return false;
}