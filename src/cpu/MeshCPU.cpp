#include "cpu/MeshCPU.h"

MeshCPU::MeshCPU()
{
    isRayIdepth = false;
};

void MeshCPU::init(frameCPU &frame, dataCPU<float> &idepth, camera &cam, int lvl)
{
    vertices.clear();
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
            VerticeCPU vertex(point, pix);

            vertices[vertices.size()] = vertex;
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
                    tri1[0] = x - 1 + y * (MESH_WIDTH);
                    tri1[1] = x + (y - 1) * (MESH_WIDTH);
                    tri1[2] = x - 1 + (y - 1) * (MESH_WIDTH);

                    triangles[triangles.size()] = tri1;

                    std::array<unsigned int, 3> tri2;
                    tri2[0] = x + y * (MESH_WIDTH);
                    tri2[1] = x + (y - 1) * (MESH_WIDTH);
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
            VerticeCPU vertex(point, pix);

            vertices[vertices.size()] = vertex;
        }
    }

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
        VerticeCPU vertex(point, pix);

        vertices[vertices.size()] = vertex;
    }

    buildTriangles(cam, lvl);
}

void MeshCPU::buildTriangles(camera &cam, int lvl)
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

            MeshCPU copyMesh = getCopy();

            // search closest
            unsigned int closest_vertice_id = copyMesh.getClosestVerticeId(pix);

            // search second closest
            copyMesh.vertices.erase(closest_vertice_id);
            unsigned int second_closest_vertice_id = copyMesh.getClosestVerticeId(pix);

            // search third closest
            copyMesh.vertices.erase(second_closest_vertice_id);
            unsigned int third_closest_vertice_id = copyMesh.getClosestVerticeId(pix);

            std::array<unsigned int, 3> tri = {closest_vertice_id, second_closest_vertice_id, third_closest_vertice_id};
            if (isTrianglePresent(tri))
                continue;

            TriangleCPU ttri(vertices[closest_vertice_id], vertices[second_closest_vertice_id], vertices[third_closest_vertice_id]);

            // arrange clockwise
            Eigen::Vector3f ray = cam.toRay(pix, lvl);
            if (ray.dot(ttri.getNormal()) <= 0)
            {
                tri[1] = third_closest_vertice_id;
                tri[2] = second_closest_vertice_id;
            }

            ttri.computeTinv();

            ttri.computeBarycentric(pix);
            if (!ttri.isBarycentricOk())
                continue;

            // rasterize triangle, so we know which pixels are already taken by a triangle
            bool otherTriangle = false;
            std::array<Eigen::Vector2f, 2> minmax = ttri.getMinMax();
            for (int py = minmax[0](1); py <= minmax[1](1); py++)
            {
                if (otherTriangle)
                    break;
                for (int px = minmax[0](0); px <= minmax[1](0); px++)
                {
                    Eigen::Vector2f ppix = Eigen::Vector2f(px, py);
                    if (!cam.isPixVisible(ppix, lvl))
                        continue;

                    ttri.computeBarycentric(ppix);
                    if (!ttri.isBarycentricOk())
                        continue;

                    if (triIdImage.get(ppix(1), ppix(0), lvl) != triIdImage.nodata)
                    {
                        otherTriangle = true;
                        break;
                    }
                }
            }

            if (otherTriangle)
                continue;

            triangles[triangles.size()] = tri;

            // rasterize triangle, so we know which pixels are already taken by a triangle

            for (int py = minmax[0](1); py <= minmax[1](1); py++)
            {
                for (int px = minmax[0](0); px <= minmax[1](0); px++)
                {
                    Eigen::Vector2f ppix = Eigen::Vector2f(px, py);
                    if (!cam.isPixVisible(ppix, lvl))
                        continue;

                    ttri.computeBarycentric(ppix);
                    if (!ttri.isBarycentricOk())
                        continue;

                    triIdImage.set(triangles.size(), ppix(1), ppix(0), lvl);
                }
            }
        }
    }
}

void MeshCPU::toRayIdepth()
{
    if (!isRayIdepth)
    {
        for (std::map<unsigned int, VerticeCPU>::iterator it = vertices.begin(); it != vertices.end(); ++it)
        {
            it->second.position = fromVertexToRayIdepth(it->second.position);
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
        for (std::map<unsigned int, VerticeCPU>::iterator it = vertices.begin(); it != vertices.end(); ++it)
        {
            it->second.position = fromRayIdepthToVertex(it->second.position);
            // key.push_back(it->first);
            // value.push_back(it->second);
            // std::cout << "Key: " << it->first << std::endl;
            // std::cout << "Value: " << it->second << std::endl;
        }
        isRayIdepth = false;
    }
}

void MeshCPU::transform(Sophus::SE3f &pose)
{
    for (std::map<unsigned int, VerticeCPU>::iterator it = vertices.begin(); it != vertices.end(); ++it)
    {
        Eigen::Vector3f pos = it->second.position;
        if (isRayIdepth)
            pos = fromRayIdepthToVertex(pos);
        pos = pose * pos;
        if (isRayIdepth)
            pos = fromVertexToRayIdepth(pos);
        it->second.position = pos;
    }
}

void MeshCPU::computeTexCoords(camera &cam, int lvl)
{
    for (std::map<unsigned int, VerticeCPU>::iterator it = vertices.begin(); it != vertices.end(); ++it)
    {
        Eigen::Vector3f ray;
        if (isRayIdepth)
            ray = it->second.position;
        else
            ray = it->second.position / it->second.position(2);

        Eigen::Vector2f pix;
        pix(0) = cam.fx[lvl] * ray(0) + cam.cx[lvl];
        pix(1) = cam.fy[lvl] * ray(1) + cam.cy[lvl];

        it->second.texcoord = pix;
    }
}

void MeshCPU::computeTexCoords(Sophus::SE3f &pose, camera &cam, int lvl)
{
    for (std::map<unsigned int, VerticeCPU>::iterator it = vertices.begin(); it != vertices.end(); ++it)
    {
        Eigen::Vector3f ray;
        if (isRayIdepth)
        {
            ray = fromRayIdepthToVertex(it->second.position);
            ray(2) = 1.0;
        }
        else
        {
            Eigen::Vector3f pos = it->second.position;
            Eigen::Vector3f ray = pos / pos(2);
        }

        Eigen::Vector2f pix;
        pix(0) = cam.fx[lvl] * ray(0) + cam.cx[lvl];
        pix(1) = cam.fy[lvl] * ray(1) + cam.cy[lvl];

        it->second.texcoord = pix;
    }
}

bool MeshCPU::isTrianglePresent(std::array<unsigned int, 3> &tri)
{
    for (std::map<unsigned int, std::array<unsigned int, 3>>::iterator it = triangles.begin(); it != triangles.end(); ++it)
    {
        std::array<unsigned int, 3> tri2 = it->second;
        bool isVertThere[3];
        isVertThere[0] = false;
        isVertThere[1] = false;
        isVertThere[2] = false;
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                if (tri[j] == tri2[k])
                    isVertThere[j] = true;
            }
        }
        if (isVertThere[0] == true && isVertThere[1] == true && isVertThere[2] == true)
            return true;
    }
    return false;
}

unsigned int MeshCPU::getClosestTriangleId(Eigen::Vector3f &pos)
{
    float closest_distance = std::numeric_limits<float>::max();
    unsigned int closest_id = 0;
    for (std::map<unsigned int, std::array<unsigned int, 3>>::iterator it = triangles.begin(); it != triangles.end(); ++it)
    {
        TriangleCPU tri = getTriangleStructure(it->first);
        Eigen::Vector3f tri_mean = tri.getMeanPosition();
        float distance = (tri_mean - pos).norm();
        if (distance < closest_distance)
        {
            closest_distance = distance;
            closest_id = it->first;
        }
    }
    return closest_id;
}

unsigned int MeshCPU::getClosestTriangleId(Eigen::Vector2f &tex)
{
    float closest_distance = std::numeric_limits<float>::max();
    unsigned int closest_id = 0;
    for (std::map<unsigned int, std::array<unsigned int, 3>>::iterator it = triangles.begin(); it != triangles.end(); ++it)
    {
        TriangleCPU tri = getTriangleStructure(it->first);
        Eigen::Vector2f tri_mean = tri.getMeanTexCoord();
        float distance = (tri_mean - tex).norm();
        if (distance < closest_distance)
        {
            closest_distance = distance;
            closest_id = it->first;
        }
    }
    return closest_id;
}

unsigned int MeshCPU::getClosestVerticeId(Eigen::Vector2f &pix)
{
    float closest_distance = std::numeric_limits<float>::max();
    unsigned int closest_d = 0;
    for (std::map<unsigned int, VerticeCPU>::iterator it = vertices.begin(); it != vertices.end(); ++it)
    {
        float distance = (it->second.texcoord - pix).norm();

        // float x_distance = std::fabs(v_vector[i].texcoord(0) - v.texcoord(0));
        // float y_distance = std::fabs(v_vector[i].texcoord(1) - v.texcoord(1));
        // float distance = std::max(x_distance, y_distance);

        if (distance < closest_distance)
        {
            closest_distance = distance;
            closest_d = it->first;
        }
    }
    return closest_d;
}

unsigned int MeshCPU::getClosestVerticeId(Eigen::Vector3f &v)
{
    float closest_distance = std::numeric_limits<float>::max();
    unsigned int closest_vertice = 0;
    for (std::map<unsigned int, VerticeCPU>::iterator it = vertices.begin(); it != vertices.end(); ++it)
    {
        float distance = (it->second.position - v).norm();
        if (distance < closest_distance)
        {
            closest_distance = distance;
            closest_vertice = it->first;
        }
    }
    return closest_vertice;
}
