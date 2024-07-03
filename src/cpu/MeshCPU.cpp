#include "cpu/MeshCPU.h"

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
            pix[0] = float(x) / (MESH_WIDTH - 1);
            pix[1] = float(y) / (MESH_HEIGHT - 1);
            // pix[0] = rand() % cam.width[lvl];
            // pix[1] = rand() % cam.height[lvl];
            // pix[0] = rand() % cam.width[lvl] * 0.75 + cam.width[lvl] * 0.125;
            // pix[1] = rand() % cam.height[lvl] * 0.75 + cam.height[lvl] * 0.125;
            // pix[0] = (float(x)/(MESH_WIDTH-1)) * cam.width[lvl] / 2.0 + cam.width[lvl] / 4.0;
            // pix[1] = (float(y)/(MESH_HEIGHT-1)) * cam.height[lvl] / 2.0 + cam.height[lvl] / 4.0;
            Eigen::Vector3f ray = cam.pixToRayNormalized(pix);
            float id = idepth.getNormalized(pix[1], pix[0], lvl);
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

    buildTriangles(cam);
}

void MeshCPU::setVerticeIdepth(float idepth, unsigned int id)
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

Eigen::Vector3f MeshCPU::getVertice(unsigned int id)
{
    return vertices[id];
}

Eigen::Vector2f &MeshCPU::getTexCoord(unsigned int id)
{
    return texcoords[id];
}

std::array<unsigned int, 3> MeshCPU::getTriangleIndices(unsigned int id)
{
    return triangles[id];
}

Triangle2D MeshCPU::getTriangle2D(unsigned int id)
{
    std::array<unsigned int, 3> tri = triangles[id];
    Triangle2D t(texcoords[tri[0]], texcoords[tri[1]], texcoords[tri[2]]);
    return t;
}

Triangle3D MeshCPU::getTriangle3D(unsigned int id)
{
    std::array<unsigned int, 3> tri = triangles[id];
    Triangle3D t(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]);
    return t;
}

unsigned int MeshCPU::addVertice(Eigen::Vector3f &vert)
{
    unsigned int v_id = 0;
    for (auto vert : vertices)
    {
        if (vert.first > v_id)
            v_id = vert.first;
    }
    v_id++;
    vertices[v_id] = vert;
    return v_id;
}

unsigned int MeshCPU::addVertice(Eigen::Vector3f &vert, Eigen::Vector2f &tex)
{
    unsigned int v_id = 0;
    for (auto vert : vertices)
    {
        if (vert.first > v_id)
            v_id = vert.first;
    }
    v_id++;
    vertices[v_id] = vert;
    texcoords[v_id] = tex;
    return v_id;
}

unsigned int MeshCPU::addTriangle(std::array<unsigned int, 3> &tri)
{
    unsigned int t_id = 0;
    for (auto tri : triangles)
    {
        if (tri.first > t_id)
            t_id = tri.first;
    }
    t_id++;
    triangles[t_id] = tri;
    return t_id;
}

MeshCPU MeshCPU::getCopy()
{
    MeshCPU meshCopy;

    meshCopy.vertices = vertices;
    meshCopy.triangles = triangles;
    meshCopy.isRayIdepth = isRayIdepth;

    return meshCopy;
}

std::vector<unsigned int> MeshCPU::getVerticesIds()
{
    std::vector<unsigned int> keys;
    for (auto it = vertices.begin(); it != vertices.end(); ++it)
    {
        keys.push_back(it->first);
    }
    return keys;
}

std::vector<unsigned int> MeshCPU::getTrianglesIds()
{
    std::vector<unsigned int> keys;
    for (auto it = triangles.begin(); it != triangles.end(); ++it)
    {
        keys.push_back(it->first);
    }
    return keys;
}

void MeshCPU::computeTexCoords(camera &cam)
{
    texcoords.clear();
    for (auto it = vertices.begin(); it != vertices.end(); ++it)
    {
        Eigen::Vector3f ray;
        if (isRayIdepth)
            ray = it->second;
        else
            ray = it->second / it->second(2);

        Eigen::Vector2f pix = cam.rayToPix(ray);

        texcoords[it->first] = pix;
    }
}

void MeshCPU::computeNormalizedTexCoords(camera &cam)
{
    texcoords.clear();
    for (auto it = vertices.begin(); it != vertices.end(); ++it)
    {
        Eigen::Vector3f ray;
        if (isRayIdepth)
            ray = it->second;
        else
            ray = it->second / it->second(2);

        Eigen::Vector2f pix = cam.rayToPix(ray);

        texcoords[it->first] = pix;
    }
}

std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> MeshCPU::computeEdgeFront()
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

std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> MeshCPU::getSortedEdgeFront(Eigen::Vector2f &pix)
{
    std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> sortedEdgeFront;

    std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> edgeFront = computeEdgeFront();

    while (edgeFront.size() > 0)
    {
        size_t closest;
        float closest_distance = std::numeric_limits<float>::max();
        for (auto it = edgeFront.begin(); it != edgeFront.end(); it++)
        {
            std::array<unsigned int, 2> edge = it->first;
            Eigen::Vector2f e1 = texcoords[edge[0]];
            Eigen::Vector2f e2 = texcoords[edge[1]];
            float distance = ((e1 + e2) / 2.0 - pix).norm();
            if (distance < closest_distance)
            {
                closest_distance = distance;
                closest = it - edgeFront.begin();
            }
        }
        sortedEdgeFront.push_back(edgeFront[closest]);
        edgeFront.erase(edgeFront.begin() + closest);
    }

    return sortedEdgeFront;
}

std::vector<unsigned int> MeshCPU::getSortedTriangles(Eigen::Vector2f &pix)
{
    std::vector<unsigned int> sortedTriangles;
    std::vector<unsigned int> trisIds = getTrianglesIds();

    while (trisIds.size() > 0)
    {
        size_t closestIndex;
        float closest_distance = std::numeric_limits<float>::max();
        for (auto it = trisIds.begin(); it != trisIds.end(); it++)
        {
            Triangle2D tri = getTriangle2D(*it);
            float distance = (tri.getMean() - pix).norm();
            if (distance < closest_distance)
            {
                closest_distance = distance;
                closestIndex = it - trisIds.begin();
            }
        }
        sortedTriangles.push_back(trisIds[closestIndex]);
        trisIds.erase(trisIds.begin() + closestIndex);
    }

    return sortedTriangles;
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