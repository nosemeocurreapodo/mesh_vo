#include "cpu/MeshCPU.h"


std::array<unsigned int, 3> MeshCPU::getTriangleIndices(unsigned int id)
{
    if (!triangles.count(id))
        throw std::out_of_range("getTriangleIndices invalid id");
    return triangles[id];
}

unsigned int MeshCPU::addTriangle(std::array<unsigned int, 3> &tri)
{
    last_t_id++;
    if (triangles.count(last_t_id))
        throw std::out_of_range("addTriangle id already exist");
    triangles[last_t_id] = tri;
    return last_t_id;
}

void MeshCPU::setTriangleIndices(std::array<unsigned int, 3> &tri, unsigned int id)
{
    if (!triangles.count(id))
        throw std::out_of_range("setTriangleIndices invalid id");
    triangles[id] = tri;
}


Polygon MeshCPU::getCartesianTriangle(unsigned int id)
{
    // always return triangle in cartesian
    std::array<unsigned int, 3> tri = getTriangleIndices(id);
    Polygon t(getVertice(tri[0]), getVertice(tri[1]), getVertice(tri[2]));
    return t;
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

void MeshCPU::removePointsWithoutTriangles()
{
    std::vector<unsigned int> vertsIds = getVerticesIds();
    for (auto it = vertsIds.begin(); it != vertsIds.end(); it++)
    {
        bool remove = true;
        for (auto t_it = triangles.begin(); t_it != triangles.end(); t_it++)
        {
            if (*it == t_it->second[0] || *it == t_it->second[1] || *it == t_it->second[2])
            {
                remove = false;
                break;
            }
        }
        if (remove)
            removeVertice(*it);
    }
}

void MeshCPU::removeTrianglesWithoutPoints()
{
    std::vector<unsigned int> trisIds = getTrianglesIds();
    for (auto it = trisIds.begin(); it != trisIds.end(); it++)
    {
        std::array<unsigned int, 3> vertIds = getTriangleIndices(*it);
        if (!vertices.count(vertIds[0]) || !vertices.count(vertIds[1]) || !vertices.count(vertIds[2]))
        {
            triangles.erase(*it);
        }
    }
}
