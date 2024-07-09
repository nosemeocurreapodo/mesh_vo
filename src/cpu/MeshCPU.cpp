#include "cpu/MeshCPU.h"

MeshCPU::MeshCPU()
{
    representation = cartesian;
    last_v_id = 0;
    last_t_id = 0;
};

MeshCPU MeshCPU::getCopy()
{
    MeshCPU meshCopy;

    meshCopy.vertices = vertices;
    meshCopy.texcoords = texcoords;
    meshCopy.triangles = triangles;
    meshCopy.globalPose = globalPose;
    meshCopy.representation = representation;
    meshCopy.last_v_id = last_v_id;
    meshCopy.last_t_id = last_t_id;

    return meshCopy;
}

Eigen::Vector3f MeshCPU::getVertice(unsigned int id)
{
    if (!vertices.count(id))
        throw std::out_of_range("getVertice invalid id");
    // always return in cartesian
    if (representation == rayIdepth)
        return rayIdepthToCartesian(vertices[id]);
    return vertices[id];
}

Eigen::Vector2f MeshCPU::getTexCoord(unsigned int id)
{
    if (!texcoords.count(id))
        throw std::out_of_range("getTexCoord invalid id");
    return texcoords[id];
}

std::array<unsigned int, 3> MeshCPU::getTriangleIndices(unsigned int id)
{
    if (!triangles.count(id))
        throw std::out_of_range("getTriangleIndices invalid id");
    return triangles[id];
}

unsigned int MeshCPU::addVertice(Eigen::Vector3f &vert)
{
    // the input vertice is always in cartesian
    last_v_id++;
    if (vertices.count(last_v_id))
        throw std::out_of_range("addVertice id already exist");
    if (representation == rayIdepth)
        vertices[last_v_id] = cartesianToRayIdepth(vert);
    if (representation == cartesian)
        vertices[last_v_id] = vert;
    return last_v_id;
}

unsigned int MeshCPU::addTriangle(std::array<unsigned int, 3> &tri)
{
    last_t_id++;
    if (triangles.count(last_t_id))
        throw std::out_of_range("addTriangle id already exist");
    triangles[last_t_id] = tri;
    return last_t_id;
}

void MeshCPU::setVertice(Eigen::Vector3f &vert, unsigned int id)
{
    if (!vertices.count(id))
        throw std::out_of_range("setVertice invalid id");
    if (representation == rayIdepth)
        vertices[last_v_id] = cartesianToRayIdepth(vert);
    if (representation == cartesian)
        vertices[last_v_id] = vert;
}

void MeshCPU::setTriangleIndices(std::array<unsigned int, 3> &tri, unsigned int id)
{
    if (!triangles.count(id))
        throw std::out_of_range("setTriangleIndices invalid id");
    triangles[id] = tri;
}

void MeshCPU::setVerticeDepth(float depth, unsigned int id)
{
    if (!vertices.count(id))
        throw std::out_of_range("setVerticeDepth invalid id");
    if (representation == rayIdepth)
        vertices[id](2) = 1.0 / depth;
    if (representation == cartesian)
    {
        vertices[id] = depth * vertices[id] / vertices[id](2);
    }
}

float MeshCPU::getVerticeDepth(unsigned int id)
{
    if (!vertices.count(id))
        throw std::out_of_range("getVerticeDepth invalid id");
    if (representation == rayIdepth)
        return 1.0 / vertices[id](2);
    if (representation == cartesian)
        return vertices[id](2);
}

Triangle2D MeshCPU::getTexCoordTriangle(unsigned int id)
{
    std::array<unsigned int, 3> tri = getTriangleIndices(id);
    Triangle2D t(getTexCoord(tri[0]), getTexCoord(tri[1]), getTexCoord(tri[2]));
    return t;
}

Triangle3D MeshCPU::getCartesianTriangle(unsigned int id)
{
    // always return triangle in cartesian
    std::array<unsigned int, 3> tri = getTriangleIndices(id);
    Triangle3D t(getVertice(tri[0]), getVertice(tri[1]), getVertice(tri[2]));
    return t;
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

void MeshCPU::transform(Sophus::SE3f newGlobalPose)
{
    Sophus::SE3f relativePose = newGlobalPose*globalPose.inverse();
    for (auto it = vertices.begin(); it != vertices.end(); ++it)
    {
        Eigen::Vector3f pos = it->second;
        if (representation == rayIdepth)
            pos = rayIdepthToCartesian(pos);
        pos = relativePose * pos;
        if (representation == rayIdepth)
            pos = cartesianToRayIdepth(pos);
        it->second = pos;
    }
    globalPose = newGlobalPose;
}

void MeshCPU::computeTexCoords(camera &cam)
{
    texcoords.clear();
    for (auto it = vertices.begin(); it != vertices.end(); ++it)
    {
        Eigen::Vector3f ray;
        if (representation == rayIdepth)
        {
            ray = it->second;
            ray(2) = 1.0;
        }
        if (representation == cartesian)
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
        if (representation == rayIdepth)
        {
            ray = it->second;
            ray(2) = 1.0;
        }

        if (representation == cartesian)
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

std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> MeshCPU::getSortedEdgeFront(std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> &edgeFront, Eigen::Vector2f &pix)
{
    std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> sortedEdgeFront;

    auto auxEdgeFront = edgeFront;

    while (auxEdgeFront.size() > 0)
    {
        size_t closest;
        float closest_distance = std::numeric_limits<float>::max();
        for (auto it = auxEdgeFront.begin(); it != auxEdgeFront.end(); it++)
        {
            Triangle2D tri = getTexCoordTriangle(it->second);
            std::array<unsigned int, 2> edge = it->first;
            Eigen::Vector2f e1 = getTexCoord(edge[0]);
            Eigen::Vector2f e2 = getTexCoord(edge[1]);
            float distance = std::numeric_limits<float>::max();
            if (tri.getArea() > 10)
                distance = ((e1 + e2) / 2.0 - pix).norm();
            if (distance <= closest_distance)
            {
                closest_distance = distance;
                closest = it - auxEdgeFront.begin();
            }
        }
        sortedEdgeFront.push_back(auxEdgeFront[closest]);
        auxEdgeFront.erase(auxEdgeFront.begin() + closest);
    }

    return sortedEdgeFront;
}

std::vector<unsigned int> MeshCPU::getSortedTexCoordTriangles(Eigen::Vector2f &pix)
{
    std::vector<unsigned int> sortedTriangles;
    std::vector<unsigned int> trisIds = getTrianglesIds();

    while (trisIds.size() > 0)
    {
        size_t closestIndex;
        float closest_distance = std::numeric_limits<float>::max();
        for (auto it = trisIds.begin(); it != trisIds.end(); it++)
        {
            Triangle2D tri = getTexCoordTriangle(*it);
            float distance = std::numeric_limits<float>::max();
            if (tri.getArea() > 10)
                distance = (tri.getMean() - pix).norm();
            if (distance <= closest_distance)
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
    if (representation == cartesian)
    {
        for (auto it = vertices.begin(); it != vertices.end(); ++it)
        {
            it->second = cartesianToRayIdepth(it->second);
        }
    }

    representation = rayIdepth;
}

void MeshCPU::toCartesian()
{
    if (representation == rayIdepth)
    {
        for (auto it = vertices.begin(); it != vertices.end(); ++it)
        {
            it->second = rayIdepthToCartesian(it->second);
        }
    }
    representation = cartesian;
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

unsigned int MeshCPU::getClosestTexCoordTriangle(Eigen::Vector2f &pix)
{
    float min_distance = std::numeric_limits<float>::max();
    unsigned int min_id = 0;
    for (auto tri : triangles)
    {
        Triangle2D tri2D = getTexCoordTriangle(tri.first);
        if (tri2D.getArea() < 10)
            continue;
        float distance1 = (tri2D.vertices[0] - pix).norm();
        float distance2 = (tri2D.vertices[1] - pix).norm();
        float distance3 = (tri2D.vertices[2] - pix).norm();

        float distance = std::min(distance1, std::min(distance2, distance3));
        if (distance < min_distance)
        {
            min_distance = distance;
            min_id = tri.first;
        }
    }
    return min_id;
}

void MeshCPU::buildTriangles(camera &cam)
{
    triangles.clear();
    computeTexCoords(cam);
    DelaunayTriangulation triangulation;
    triangulation.loadPoints(texcoords);
    triangulation.triangulate();
    triangles = triangulation.getTriangles();
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
            vertices.erase(*it);
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

void MeshCPU::removeOcluded(camera &cam)
{
    std::vector<unsigned int> vetsToRemove;

    computeTexCoords(cam);
    std::vector<unsigned int> trisIds = getTrianglesIds();
    for (auto it = trisIds.begin(); it != trisIds.end(); it++)
    {
        // remove triangle if:
        // 1-is backface
        // 2-the points are behind the camera
        // 2-all vertices lays outside the image
        // afterwards,
        // remove vertices without triangles
        Triangle3D tri3D = getCartesianTriangle(*it);
        if (tri3D.vertices[0](2) <= 0.0 || tri3D.vertices[1](2) <= 0.0 || tri3D.vertices[2](2) <= 0.0)
        {
            triangles.erase(*it);
            continue;
        }
        Triangle2D tri2D = getTexCoordTriangle(*it);
        if (tri2D.getArea() < 1.0)
        {
            triangles.erase(*it);
            continue;
        }
        if (!cam.isPixVisible(tri2D.vertices[0]) && !cam.isPixVisible(tri2D.vertices[1]) && !cam.isPixVisible(tri2D.vertices[2]))
        {
            triangles.erase(*it);
            continue;
        }
    }
    removePointsWithoutTriangles();
}