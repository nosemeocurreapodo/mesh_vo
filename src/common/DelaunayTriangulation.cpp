#include <cmath>
#include "common/DelaunayTriangulation.h"
#include "common/common.h"

std::array<vec2<float>, 3> DelaunayTriangulation::getSuperTriangle()
{
    double minX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double maxY = std::numeric_limits<double>::lowest();

    for (auto it = vertices.begin(); it != vertices.end(); ++it)
    // for (const auto &point : points)
    {
        vec2<float> point = *it;

        if (point(0) < minX)
            minX = point(0);
        if (point(1) < minY)
            minY = point(1);
        if (point(0) > maxX)
            maxX = point(0);
        if (point(1) > maxY)
            maxY = point(1);
    }

    double dx = maxX - minX;
    double dy = maxY - minY;
    double deltaMax = std::max(dx, dy);
    double midX = (minX + maxX) / 2;
    double midY = (minY + maxY) / 2;

    std::array<vec2<float>, 3> superTriangle;

    superTriangle[0] = vec2<float>(midX - 2 * deltaMax, midY - deltaMax);
    superTriangle[1] = vec2<float>(midX, midY + 2 * deltaMax);
    superTriangle[2] = vec2<float>(midX + 2 * deltaMax, midY - deltaMax);

    return superTriangle;
}

std::pair<vec2<float>, double> DelaunayTriangulation::circumcircle(vec3<int> &tri)
{
    vec2<float> A = vertices[tri(0)];
    vec2<float> B = vertices[tri(1)];
    vec2<float> C = vertices[tri(2)];

    double D = 2 * (A(0) * (B(1) - C(1)) + B(0) * (C(1) - A(1)) + C(0) * (A(1) - B(1)));
    double Ux = ((A(0) * A(0) + A(1) * A(1)) * (B(1) - C(1)) + (B(0) * B(0) + B(1) * B(1)) * (C(1) - A(1)) + (C(0) * C(0) + C(1) * C(1)) * (A(1) - B(1))) / D;
    double Uy = ((A(0) * A(0) + A(1) * A(1)) * (C(0) - B(0)) + (B(0) * B(0) + B(1) * B(1)) * (A(0) - C(0)) + (C(0) * C(0) + C(1) * C(1)) * (B(0) - A(0))) / D;

    vec2<float> circumcenter(Ux, Uy);
    double circumradius = std::sqrt((circumcenter(0) - A(0)) * (circumcenter(0) - A(0)) + (circumcenter(1) - A(1)) * (circumcenter(1) - A(1)));

    return {circumcenter, circumradius};
}

bool DelaunayTriangulation::isPointInCircumcircle(vec2<float> &point, vec3<int> &tri)
{
    auto [circumcenter, circumradius] = circumcircle(tri);
    double dist = std::sqrt((point(0) - circumcenter(0)) * (point(0) - circumcenter(0)) + (point(1) - circumcenter(1)) * (point(1) - circumcenter(1)));
    return dist <= circumradius;
}

void DelaunayTriangulation::triangulateVertice(int v_id)
{
    std::vector<vec3<int>> goodTriangles;
    std::vector<vec3<int>> badTriangles;
    std::vector<vec2<int>> polygon;

    // check for bad triangles
    for (auto it = triangles.begin(); it != triangles.end(); ++it)
    {
        if (isPointInCircumcircle(vertices[v_id], *it))
            badTriangles.push_back(*it);
        else
            goodTriangles.push_back(*it);
    }

    for (const auto &tri : badTriangles)
    {
        std::array<vec2<int>, 3> edges;
        edges[0] = {tri(0), tri(1)};
        edges[1] = {tri(1), tri(2)};
        edges[2] = {tri(2), tri(0)};

        for (size_t j = 0; j < edges.size(); j++)
        {
            vec2<int> edge = edges[j];
            int edge_index = -1;
            for (size_t k = 0; k < polygon.size(); k++)
            {
                vec2<int> pol = polygon[k];

                if (isEdgeEqual(edge, pol))
                {
                    edge_index = k;
                    break;
                }
            }

            if (edge_index >= 0)
            {
                polygon.erase(polygon.begin() + edge_index);
            }
            else
            {
                polygon.push_back(edge);
            }
        }
    }

    triangles = goodTriangles;

    for (const auto &edge : polygon)
    {
        vec3<int> tri;
        tri(0) = edge(0);
        tri(1) = edge(1);
        tri(2) = v_id;

        /*
        Triangle2D triStru = Triangle2D(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]);
        float area = triStru.getArea();
        if(area <= 0)
        {
            tri[0] = edge[1];
            tri[1] = edge[0];
        }
        */

        triangles.push_back(tri);
    }
}

void DelaunayTriangulation::triangulate()
{
    triangles.clear();
    std::array<vec2<float>, 3> superTriangleVertices = getSuperTriangle();
    vec3<int> superTriangleIndices;

    superTriangleIndices(0) = vertices.size();
    vertices.push_back(superTriangleVertices[0]);

    superTriangleIndices(1) = vertices.size();
    vertices.push_back(superTriangleVertices[1]);

    superTriangleIndices(2) = vertices.size();
    vertices.push_back(superTriangleVertices[2]);

    triangles.push_back(superTriangleIndices);

    for (int it = 0; it < (int)vertices.size() - 3; ++it)
    {
        triangulateVertice(it);
    }

    removeVertice(superTriangleIndices(0));
    removeVertice(superTriangleIndices(1));
    removeVertice(superTriangleIndices(2));
}

void DelaunayTriangulation::removeVertice(int v_id)
{
    // int superA = superTriangle[0];
    // int superB = superTriangle[1];
    // int superC = superTriangle[2];

    // triangles.erase(std::remove_if(triangles.begin(), triangles.end(), [&](const std::array<unsigned int, 3> &tri)
    //                                { return (tri[0] == superA || tri[0] == superB || tri[0] == superC ||
    //                                          tri[1] == superA || tri[1] == superB || tri[1] == superC ||
    //                                          tri[2] == superA || tri[2] == superB || tri[2] == superC); }),
    //                 triangles.end());

    vertices.erase(vertices.begin() + v_id);

    std::vector<int> to_remove;
    for (int it = 0; it < (int)triangles.size(); it++)
    {
        vec3<int> tri = triangles[it];

        if (v_id == tri(0) || v_id == tri(1) || v_id == tri(2))
        {
            to_remove.push_back(it);
        }
    }

    if (to_remove.size() > 0)
    {
        for (int it = to_remove.size() - 1; it >= 0; it--)
        {
            triangles.erase(triangles.begin() + to_remove[it]);
        }
    }

    /*
    for (auto it = triangles.begin(); it != triangles.end();)
    {
        bool todelete = false;
        vec3<unsigned int> tri = *it;
        for (int i = 0; i < 3; i++)
        {
            if (tri(i) == v_id)
            {
                todelete = true;
                break;
            }
        }
        if (todelete)
            it = triangles.erase(it);
        else
            ++it;
    }
    */
}

std::vector<vec3<int>> DelaunayTriangulation::getTriangles()
{
    return triangles;
}
