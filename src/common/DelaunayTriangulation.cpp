#include "common/DelaunayTriangulation.h"
#include "common/common.h"

void DelaunayTriangulation::addSuperTriangle()
{
    double minX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double maxY = std::numeric_limits<double>::lowest();

    for (auto it = vertices.begin(); it != vertices.end(); ++it)
    // for (const auto &point : points)
    {
        Eigen::Vector2f point = it->second;

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

    unsigned int superA = std::numeric_limits<unsigned int>::max() - 3;
    unsigned int superB = std::numeric_limits<unsigned int>::max() - 2;
    unsigned int superC = std::numeric_limits<unsigned int>::max() - 1;

    // vertices[superA] = {minX, minY};
    // vertices[superB] = {maxX * 2, minY};
    // vertices[superC] = {minX, maxY * 2};

    vertices[superA] = {midX - 2 * deltaMax, midY - deltaMax};
    vertices[superB] = {midX, midY + 2 * deltaMax};
    vertices[superC] = {midX + 2 * deltaMax, midY - deltaMax};

    // points.push_back({midX - 2 * deltaMax, midY - deltaMax});
    // points.push_back({midX, midY + 2 * deltaMax});
    // points.push_back({midX + 2 * deltaMax, midY - deltaMax});

    superTriangle[0] = superA;
    superTriangle[1] = superB;
    superTriangle[2] = superC;

    triangles[triangles.size()] = superTriangle;
}

std::pair<Eigen::Vector2f, double> DelaunayTriangulation::circumcircle(std::array<unsigned int, 3> &tri)
{
    Eigen::Vector2f A = vertices[tri[0]];
    Eigen::Vector2f B = vertices[tri[1]];
    Eigen::Vector2f C = vertices[tri[2]];

    double D = 2 * (A(0) * (B(1) - C(1)) + B(0) * (C(1) - A(1)) + C(0) * (A(1) - B(1)));
    double Ux = ((A(0) * A(0) + A(1) * A(1)) * (B(1) - C(1)) + (B(0) * B(0) + B(1) * B(1)) * (C(1) - A(1)) + (C(0) * C(0) + C(1) * C(1)) * (A(1) - B(1))) / D;
    double Uy = ((A(0) * A(0) + A(1) * A(1)) * (C(0) - B(0)) + (B(0) * B(0) + B(1) * B(1)) * (A(0) - C(0)) + (C(0) * C(0) + C(1) * C(1)) * (B(0) - A(0))) / D;

    Eigen::Vector2f circumcenter(Ux, Uy);
    double circumradius = std::sqrt((circumcenter(0) - A(0)) * (circumcenter(0) - A(0)) + (circumcenter(1) - A(1)) * (circumcenter(1) - A(1)));

    return {circumcenter, circumradius};
}

bool DelaunayTriangulation::isPointInCircumcircle(Eigen::Vector2f &point, std::array<unsigned int, 3> &tri)
{
    auto [circumcenter, circumradius] = circumcircle(tri);
    double dist = std::sqrt((point(0) - circumcenter(0)) * (point(0) - circumcenter(0)) + (point(1) - circumcenter(1)) * (point(1) - circumcenter(1)));
    return dist <= circumradius;
}

void DelaunayTriangulation::triangulateVertice(Eigen::Vector2f &vertice, unsigned int id)
{
    vertices[id] = vertice;

    std::vector<std::array<unsigned int, 3>> goodTriangles;
    std::vector<std::array<unsigned int, 3>> badTriangles;
    std::vector<std::array<unsigned int, 2>> polygon;

    // check for bad triangles
    for (auto it = triangles.begin(); it != triangles.end(); ++it)
    {
        if (isPointInCircumcircle(vertice, it->second))
            badTriangles.push_back(it->second);
        else
            goodTriangles.push_back(it->second);
    }

    for (const auto &tri : badTriangles)
    {
        std::array<std::array<unsigned int, 2>, 3> edges;
        edges[0] = {tri[0], tri[1]};
        edges[1] = {tri[1], tri[2]};
        edges[2] = {tri[2], tri[0]};

        for (int j = 0; j < edges.size(); j++)
        {
            std::array<unsigned int, 2> edge = edges[j];
            int edge_index = -1;
            for (int k = 0; k < polygon.size(); k++)
            {
                std::array<unsigned int, 2> pol = polygon[k];

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

    triangles.clear();
    for (auto tri : goodTriangles)
    {
        triangles[triangles.size()] = tri;
    }

    for (const auto &edge : polygon)
    {
        std::array<unsigned int, 3> tri;
        tri[0] = edge[0];
        tri[1] = edge[1];
        tri[2] = static_cast<unsigned int>(id);

        /*
        Triangle2D triStru = Triangle2D(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]);
        float area = triStru.getArea();
        if(area <= 0)
        {
            tri[0] = edge[1];
            tri[1] = edge[0];
        }
        */

        triangles[triangles.size()] = tri;
    }
}

void DelaunayTriangulation::triangulate()
{
    triangles.clear();
    addSuperTriangle();
    for (auto it = vertices.begin(); it != vertices.end(); ++it)
    {
        if (it->first == superTriangle[0] || it->first == superTriangle[1] || it->first == superTriangle[2])
            continue;
        triangulateVertice(it->second, it->first);
    }
    removeSuperTriangle();
}

void DelaunayTriangulation::removeSuperTriangle()
{
    // int superA = superTriangle[0];
    // int superB = superTriangle[1];
    // int superC = superTriangle[2];

    // triangles.erase(std::remove_if(triangles.begin(), triangles.end(), [&](const std::array<unsigned int, 3> &tri)
    //                                { return (tri[0] == superA || tri[0] == superB || tri[0] == superC ||
    //                                          tri[1] == superA || tri[1] == superB || tri[1] == superC ||
    //                                          tri[2] == superA || tri[2] == superB || tri[2] == superC); }),
    //                 triangles.end());

    for (auto it = triangles.begin(); it != triangles.end();)
    {
        bool todelete = false;
        std::array<unsigned int, 3> tri = it->second;
        for (int i = 0; i < 3; i++)
        {
            if (tri[i] == superTriangle[0] || tri[i] == superTriangle[1] || tri[i] == superTriangle[2])
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
}

std::map<unsigned int, std::array<unsigned int, 3>> DelaunayTriangulation::getTriangles()
{
    return triangles;
}
