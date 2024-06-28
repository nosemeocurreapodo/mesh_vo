#include "DelaunayTriangulation.h"

DelaunayTriangulation::DelaunayTriangulation(const std::vector<Eigen::Vector2f> &points) : points(points)
{
    createSuperTriangle();
    triangles.push_back(superTriangle);
}

void DelaunayTriangulation::createSuperTriangle()
{
    double minX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double maxY = std::numeric_limits<double>::lowest();

    for (const auto &point : points)
    {
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

    points.push_back({midX - 2 * deltaMax, midY - deltaMax});
    points.push_back({midX, midY + 2 * deltaMax});
    points.push_back({midX + 2 * deltaMax, midY - deltaMax});

    unsigned int superA = points.size() - 3;
    unsigned int superB = points.size() - 2;
    unsigned int superC = points.size() - 1;

    superTriangle[0] = superA;
    superTriangle[1] = superB;
    superTriangle[2] = superC;
}

std::pair<Eigen::Vector2f, double> DelaunayTriangulation::circumcircle(const std::array<unsigned int, 3> &tri) const
{
    const Eigen::Vector2f &A = points[tri[0]];
    const Eigen::Vector2f &B = points[tri[1]];
    const Eigen::Vector2f &C = points[tri[2]];

    double D = 2 * (A(0) * (B(1) - C(1)) + B(0) * (C(1) - A(1)) + C(0) * (A(1) - B(1)));
    double Ux = ((A(0) * A(0) + A(1) * A(1)) * (B(1) - C(1)) + (B(0) * B(0) + B(1) * B(1)) * (C(1) - A(1)) + (C(0) * C(0) + C(1) * C(1)) * (A(1) - B(1))) / D;
    double Uy = ((A(0) * A(0) + A(1) * A(1)) * (C(0) - B(0)) + (B(0) * B(0) + B(1) * B(1)) * (A(0) - C(0)) + (C(0) * C(0) + C(1) * C(1)) * (B(0) - A(0))) / D;

    Eigen::Vector2f circumcenter(Ux, Uy);
    double circumradius = std::sqrt((circumcenter(0) - A(0)) * (circumcenter(0) - A(0)) + (circumcenter(1) - A(1)) * (circumcenter(1) - A(1)));

    return {circumcenter, circumradius};
}

bool DelaunayTriangulation::isPointInCircumcircle(const Eigen::Vector2f &point, const std::array<unsigned int, 3> &tri) const
{
    auto [circumcenter, circumradius] = circumcircle(tri);
    double dist = std::sqrt((point(0) - circumcenter(0)) * (point(0) - circumcenter(0)) + (point(1) - circumcenter(1)) * (point(1) - circumcenter(1)));
    return dist <= circumradius;
}

void DelaunayTriangulation::triangulate()
{
    for (size_t i = 0; i < points.size() - 3; ++i)
    { // Exclude super triangle points
        const Eigen::Vector2f &point = points[i];
        std::vector<std::array<unsigned int, 3>> badTriangles;
        std::vector<std::array<unsigned int, 2>> polygon;

        for (const auto &tri : triangles)
        {
            if (isPointInCircumcircle(point, tri))
            {
                badTriangles.push_back(tri);
                std::array<std::array<unsigned int, 2>, 3> edges;
                edges[0] = {tri[0], tri[1]};
                edges[1] = {tri[1], tri[2]};
                edges[2] = {tri[2], tri[0]};

                for( int j = 0; j < polygon.size(); j++)
                {
                    std::array<unsigned int, 2> pol = polygon[j];
                    for(int k = 0: k)
                    if(edges[0][0] == )
                }

                /*
                for (const auto &edge : {std::make_pair(tri[0], tri[1]), std::make_pair(tri[1], tri[2]), std::make_pair(tri[2], tri[0])})
                {
                    if (std::find(polygon.begin(), polygon.end(), edge) != polygon.end())
                    {
                        polygon.erase(std::remove(polygon.begin(), polygon.end(), edge), polygon.end());
                    }
                    else
                    {
                        polygon.push_back(edge);
                    }
                }
                */
            }
        }

        triangles.erase(std::remove_if(triangles.begin(), triangles.end(), [&](const std::array<unsigned int, 3> &tri)
                                       { return std::find(badTriangles.begin(), badTriangles.end(), tri) != badTriangles.end(); }),
                        triangles.end());

        for (const auto &edge : polygon)
        {
            std::array<unsigned int, 3> tri;
            tri[0] = edge.first;
            tri[1] = edge.second;
            tri[2] = static_cast<unsigned int>(i);
            triangles.push_back(tri);
        }
    }
    removeSuperTriangle();
}

void DelaunayTriangulation::removeSuperTriangle()
{
    int superA = points.size() - 3;
    int superB = points.size() - 2;
    int superC = points.size() - 1;

    triangles.erase(std::remove_if(triangles.begin(), triangles.end(), [&](const std::array<unsigned int, 3> &tri)
                                   { return (tri[0] == superA || tri[0] == superB || tri[0] == superC ||
                                             tri[1] == superA || tri[1] == superB || tri[1] == superC ||
                                             tri[2] == superA || tri[2] == superB || tri[2] == superC); }),
                    triangles.end());
}

std::vector<std::array<unsigned int, 3>> DelaunayTriangulation::getTriangles() const
{
    return triangles;
}

void DelaunayTriangulation::plot() const
{
    // Implement visualization logic here if needed
}
