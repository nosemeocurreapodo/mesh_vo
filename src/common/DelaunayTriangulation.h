#pragma once

#include <map>
#include <vector>
#include <array>
#include <cmath>

#include "common/types.h"
#include "common/comparison.h"

class DelaunayTriangulation
{
public:
    DelaunayTriangulation()
    {
    }

    void loadPoints(std::vector<vec2f> texcoords)
    {
        vertices = texcoords;
    }

    void loadTriangles(std::vector<vec3i> &tris)
    {
        triangles = tris;
    }

    std::vector<vec3i> getTriangles()
    {
        return triangles;
    }

    void triangulateVertice(int v_id);
    void triangulate();

private:

    std::array<vec2f, 3> getSuperTriangle();
    void removeVertice(int v_id);

    bool isPointInCircumcircle(vec2f &vertice, vec3i &tri);
    std::pair<vec2f, double> circumcircle(vec3i &tri);

    std::vector<vec2f> vertices;
    std::vector<vec3i> triangles;
};
