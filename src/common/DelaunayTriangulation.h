#pragma once

#include <map>
#include <vector>
//#include <cmath>
//#include <algorithm>
//#include <iostream>

//#include <Eigen/Core>
#include "common/types.h"

class DelaunayTriangulation
{
public:
    DelaunayTriangulation()
    {
    }

    void loadPoints(std::vector<vec2<float>> &texcoords)
    {
        vertices = texcoords;
    }

    void loadTriangles(std::vector<vec3<int>> &tris)
    {
        triangles = tris;
    }

    void triangulateVertice(int v_id);
    void triangulate();
    std::vector<vec3<int>> getTriangles();

private:

    std::array<vec2<float>, 3> getSuperTriangle();
    void removeVertice(int v_id);

    bool isPointInCircumcircle(vec2<float> &vertice, vec3<int> &tri);
    std::pair<vec2<float>, double> circumcircle(vec3<int> &tri);

    std::vector<vec2<float>> vertices;
    std::vector<vec3<int>> triangles;
};
