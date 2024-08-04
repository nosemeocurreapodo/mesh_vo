#pragma once

#include <map>
#include <vector>
//#include <cmath>
//#include <algorithm>
//#include <iostream>

#include <Eigen/Core>

class DelaunayTriangulation
{
public:
    DelaunayTriangulation()
    {
    }

    void addSuperTriangle();
    void removeSuperTriangle();

    void loadPoints(std::unordered_map<unsigned int, Eigen::Vector2f> &texcoords)
    {
        vertices = texcoords;
    }

    void loadTriangles(std::unordered_map<unsigned int, std::array<unsigned int, 3>> &tris)
    {
        for (auto it = tris.begin(); it != tris.end(); ++it)
        {
            triangles[it->first] = it->second;
        }
    }

    void triangulateVertice(Eigen::Vector2f &vertice, unsigned int id);
    void triangulate();
    std::unordered_map<unsigned int, std::array<unsigned int, 3>> getTriangles();

private:
    std::unordered_map<unsigned int, Eigen::Vector2f> vertices;
    std::unordered_map<unsigned int, std::array<unsigned int, 3>> triangles;

    std::array<unsigned int, 3> superTriangle;

    bool isPointInCircumcircle(Eigen::Vector2f &vertice, std::array<unsigned int, 3> &tri);
    std::pair<Eigen::Vector2f, double> circumcircle(std::array<unsigned int, 3> &tri);
};
