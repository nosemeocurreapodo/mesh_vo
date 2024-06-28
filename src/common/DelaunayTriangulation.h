#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

#include <Eigen/Core>

class DelaunayTriangulation {
public:
    DelaunayTriangulation(const std::vector<Eigen::Vector2f>& points);
    void triangulate();
    void plot() const; // For visualization purposes
    std::vector<std::array<unsigned int, 3>> getTriangles() const;

private:
    std::vector<Eigen::Vector2f> points;
    std::vector<std::array<unsigned int, 3>> triangles;
    std::array<unsigned int, 3> superTriangle;

    void createSuperTriangle();
    bool isPointInCircumcircle(const Eigen::Vector2f& point, const std::array<unsigned int, 3>& tri) const;
    void removeSuperTriangle();
    std::pair<Eigen::Vector2f, double> circumcircle(const std::array<unsigned int, 3>& tri) const;
};

