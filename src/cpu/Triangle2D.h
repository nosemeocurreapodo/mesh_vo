#pragma once

#include <Eigen/Core>

class Triangle2D
{
public:

    Triangle2D(Eigen::Vector2f &vert1, Eigen::Vector2f &vert2, Eigen::Vector2f &vert3)
    {
        vertices[0] = vert1;
        vertices[1] = vert2;
        vertices[2] = vert3;
    };

    Eigen::Vector2f getMean()
    {
        return (vertices[0] + vertices[1] + vertices[2]) / 3.0;
    };

    float getArea()
    {
        float area = vertices[0](0) * (vertices[1](1) - vertices[2](1));
        area += vertices[1](0) * (vertices[2](1) - vertices[0](1));
        area += vertices[2](0) * (vertices[0](1) - vertices[1](1));
        return area;
    }

    void computeTinv()
    {
        Eigen::Matrix2f T;
        T(0, 0) = vertices[0](0) - vertices[2](0);
        T(0, 1) = vertices[1](0) - vertices[2](0);
        T(1, 0) = vertices[0](1) - vertices[2](1);
        T(1, 1) = vertices[1](1) - vertices[2](1);
        T_inv = T.inverse();
    };

    void computeBarycentric(Eigen::Vector2f p)
    {
        Eigen::Vector2f lambda = T_inv * (p - vertices[2]);
        barycentric = Eigen::Vector3f(lambda(0), lambda(1), 1 - lambda(0) - lambda(1));
    };

    bool isBarycentricOk()
    {
        if (barycentric(0) <= 0.0 || barycentric(1) <= 0.0 || barycentric(2) <= 0.0)
            return false;
        if (barycentric(0) >= 1.0 || barycentric(1) >= 1.0 || barycentric(2) >= 1.0)
            return false;
        return true;
    };

    template <typename Type>
    Type interpolate(Type &d1, Type &d2, Type &d3)
    {
        return barycentric(0) * d1 + barycentric(1) * d2 + barycentric(2) * d3;
    };

    bool isLine()
    {
        if (barycentric(0) < 0.04 || barycentric(1) < 0.04 || barycentric(2) < 0.04)
            return true;
        return false;
    };

    bool isPoint()
    {
        if (barycentric(0) > 0.98 || barycentric(1) > 0.98 || barycentric(2) > 0.98)
            return true;
        return false;
    };

    std::array<Eigen::Vector2f, 2> getMinMax()
    {
        std::array<Eigen::Vector2f, 2> minmax;
        minmax[0](0) = std::min(std::min(vertices[0](0), vertices[1](0)), vertices[2](0));
        minmax[0](1) = std::min(std::min(vertices[0](1), vertices[1](1)), vertices[2](1));
        minmax[1](0) = std::max(std::max(vertices[0](0), vertices[1](0)), vertices[2](0));
        minmax[1](1) = std::max(std::max(vertices[0](1), vertices[1](1)), vertices[2](1));

        return minmax;
    };

    std::array<Eigen::Vector2f, 3> vertices;

    Eigen::Matrix2f T_inv;
    Eigen::Vector3f barycentric;

private:
};