#pragma once

#include <Eigen/Core>
#include <math.h>

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

    std::array<float, 3> getAngles()
    {
        Eigen::Vector2f a = (vertices[1] - vertices[0]).normalized();
        Eigen::Vector2f b = (vertices[2] - vertices[0]).normalized();

        float cosalpha = a.dot(b);
        float alpha = acos(cosalpha);

        a = (vertices[0] - vertices[1]).normalized();
        b = (vertices[2] - vertices[1]).normalized();

        float cosbeta = a.dot(b);
        float beta = acos(cosbeta);

        float gamma = M_PI - alpha - beta;

        return {alpha, beta, gamma};
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
        barycentric = Eigen::Vector3f(lambda(0), lambda(1), 1.0 - lambda(0) - lambda(1));
    };

    bool isBarycentricOk()
    {
        if (barycentric(0) < -0.0 || barycentric(1) < -0.0 || barycentric(2) < -0.0)
            return false;
        if (barycentric(0) >  1.0 || barycentric(1) >  1.0 || barycentric(2) >  1.0)
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

    std::array<int, 4> getMinMax()
    {
        std::array<int, 4> minmax;
        minmax[0] = (int)std::min(std::min(vertices[0](0), vertices[1](0)), vertices[2](0))-1;
        minmax[1] = (int)std::max(std::max(vertices[0](0), vertices[1](0)), vertices[2](0))+1;
        minmax[2] = (int)std::min(std::min(vertices[0](1), vertices[1](1)), vertices[2](1))-1;
        minmax[3] = (int)std::max(std::max(vertices[0](1), vertices[1](1)), vertices[2](1))+1;

        return minmax;
    };

    std::array<Eigen::Vector2f, 3> vertices;

    Eigen::Matrix2f T_inv;
    Eigen::Vector3f barycentric;

private:
};