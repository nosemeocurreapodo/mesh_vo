#pragma once

#include <Eigen/Core>

class Polygon
{
public:
    Polygon(){

    };

    virtual float getDepth(Eigen::Vector3f &ray)
    {
    }

    virtual Eigen::Vector3f getNormal(Eigen::Vector3f &ray)
    {
    }

    virtual Eigen::Vector3f getMean() {

    };

    virtual float getArea() {

    };
};

class PolygonFlat : public Polygon
{
public:
    PolygonFlat(Eigen::Vector3f vert1, Eigen::Vector3f vert2, Eigen::Vector3f vert3,
                Eigen::Vector2f texc1, Eigen::Vector2f texc2, Eigen::Vector3f texc3)
    {
        vertices[0] = vert1;
        vertices[1] = vert2;
        vertices[2] = vert3;

        texcoords[0] = texc1;
        texcoords[1] = texc2;
        texcoords[2] = texc3;

        computeNormal();
        computeTinv();
    };

    //3D
    float getDepth(Eigen::Vector3f &ray)
    {
        float ray_depth = vertices[0].dot(normal) / ray.dot(normal);
        return ray_depth;
    }

    Eigen::Vector3f getNormal(Eigen::Vector3f &ray)
    {
        return normal;
    }

    Eigen::Vector3f getMean()
    {
        return (vertices[0] + vertices[1] + vertices[2]) / 3.0;
    };

    float getArea()
    {
        return normal.norm() / 2.0;
    }

    //2D

    float getScreenArea()
    {
        float area = texcoords[0](0) * (texcoords[1](1) - texcoords[2](1));
        area += texcoords[1](0) * (texcoords[2](1) - texcoords[0](1));
        area += texcoords[2](0) * (texcoords[0](1) - texcoords[1](1));
        return area;
    }
    std::array<float, 3> getScreenAngles()
    {
        Eigen::Vector2f a = (texcoords[1] - texcoords[0]).normalized();
        Eigen::Vector2f b = (texcoords[2] - texcoords[0]).normalized();

        float cosalpha = a.dot(b);
        float alpha = acos(cosalpha);

        a = (texcoords[0] - texcoords[1]).normalized();
        b = (texcoords[2] - texcoords[1]).normalized();

        float cosbeta = a.dot(b);
        float beta = acos(cosbeta);

        float gamma = M_PI - alpha - beta;

        return {alpha, beta, gamma};
    }

    void computeBarycentric(Eigen::Vector2f p)
    {
        Eigen::Vector2f lambda = T_inv * (p - texcoords[2]);
        barycentric = Eigen::Vector3f(lambda(0), lambda(1), 1.0 - lambda(0) - lambda(1));
    };

    bool isBarycentricOk()
    {
        if (barycentric(0) < -0.0 || barycentric(1) < -0.0 || barycentric(2) < -0.0)
            return false;
        if (barycentric(0) > 1.0 || barycentric(1) > 1.0 || barycentric(2) > 1.0)
            return false;
        return true;
    };

    template <typename Type>
    Type interpolate(Type &d1, Type &d2, Type &d3)
    {
        return barycentric(0) * d1 + barycentric(1) * d2 + barycentric(2) * d3;
    };

    std::array<int, 4> getScreenBounds()
    {
        std::array<int, 4> minmax;
        minmax[0] = (int)std::min(std::min(texcoords[0](0), texcoords[1](0)), texcoords[2](0)) - 1;
        minmax[1] = (int)std::max(std::max(texcoords[0](0), texcoords[1](0)), texcoords[2](0)) + 1;
        minmax[2] = (int)std::min(std::min(texcoords[0](1), texcoords[1](1)), texcoords[2](1)) - 1;
        minmax[3] = (int)std::max(std::max(texcoords[0](1), texcoords[1](1)), texcoords[2](1)) + 1;

        return minmax;
    };

    bool isEdge()
    {
        if (barycentric(0) < 0.04 || barycentric(1) < 0.04 || barycentric(2) < 0.04)
            return true;
        return false;
    };

    bool isVertice()
    {
        if (barycentric(0) > 0.98 || barycentric(1) > 0.98 || barycentric(2) > 0.98)
            return true;
        return false;
    };

private:
    void computeNormal()
    {
        normal = ((vertices[0] - vertices[2]).cross(vertices[0] - vertices[1]));
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

    Eigen::Vector3f vertices[3];
    Eigen::Vector3f texcoords[3];

    Eigen::Vector3f normal;

    Eigen::Matrix2f T_inv;
    Eigen::Vector3f barycentric;
};

class PolygonCurved : public Polygon
{
public:
    PolygonCurved(Eigen::Vector3f vert1, Eigen::Vector3f vert2, Eigen::Vector3f vert3, Eigen::Vector3f norm1, Eigen::Vector3f norm2, Eigen::Vector3f norm3)
    {
        vertices[0] = vert1;
        vertices[1] = vert2;
        vertices[2] = vert3;

        normals[0] = norm1;
        normals[1] = norm2;
        normals[2] = norm3;
    };

    float getDepth(Eigen::Vector3f &ray)
    {
        float ray_depth = 0.0;
        ray_depth += vertices[0].dot(normals[0]) / ray.dot(normals[0]);
        ray_depth += vertices[1].dot(normals[1]) / ray.dot(normals[1]);
        ray_depth += vertices[2].dot(normals[2]) / ray.dot(normals[2]);
        ray_depth /= 3;
        return ray_depth;
    }

    Eigen::Vector3f getNormal(Eigen::Vector3f &ray)
    {
        return (vertices[0] - vertices[2]).cross(vertices[0] - vertices[1]);
    }

    Eigen::Vector3f getMean()
    {
        return (vertices[0] + vertices[1] + vertices[2]) / 3.0;
    };

    float getArea()
    {
        return 1.0;
    }

private:
    Eigen::Vector3f vertices[3];
    Eigen::Vector3f normals[3];
};
