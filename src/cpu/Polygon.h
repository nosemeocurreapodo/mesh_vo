#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/camera.h"
class Polygon
{
public:
    Polygon(){

    };

    virtual Eigen::Vector3f getNormal(Eigen::Vector3f &ray)
    {
    }

    virtual float getArea()
    {
    }

    virtual std::array<int, 4> getScreenBounds(camera &cam)
    {
    }

    virtual float getRayDepth(Eigen::Vector3f &ray)
    {
    }

    bool isRayInPolygon(Eigen::Vector3f &ray)
    {
    }

    virtual Eigen::Vector3f getVertice(unsigned int id)
    {
    }
};

class PolygonFlat : public Polygon
{
public:
    PolygonFlat(Eigen::Vector3f vert1, Eigen::Vector3f vert2, Eigen::Vector3f vert3)
    {
        vertices[0] = vert1;
        vertices[1] = vert2;
        vertices[2] = vert3;

        rays[0] = vertices[0] / vertices[0](2);
        rays[1] = vertices[1] / vertices[1](2);
        rays[2] = vertices[2] / vertices[2](2);

        computeNormal();
        computeTinv();
    };

    float getRayDepth(Eigen::Vector3f &ray)
    {
        float ray_depth = vertices[0].dot(normal) / ray.dot(normal);
        return ray_depth;
    }

    Eigen::Vector3f dDepthdVert(Eigen::Vector3f &ray)
    {
        Eigen::Vector3f dDepthdVert0 = 
    }

    Eigen::Vector3f getNormal(Eigen::Vector3f &ray)
    {
        return normal;
    }

    float getArea()
    {
        return normal.norm() / 2.0;
    }
    /*
    float getScreenArea()
    {
        float area = rays[0](0) * (rays[1](1) - rays[2](1));
        area += rays[1](0) * (rays[2](1) - rays[0](1));
        area += rays[2](0) * (rays[0](1) - rays[1](1));
        return area;
    }
    */
    /*
     std::array<float, 3> getScreenAngles()
     {
         Eigen::Vector2f a = (rays[1] - rays[0]).normalized();
         Eigen::Vector2f b = (rays[2] - rays[0]).normalized();

         float cosalpha = a.dot(b);
         float alpha = acos(cosalpha);

         a = (rays[0] - rays[1]).normalized();
         b = (rays[2] - rays[1]).normalized();

         float cosbeta = a.dot(b);
         float beta = acos(cosbeta);

         float gamma = M_PI - alpha - beta;

         return {alpha, beta, gamma};
     }
     */

    bool isRayInPolygon(Eigen::Vector3f &ray)
    {
        Eigen::Vector2f lambda = T_inv * (Eigen::Vector2f(ray(0), ray(1)) - Eigen::Vector2f(rays[2](0), rays[2](1)));
        barycentric = Eigen::Vector3f(lambda(0), lambda(1), 1.0 - lambda(0) - lambda(1));

        if (barycentric(0) < -0.0 || barycentric(1) < -0.0 || barycentric(2) < -0.0)
            return false;
        if (barycentric(0) > 1.0 || barycentric(1) > 1.0 || barycentric(2) > 1.0)
            return false;
        return true;
    };

    /*
    template <typename Type>
    Type interpolate(Type &d1, Type &d2, Type &d3)
    {
        return barycentric(0) * d1 + barycentric(1) * d2 + barycentric(2) * d3;
    };
    */

    std::array<int, 4> getScreenBounds(camera &cam)
    {
        Eigen::Vector2f screencoords[3];
        screencoords[0] = cam.rayToPix(rays[0]);
        screencoords[1] = cam.rayToPix(rays[1]);
        screencoords[2] = cam.rayToPix(rays[2]);

        std::array<int, 4> minmax;
        minmax[0] = (int)std::min(std::min(screencoords[0](0), screencoords[1](0)), screencoords[2](0)) - 1;
        minmax[1] = (int)std::max(std::max(screencoords[0](0), screencoords[1](0)), screencoords[2](0)) + 1;
        minmax[2] = (int)std::min(std::min(screencoords[0](1), screencoords[1](1)), screencoords[2](1)) - 1;
        minmax[3] = (int)std::max(std::max(screencoords[0](1), screencoords[1](1)), screencoords[2](1)) + 1;

        return minmax;
    };

    Eigen::Vector3f getVertice(unsigned int id)
    {
        return vertices[id];
    }

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
        T(0, 0) = rays[0](0) - rays[2](0);
        T(0, 1) = rays[1](0) - rays[2](0);
        T(1, 0) = rays[0](1) - rays[2](1);
        T(1, 1) = rays[1](1) - rays[2](1);
        T_inv = T.inverse();
    };

    Eigen::Vector3f vertices[3];
    Eigen::Vector3f rays[3];

    Eigen::Vector3f normal;

    Eigen::Matrix2f T_inv;
    Eigen::Vector3f barycentric;
};

class PolygonSmooth : public Polygon
{
public:
    PolygonSmooth(Eigen::Vector3f vert1, Eigen::Vector3f vert2, Eigen::Vector3f vert3,
                  Eigen::Vector3f norm1, Eigen::Vector3f norm2, Eigen::Vector3f norm3)
    {
        vertices[0] = vert1;
        vertices[1] = vert2;
        vertices[2] = vert3;

        normals[0] = norm1;
        normals[1] = norm2;
        normals[2] = norm3;

        rays[0] = vertices[0] / vertices[0](2);
        rays[1] = vertices[1] / vertices[1](2);
        rays[2] = vertices[2] / vertices[2](2);

        computeTinv();
    };

    float getRayDepth(Eigen::Vector3f &ray)
    {
        float ray_depth = 0.0;
        for (int i = 0; i < 3; i++)
            ray_depth += vertices[0].dot(normals[0]) / ray.dot(normals[0]);
        ray_depth /= 3.0;
        return ray_depth;
    }

    Eigen::Vector3f getNormal(Eigen::Vector3f &ray)
    {
        return normal;
    }

    float getArea()
    {
        return normal.norm() / 2.0;
    }
    /*
    float getScreenArea()
    {
        float area = rays[0](0) * (rays[1](1) - rays[2](1));
        area += rays[1](0) * (rays[2](1) - rays[0](1));
        area += rays[2](0) * (rays[0](1) - rays[1](1));
        return area;
    }
    */
    /*
     std::array<float, 3> getScreenAngles()
     {
         Eigen::Vector2f a = (rays[1] - rays[0]).normalized();
         Eigen::Vector2f b = (rays[2] - rays[0]).normalized();

         float cosalpha = a.dot(b);
         float alpha = acos(cosalpha);

         a = (rays[0] - rays[1]).normalized();
         b = (rays[2] - rays[1]).normalized();

         float cosbeta = a.dot(b);
         float beta = acos(cosbeta);

         float gamma = M_PI - alpha - beta;

         return {alpha, beta, gamma};
     }
     */

    bool isRayInPolygon(Eigen::Vector3f &ray)
    {
        Eigen::Vector2f lambda = T_inv * (Eigen::Vector2f(ray(0), ray(1)) - Eigen::Vector2f(rays[2](0), rays[2](1)));
        barycentric = Eigen::Vector3f(lambda(0), lambda(1), 1.0 - lambda(0) - lambda(1));

        if (barycentric(0) < -0.0 || barycentric(1) < -0.0 || barycentric(2) < -0.0)
            return false;
        if (barycentric(0) > 1.0 || barycentric(1) > 1.0 || barycentric(2) > 1.0)
            return false;
        return true;
    };

    /*
    template <typename Type>
    Type interpolate(Type &d1, Type &d2, Type &d3)
    {
        return barycentric(0) * d1 + barycentric(1) * d2 + barycentric(2) * d3;
    };
    */

    std::array<int, 4> getScreenBounds(camera &cam)
    {
        Eigen::Vector2f screencoords[3];
        screencoords[0] = cam.rayToPix(rays[0]);
        screencoords[1] = cam.rayToPix(rays[1]);
        screencoords[2] = cam.rayToPix(rays[2]);

        std::array<int, 4> minmax;
        minmax[0] = (int)std::min(std::min(screencoords[0](0), screencoords[1](0)), screencoords[2](0)) - 1;
        minmax[1] = (int)std::max(std::max(screencoords[0](0), screencoords[1](0)), screencoords[2](0)) + 1;
        minmax[2] = (int)std::min(std::min(screencoords[0](1), screencoords[1](1)), screencoords[2](1)) - 1;
        minmax[3] = (int)std::max(std::max(screencoords[0](1), screencoords[1](1)), screencoords[2](1)) + 1;

        return minmax;
    };

    Eigen::Vector3f getVertice(unsigned int id)
    {
        return vertices[id];
    }

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
    void computeTinv()
    {
        Eigen::Matrix2f T;
        T(0, 0) = rays[0](0) - rays[2](0);
        T(0, 1) = rays[1](0) - rays[2](0);
        T(1, 0) = rays[0](1) - rays[2](1);
        T(1, 1) = rays[1](1) - rays[2](1);
        T_inv = T.inverse();
    };

    Eigen::Vector3f vertices[3];
    Eigen::Vector3f normals[3];

    Eigen::Vector3f rays[3];

    Eigen::Matrix2f T_inv;
    Eigen::Vector3f barycentric;
};
