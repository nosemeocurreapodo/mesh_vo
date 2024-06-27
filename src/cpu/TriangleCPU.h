#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/camera.h"
#include "common/common.h"
#include "cpu/VerticeCPU.h"

class TriangleCPU
{
public:
    TriangleCPU()
    {

    };

    TriangleCPU(VerticeCPU &vert1, VerticeCPU &vert2, VerticeCPU &vert3)
    {
        vertices[0] = &vert1;
        vertices[1] = &vert2;
        vertices[2] = &vert3;

    };

    std::array<VerticeCPU *, 2> toConnect(Eigen::Vector2f &pix)
    {
        computeTinv();

        std::array<VerticeCPU *, 2> toc;
        int index = 0;
        for (int i = 0; i < 3; i++)
        {
            Eigen::Vector2f shifted = vertices[i]->texcoord + (pix - vertices[i]->texcoord).normalized()*2.0;
            computeBarycentric(shifted);
            if (!isBarycentricOk())
            {
                toc[index] = vertices[i];
                index++;
            }
        }
        return toc;
    }

    Eigen::Vector3f getNormal()
    {
        return (vertices[0]->position - vertices[2]->position).cross(vertices[0]->position - vertices[1]->position);
    }

    Eigen::Vector2f getMeanTexCoord()
    {
        return (vertices[0]->texcoord + vertices[1]->texcoord + vertices[2]->texcoord) / 3.0;
    };

    Eigen::Vector3f getMeanPosition()
    {
        return (vertices[0]->position + vertices[1]->position + vertices[2]->position) / 3.0;
    };

    float getTexArea()
    {
        float area = vertices[0]->texcoord(0) * (vertices[1]->texcoord(1) - vertices[2]->texcoord(1));
        area += vertices[1]->texcoord(0) * (vertices[2]->texcoord(1) - vertices[0]->texcoord(1));
        area += vertices[2]->texcoord(0) * (vertices[0]->texcoord(1) - vertices[1]->texcoord(1));
        return area;
    }

    float getArea()
    {
        return getNormal().norm() / 2.0;
    }

    void arrageClockwise(Eigen::Vector3f &reference)
    {
        if (reference.dot(getNormal()) <= 0)
        {
            VerticeCPU *temp = vertices[1];
            vertices[1] = vertices[2];
            vertices[2] = temp;
        }
    }

    bool isBackFace()
    {
        Eigen::Vector3f f_tri_nor = getNormal();
        // back-face culling
        float point_dot_normal = vertices[0]->position.dot(f_tri_nor);
        if (point_dot_normal <= 0.0)
            return true;
        return false;
    };

    void computeTinv()
    {
        Eigen::Matrix2f T;
        T(0, 0) = vertices[0]->texcoord(0) - vertices[2]->texcoord(0);
        T(0, 1) = vertices[1]->texcoord(0) - vertices[2]->texcoord(0);
        T(1, 0) = vertices[0]->texcoord(1) - vertices[2]->texcoord(1);
        T(1, 1) = vertices[1]->texcoord(1) - vertices[2]->texcoord(1);
        T_inv = T.inverse();
    };

    void computeBarycentric(Eigen::Vector2f p)
    {
        Eigen::Vector2f lambda = T_inv * (p - vertices[2]->texcoord);
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
        if (barycentric(0) < 0.01 || barycentric(1) < 0.01 || barycentric(2) < 0.01)
            return true;
        return false;
    };

    bool isPoint()
    {
        if (barycentric(0) > 0.99 || barycentric(1) > 0.99 || barycentric(2) > 0.99)
            return true;
        return false;
    };

    std::array<Eigen::Vector2f, 2> getMinMax()
    {
        std::array<Eigen::Vector2f, 2> minmax;
        minmax[0](0) = std::min(std::min(vertices[0]->texcoord(0), vertices[1]->texcoord(0)), vertices[2]->texcoord(0));
        minmax[0](1) = std::min(std::min(vertices[0]->texcoord(1), vertices[1]->texcoord(1)), vertices[2]->texcoord(1));
        minmax[1](0) = std::max(std::max(vertices[0]->texcoord(0), vertices[1]->texcoord(0)), vertices[2]->texcoord(0));
        minmax[1](1) = std::max(std::max(vertices[0]->texcoord(1), vertices[1]->texcoord(1)), vertices[2]->texcoord(1));

        return minmax;
    };

    std::array<VerticeCPU *, 3> vertices;

    Eigen::Matrix2f T_inv;
    Eigen::Vector3f barycentric;

private:
};