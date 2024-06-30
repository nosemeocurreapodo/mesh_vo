#pragma once

#include <Eigen/Core>

class Triangle3D
{
public:

    Triangle3D(Eigen::Vector3f &vert1, Eigen::Vector3f &vert2, Eigen::Vector3f &vert3)
    {
        vertices[0] = vert1;
        vertices[1] = vert2;
        vertices[2] = vert3;

    };

    float getDepth(Eigen::Vector3f &ray)
    {
        float ray_depth = vertices[0].dot(getNormal()) / ray.dot(getNormal());
        return ray_depth;
    }

    Eigen::Vector3f getNormal()
    {
        return (vertices[0] - vertices[2]).cross(vertices[0] - vertices[1]);
    }

    Eigen::Vector3f getMean()
    {
        return (vertices[0] + vertices[1] + vertices[2]) / 3.0;
    };

    float getArea()
    {
        return getNormal().norm() / 2.0;
    }

    bool isBackFace()
    {
        Eigen::Vector3f f_tri_nor = getNormal();
        // back-face culling
        float point_dot_normal = vertices[0].dot(f_tri_nor);
        if (point_dot_normal <= 0.0)
            return true;
        return false;
    };


    std::array<Eigen::Vector3f, 3> vertices;
private:

};