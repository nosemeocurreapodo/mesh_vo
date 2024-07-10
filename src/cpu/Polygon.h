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
    PolygonFlat(Eigen::Vector3f vert1, Eigen::Vector3f vert2, Eigen::Vector3f vert3)
    {
        vertices[0] = vert1;
        vertices[1] = vert2;
        vertices[2] = vert3;

        computeNormal();
    };

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

private:
    Eigen::Vector3f vertices[3];
    Eigen::Vector3f normal;

    void computeNormal()
    {
        normal = ((vertices[0] - vertices[2]).cross(vertices[0] - vertices[1]));
    }
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
        return getNormal().norm() / 2.0;
    }

private:
    Eigen::Vector3f vertices[3];
    Eigen::Vector3f normals[3];
};
