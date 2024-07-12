#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/camera.h"

class Polygon
{
public:
    virtual Eigen::Vector3f getNormal(Eigen::Vector3f &ray) const = 0;
    virtual float getArea() const = 0;
    virtual std::array<int, 4> getScreenBounds(camera &cam) = 0;
    virtual float getRayDepth(Eigen::Vector3f &ray) const = 0;
    virtual bool isRayInPolygon(Eigen::Vector3f &ray) = 0;
    virtual std::vector<float> getJacobian(Eigen::Vector3f &d_f_i_d_f_ver, Eigen::Vector3f &kf_ray, Sophus::SE3f &relPose) const = 0;
};

class PolygonFlat : public Polygon
{
public:
    PolygonFlat(Eigen::Vector3f vert1, Eigen::Vector3f vert2, Eigen::Vector3f vert3, MapJacobianMethod jacMethod)
    {
        vertices[0] = vert1;
        vertices[1] = vert2;
        vertices[2] = vert3;

        rays[0] = vertices[0] / vertices[0](2);
        rays[1] = vertices[1] / vertices[1](2);
        rays[2] = vertices[2] / vertices[2](2);

        computeNormal();
        computeTinv();

        prepareForMapJacobian(jacMethod);
    };

    float getRayDepth(Eigen::Vector3f &ray) const override
    {
        float ray_depth = vertices[0].dot(normal) / ray.dot(normal);
        return ray_depth;
    }

    /*
    Eigen::Vector3f dDepthdVert(Eigen::Vector3f &ray)
    {
        Eigen::Vector3f dDepthdVert0 =
    }
    */

    Eigen::Vector3f getNormal(Eigen::Vector3f &ray) const override
    {
        return normal;
    }

    float getArea() const override
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

    bool isRayInPolygon(Eigen::Vector3f &ray) override
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

    std::array<int, 4> getScreenBounds(camera &cam) override
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

    std::vector<float> getJacobian(Eigen::Vector3f &d_f_i_d_f_ver, Eigen::Vector3f &kf_ray, Sophus::SE3f &relPose) const override
    {
        Eigen::Vector3f d_f_ver_d_kf_depth = relPose.rotationMatrix() * kf_ray;

        float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

        std::vector<float> J;
        for (int i = 0; i < 3; i++)
        {
            float n_p_dot_ray = n_p[i].dot(kf_ray);
            float d_kf_depth_d_z = d_n_d_z[i].dot(pr_p[i]) / n_p_dot_ray - n_p_dot_point[i] * d_n_d_z[i].dot(kf_ray) / (n_p_dot_ray * n_p_dot_ray);
            float d_f_i_d_z = d_f_i_d_kf_depth * d_kf_depth_d_z * d_z_d_iz[i];
            J.push_back(d_f_i_d_z);
        }

        return J;
    }

private:
    void prepareForMapJacobian(MapJacobianMethod jacMethod)
    {
        n_p[0] = (vertices[0] - vertices[1]).cross(vertices[2] - vertices[1]);
        n_p[1] = (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]);
        n_p[2] = (vertices[2] - vertices[1]).cross(vertices[0] - vertices[1]);

        Eigen::Vector3f pw2mpw1[3];
        pw2mpw1[0] = (vertices[2] - vertices[1]);
        pw2mpw1[1] = (vertices[2] - vertices[0]);
        pw2mpw1[2] = (vertices[0] - vertices[1]);

        n_p_dot_point[0] = n_p[0].dot(vertices[1]);
        n_p_dot_point[1] = n_p[1].dot(vertices[0]);
        n_p_dot_point[2] = n_p[2].dot(vertices[1]);

        pr_p[0] = vertices[1];
        pr_p[1] = vertices[0];
        pr_p[2] = vertices[1];

        for (int i = 0; i < 3; i++)
        {
            d_n_d_z[i] = vertices[i].cross(pw2mpw1[i]);
            // with respect to depth
            if (jacMethod == MapJacobianMethod::depthJacobian)
                d_z_d_iz[i] = 1.0;
            // with respecto to idepth (depth = 1/idepth)
            if (jacMethod == MapJacobianMethod::idepthJacobian)
                d_z_d_iz[i] = -(vertices[i](2) * vertices[i](2));
            // width respect to depth = exp(z)
            if (jacMethod == MapJacobianMethod::logDepthJacobian)
                d_z_d_iz[i] = vertices[i](2);
            if (jacMethod == MapJacobianMethod::logIdepthJacobian)
                d_z_d_iz[i] = -vertices[i](2);
        }
    }

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

    // for der
    Eigen::Vector3f n_p[3];
    Eigen::Vector3f d_n_d_z[3];
    Eigen::Vector3f pr_p[3];
    float n_p_dot_point[3];
    float d_z_d_iz[3];
};

class PolygonCircle : public Polygon
{
public:
    PolygonCircle(Eigen::Vector3f vert, Eigen::Vector3f norm, float rad, MapJacobianMethod jacMethod)
    {
        vertice = vert;
        normal = norm;
        radius = rad;

        vert_dot_normal = vertice.dot(normal);
        area = M_PI * radius * radius;
        vert_ray = vertice / vertice(2);

        prepareForMapJacobian(jacMethod);
    };

    float getRayDepth(Eigen::Vector3f &ray) const override
    {
        float ray_depth = vert_dot_normal / ray.dot(normal);
        return ray_depth;
    }

    Eigen::Vector3f getNormal(Eigen::Vector3f &ray) const override
    {
        return normal;
    }

    float getArea() const override
    {
        return area;
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
    bool isRayInPolygon(Eigen::Vector3f &ray) override
    {
        // float depth = getRayDepth(ray);
        // Eigen::Vector3f point = ray * depth;
        // if ((point - vertice).norm() > radius)
        //     return false;
        return true;
    };

    std::array<int, 4> getScreenBounds(camera &cam) override
    {
        Eigen::Vector3f ray = vertice / vertice(2);
        Eigen::Vector2f pix = cam.rayToPix(ray);

        std::array<int, 4> minmax;
        minmax[0] = pix(0) - radius;
        minmax[1] = pix(0) + radius;
        minmax[2] = pix(1) - radius;
        minmax[3] = pix(1) + radius;

        return minmax;
    };

    std::vector<float> getJacobian(Eigen::Vector3f &d_f_i_d_f_ver, Eigen::Vector3f &kf_ray, Sophus::SE3f &relPose) const override
    {
        Eigen::Vector3f d_f_ver_d_kf_depth = relPose.rotationMatrix() * kf_ray;
        float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

        float kf_ray_dot_normal = kf_ray.dot(normal);

        float d_depth_d_surfel_depth = vert_ray.dot(normal) / kf_ray_dot_normal;
        float d_depth_d_normal_x = vertice(0) / kf_ray_dot_normal - kf_ray(0) * vert_dot_normal / (kf_ray_dot_normal * kf_ray_dot_normal);
        float d_depth_d_normal_y = vertice(1) / kf_ray_dot_normal - kf_ray(1) * vert_dot_normal / (kf_ray_dot_normal * kf_ray_dot_normal);

        std::vector<float> J;

        // with respect to d
        J.push_back(d_f_i_d_kf_depth * d_depth_d_surfel_depth);
        J.push_back(d_f_i_d_kf_depth * d_depth_d_normal_x);
        J.push_back(d_f_i_d_kf_depth * d_depth_d_normal_y);

        return J;
    }

private:
    void prepareForMapJacobian(MapJacobianMethod jacMethod)
    {
        // with respect to depth
        if (jacMethod == MapJacobianMethod::depthJacobian)
            d_depth_d_theta = 1.0;
        // with respecto to idepth (depth = 1/idepth)
        if (jacMethod == MapJacobianMethod::idepthJacobian)
            d_depth_d_theta = -(vertice(2) * vertice(2));
        // width respect to depth = exp(z)
        if (jacMethod == MapJacobianMethod::logDepthJacobian)
            d_depth_d_theta = vertice(2);
        if (jacMethod == MapJacobianMethod::logIdepthJacobian)
            d_depth_d_theta = -vertice(2);
    }

    Eigen::Vector3f vertice;
    Eigen::Vector3f normal;
    float radius;

    Eigen::Vector3f vert_ray;
    float vert_dot_normal;
    float area;
    float d_depth_d_theta;
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

    float getRayDepth(Eigen::Vector3f &ray) const override
    {
        float ray_depth = 0.0;
        for (int i = 0; i < 3; i++)
            ray_depth += vertices[0].dot(normals[0]) / ray.dot(normals[0]);
        ray_depth /= 3.0;
        return ray_depth;
    }

    Eigen::Vector3f getNormal(Eigen::Vector3f &ray) const override
    {
        return normals[0];
    }

    float getArea() const override
    {
        return normals[0].norm() / 2.0;
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

    bool isRayInPolygon(Eigen::Vector3f &ray) override
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

    std::array<int, 4> getScreenBounds(camera &cam) override
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
