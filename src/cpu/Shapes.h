#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/camera.h"

class ShapeBase
{
public:
    virtual float getArea() = 0;
    virtual std::array<int, 4> getScreenBounds(camera &cam) = 0;
    virtual inline void prepareForRay(Eigen::Vector3f &r) = 0;
    virtual inline bool rayHitsShape() = 0;
    virtual inline bool isEdge() = 0;
    virtual inline float getRayDepth() = 0;
    virtual inline Eigen::Vector3f getRay(ShapeBase *shape) = 0;
    virtual inline std::vector<float> getJacobian(float &d_f_i_d_kf_depth) = 0;
};

class ShapePatch : public ShapeBase
{
public:
    ShapePatch(Eigen::Vector3f vert, float w, float h, DepthJacobianMethod jacMethod)
    {
        vertice = vert;
        width = w;
        height = h;
        prepareForMapJacobian(jacMethod);
    };

    float getArea() override
    {
        return width * height;
    }

    std::array<int, 4> getScreenBounds(camera &cam) override
    {
        Eigen::Vector3f ray = vertice / vertice(2);
        Eigen::Vector2f pix = cam.rayToPix(ray);

        std::array<int, 4> minmax;
        minmax[0] = pix(0) - width / 2;
        minmax[1] = pix(0) + width / 2;
        minmax[2] = pix(1) - height / 2;
        minmax[3] = pix(1) + height / 2;

        return minmax;
    }

    inline void prepareForRay(Eigen::Vector3f &r) override
    {
        ray = r;
    }

    inline bool rayHitsShape() override
    {
        return true;
    }

    bool isEdge() override
    {
        return false;
    }

    inline float getRayDepth() override
    {
        return vertice(2);
    }

    inline std::vector<float> getJacobian(float &d_f_i_d_kf_depth) override
    {
        float d_f_i_d_param = d_f_i_d_kf_depth * d_z_d_param;
        std::vector<float> J;
        J.push_back(d_f_i_d_param);
        return J;
    }

private:
    void prepareForMapJacobian(DepthJacobianMethod jacMethod)
    {
        // with respect to depth
        if (jacMethod == DepthJacobianMethod::depthJacobian)
            d_z_d_param = 1.0;
        // with respecto to idepth (depth = 1/idepth)
        if (jacMethod == DepthJacobianMethod::idepthJacobian)
            d_z_d_param = -(vertice(2) * vertice(2));
        // width respect to depth = exp(z)
        if (jacMethod == DepthJacobianMethod::logDepthJacobian)
            d_z_d_param = vertice(2);
        if (jacMethod == DepthJacobianMethod::logIdepthJacobian)
            d_z_d_param = -vertice(2);
    }

    Eigen::Vector3f vertice;
    Eigen::Vector3f ray;

    float width;
    float height;

    float d_z_d_param;
};

class ShapeSurfel : public ShapeBase
{
public:
    ShapeSurfel(Eigen::Vector3f vert, Eigen::Vector3f norm, float rad, DepthJacobianMethod jacMethod)
    {
        vertice = vert;
        normal = norm;
        radius = rad;

        vert_dot_normal = vertice.dot(normal);
        area = M_PI * radius * radius;
        vert_ray = vertice / vertice(2);

        prepareForMapJacobian(jacMethod);
    };

    float getArea() override
    {
        return area;
    }

    inline void prepareForRay(Eigen::Vector3f &r) override
    {
        ray = r;
    }

    inline float getRayDepth() override
    {
        float ray_depth = vert_dot_normal / ray.dot(normal);
        return ray_depth;
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
    inline bool rayHitsShape() override
    {
        // float depth = getRayDepth(ray);
        // Eigen::Vector3f point = ray * depth;
        // if ((point - vertice).norm() > radius)
        //     return false;
        return true;
    };

    bool isEdge() override
    {
        return false;
    }

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

    inline std::vector<float> getJacobian(float &d_f_i_d_kf_depth) override
    {
        float kf_ray_dot_normal = ray.dot(normal);

        float d_depth_d_surfel_depth = vert_ray.dot(normal) / kf_ray_dot_normal;
        float d_depth_d_normal_x = vertice(0) / kf_ray_dot_normal - ray(0) * vert_dot_normal / (kf_ray_dot_normal * kf_ray_dot_normal);
        float d_depth_d_normal_y = vertice(1) / kf_ray_dot_normal - ray(1) * vert_dot_normal / (kf_ray_dot_normal * kf_ray_dot_normal);

        std::vector<float> J;

        // with respect to d
        J.push_back(d_f_i_d_kf_depth * d_depth_d_surfel_depth * d_depth_d_theta);
        J.push_back(d_f_i_d_kf_depth * d_depth_d_normal_x);
        J.push_back(d_f_i_d_kf_depth * d_depth_d_normal_y);

        return J;
    }

private:
    void prepareForMapJacobian(DepthJacobianMethod jacMethod)
    {
        // with respect to depth
        if (jacMethod == DepthJacobianMethod::depthJacobian)
            d_depth_d_theta = 1.0;
        // with respecto to idepth (depth = 1/idepth)
        if (jacMethod == DepthJacobianMethod::idepthJacobian)
            d_depth_d_theta = -(vertice(2) * vertice(2));
        // width respect to depth = exp(z)
        if (jacMethod == DepthJacobianMethod::logDepthJacobian)
            d_depth_d_theta = vertice(2);
        if (jacMethod == DepthJacobianMethod::logIdepthJacobian)
            d_depth_d_theta = -vertice(2);
    }

    Eigen::Vector3f vertice;
    Eigen::Vector3f normal;
    float radius;

    Eigen::Vector3f ray;

    Eigen::Vector3f vert_ray;
    float vert_dot_normal;
    float area;
    float d_depth_d_theta;
};

class ShapeTriangleFlat : public ShapeBase
{
public:
    /*
    ShapeTriangleFlat(Eigen::Vector3f &vert1, Eigen::Vector3f &vert2, Eigen::Vector3f &vert3, DepthJacobianMethod &jacMethod)
    {
        vertices[0] = vert1;
        vertices[1] = vert2;
        vertices[2] = vert3;

        rays[0] = vertices[0] / vertices[0](2);
        rays[1] = vertices[1] / vertices[1](2);
        rays[2] = vertices[2] / vertices[2](2);

        //computeNormal();
        prepareForMapJacobian(jacMethod);

        // Calculate the area of the triangle
        denominator = (rays[1].y() - rays[2].y()) * (rays[0].x() - rays[2].x()) +
                      (rays[2].x() - rays[1].x()) * (rays[0].y() - rays[2].y());
    };
    */

    //ShapeTriangleFlat() {};

    ShapeTriangleFlat(Eigen::Vector3f &ray0, Eigen::Vector3f &ray1, Eigen::Vector3f &ray2,
                      float &depth0, float &depth1, float &depth2, DepthJacobianMethod jacMethod)
        : m_ray0(ray0),
          m_ray1(ray1),
          m_ray2(ray2),
          m_depth0(depth0),
          m_depth1(depth1),
          m_depth2(depth2)
    {
        // computeNormal();
        prepareForMapJacobian(jacMethod);

        // Calculate the area of the triangle
        denominator = (m_ray1.y() - m_ray2.y()) * (m_ray0.x() - m_ray2.x()) +
                      (m_ray2.x() - m_ray1.x()) * (m_ray0.y() - m_ray2.y());
    };

    void set(Eigen::Vector3f &ray0, Eigen::Vector3f &ray1, Eigen::Vector3f &ray2,
             float &depth0, float &depth1, float &depth2, DepthJacobianMethod jacMethod)
    {
        m_ray0 = ray0;
        m_ray1 = ray1;
        m_ray2 = ray2;

        m_depth0 = depth0;
        m_depth1 = depth1;
        m_depth2 = depth2;
        // computeNormal();
        prepareForMapJacobian(jacMethod);

        // Calculate the area of the triangle
        denominator = (m_ray1.y() - m_ray2.y()) * (m_ray0.x() - m_ray2.x()) +
                      (m_ray2.x() - m_ray1.x()) * (m_ray0.y() - m_ray2.y());
    }

    float getArea() override
    {
        return 1.0;
        // return normal.norm() / 2.0;
    }

    inline void prepareForRay(Eigen::Vector3f &r) override
    {
        computeBarycentric(r);
        ray = r;
    }

    inline float getRayDepth() override
    {
        // float ray_depth = vertices[0].dot(normal) / ray.dot(normal);
        float ray_depth = barycentric(0) * m_depth0 +
                          barycentric(1) * m_depth1 +
                          barycentric(2) * m_depth2;

        return ray_depth;
    }

    inline Eigen::Vector3f getRay(ShapeBase *shape)
    {
        ShapeTriangleFlat *sh = (ShapeTriangleFlat *)shape;
        return barycentric(0) * sh->m_ray0 + barycentric(1) * sh->m_ray1 + barycentric(2) * sh->m_ray2;
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

    inline bool rayHitsShape() override
    {
        if (barycentric(0) < -0.0 || barycentric(1) < -0.0 || barycentric(2) < -0.0)
            return false;
        if (barycentric(0) > 1.0 || barycentric(1) > 1.0 || barycentric(2) > 1.0)
            return false;
        return true;
    };

    bool isEdge() override
    {
        if (barycentric(0) < 0.05 || barycentric(1) < 0.05 || barycentric(2) < 0.05)
            return true;
        if (barycentric(0) > 0.95 || barycentric(1) > 0.95 || barycentric(2) > 0.95)
            return true;
        return false;
    }

    std::array<int, 4> getScreenBounds(camera &cam) override
    {
        Eigen::Vector2f screencoords[3];
        screencoords[0] = cam.rayToPix(m_ray0);
        screencoords[1] = cam.rayToPix(m_ray1);
        screencoords[2] = cam.rayToPix(m_ray2);

        std::array<int, 4> minmax;
        minmax[0] = (int)std::min(std::min(screencoords[0](0), screencoords[1](0)), screencoords[2](0)) - 1;
        minmax[1] = (int)std::max(std::max(screencoords[0](0), screencoords[1](0)), screencoords[2](0)) + 1;
        minmax[2] = (int)std::min(std::min(screencoords[0](1), screencoords[1](1)), screencoords[2](1)) - 1;
        minmax[3] = (int)std::max(std::max(screencoords[0](1), screencoords[1](1)), screencoords[2](1)) + 1;

        return minmax;
    };

    inline std::vector<float> getJacobian(float &d_f_i_d_kf_depth) override
    {
        Eigen::Vector3f d_kf_depth_d_z = barycentric;
        std::vector<float> J;
        for (int i = 0; i < 3; i++)
        {
            float d_f_i_d_z = d_f_i_d_kf_depth * d_kf_depth_d_z(i) * d_z_d_iz[i];
            J.push_back(d_f_i_d_z);
        }

        return J;
    }

private:
    inline void prepareForMapJacobian(DepthJacobianMethod &jacMethod)
    {
        // with respecto to idepth (depth = 1/idepth)
        if (jacMethod == DepthJacobianMethod::idepthJacobian)
        {
            d_z_d_iz[0] = -(m_depth0 * m_depth0);
            d_z_d_iz[1] = -(m_depth1 * m_depth1);
            d_z_d_iz[2] = -(m_depth2 * m_depth2);
        }
        // with respect to depth
        if (jacMethod == DepthJacobianMethod::depthJacobian)
        {
            d_z_d_iz[0] = 1.0;
            d_z_d_iz[1] = 1.0;
            d_z_d_iz[2] = 1.0;
        }
        // width respect to depth = exp(z)
        if (jacMethod == DepthJacobianMethod::logDepthJacobian)
        {
            d_z_d_iz[0] = m_depth0;
            d_z_d_iz[1] = m_depth1;
            d_z_d_iz[2] = m_depth2;
        }
        if (jacMethod == DepthJacobianMethod::logIdepthJacobian)
        {
            d_z_d_iz[0] = -m_depth0;
            d_z_d_iz[1] = -m_depth1;
            d_z_d_iz[2] = -m_depth2;
        }
    }

    void computeNormal()
    {
        // normal = ((vertices[0] - vertices[2]).cross(vertices[0] - vertices[1]));
        // normal = ((vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]));
    }

    inline void computeBarycentric(Eigen::Vector3f &ray)
    {
        // Calculate the sub-areas
        barycentric(0) = ((m_ray1.y() - m_ray2.y()) * (ray.x() - m_ray2.x()) +
                          (m_ray2.x() - m_ray1.x()) * (ray.y() - m_ray2.y())) /
                         denominator;
        barycentric(1) = ((m_ray2.y() - m_ray0.y()) * (ray.x() - m_ray2.x()) +
                          (m_ray0.x() - m_ray2.x()) * (ray.y() - m_ray2.y())) /
                         denominator;
        barycentric(2) = 1.0f - barycentric(0) - barycentric(1);
    }

    // Eigen::Vector3f vertices[3];
    float m_depth0;
    float m_depth1;
    float m_depth2;
    Eigen::Vector3f m_ray0;
    Eigen::Vector3f m_ray1;
    Eigen::Vector3f m_ray2;

    // Eigen::Vector3f normal;

    Eigen::Vector3f ray;
    Eigen::Vector3f barycentric;

    float denominator;

    // for der
    float d_z_d_iz[3];
};

class ShapeTriangleSmooth : public ShapeBase
{
public:
    ShapeTriangleSmooth(Eigen::Vector3f vert1, Eigen::Vector3f vert2, Eigen::Vector3f vert3,
                        Eigen::Vector3f norm1, Eigen::Vector3f norm2, Eigen::Vector3f norm3,
                        DepthJacobianMethod jacMethod)
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

        // computeNormal();
        prepareForMapJacobian(jacMethod);
    };

    float getArea() override
    {
        return ((vertices[1] - vertices[0]).cross(vertices[2] - vertices[0])).norm() / 2.0;
    }

    inline void prepareForRay(Eigen::Vector3f &r) override
    {
        ray = r;
        computeBarycentric(ray);
    }

    inline float getRayDepth() override
    {
        /*
        float depth_0 = vertices[0].dot(normals[0]) / ray.dot(normals[0]);
        float depth_1 = vertices[1].dot(normals[1]) / ray.dot(normals[1]);
        float depth_2 = vertices[2].dot(normals[2]) / ray.dot(normals[2]);

        // float depth = barycentric(0)*depth_0 + barycentric(1)*depth_1 + barycentric(2)*depth_2;
        float depth = barfunc(barycentric(0)) * depth_0 +
                      barfunc(barycentric(1)) * depth_1 +
                      barfunc(barycentric(2)) * depth_2;
        */

        float u = barycentric(1);
        float v = barycentric(2);
        float w = barycentric(0);

        float depth = b300(2) * pow(w, 3) + b030(2) * pow(u, 3) + b003(2) * pow(v, 3) +
                      b210(2) * 3 * pow(w, 2) * u + b120(2) * 3 * w * pow(u, 2) + b201(2) * 3 * pow(w, 2) * v +
                      b021(2) * 3 * pow(u, 2) * v + b102(2) * 3 * w * pow(v, 2) + b012(2) * 3 * pow(v, 2) * u +
                      b111(2) * 6 * w * u * v;

        return depth;
    }

    inline Eigen::Vector3f getRay(ShapeBase *shape)
    {
        ShapeTriangleSmooth *sh = (ShapeTriangleSmooth *)shape;
        return barycentric(0) * sh->rays[0] + barycentric(1) * sh->rays[1] + barycentric(2) * sh->rays[2];
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

    inline bool rayHitsShape() override
    {
        if (barycentric(0) < -0.0 || barycentric(1) < -0.0 || barycentric(2) < -0.0)
            return false;
        if (barycentric(0) > 1.0 || barycentric(1) > 1.0 || barycentric(2) > 1.0)
            return false;
        return true;
    };

    bool isEdge() override
    {
        if (barycentric(0) < 0.05 || barycentric(1) < 0.05 || barycentric(2) < 0.05)
            return true;
        if (barycentric(0) > 0.95 || barycentric(1) > 0.95 || barycentric(2) > 0.95)
            return true;
        return false;
    }

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

    inline std::vector<float> getJacobian(float &d_f_i_d_kf_depth) override
    {
        Eigen::Vector3f d_kf_depth_d_z;
        d_kf_depth_d_z(0) = barycentric(0); // * rays[0].dot(normals[0]) / ray.dot(normals[0]);
        d_kf_depth_d_z(1) = barycentric(1); // * rays[1].dot(normals[1]) / ray.dot(normals[1]);
        d_kf_depth_d_z(2) = barycentric(2); // * rays[2].dot(normals[2]) / ray.dot(normals[2]);

        // Eigen::Vector3f d_kf_depth_d_nx;
        // d_kf_depth_d_nx(0) = barycentric(0) * (rays[0](0) / ray.dot(normals[0]) -
        //                     rays[0].dot(normals[0])/(pow(ray.dot(normals[0]), 2))*ray(0));

        std::vector<float> J;
        for (int i = 0; i < 3; i++)
        {
            float d_f_i_d_z = d_f_i_d_kf_depth * d_kf_depth_d_z(i) * d_z_d_iz[i];
            J.push_back(d_f_i_d_z);
        }

        return J;
    }

private:
    void prepareForMapJacobian(DepthJacobianMethod jacMethod)
    {
        Eigen::Vector3f P1 = vertices[0];
        Eigen::Vector3f P2 = vertices[1];
        Eigen::Vector3f P3 = vertices[2];

        Eigen::Vector3f N1 = normals[0].normalized();
        Eigen::Vector3f N2 = normals[1].normalized();
        Eigen::Vector3f N3 = normals[2].normalized();

        b300 = P1;
        b030 = P2;
        b003 = P3;

        float w12 = (P2 - P1).dot(N1);
        float w21 = (P1 - P2).dot(N2);
        float w23 = (P3 - P2).dot(N2);
        float w32 = (P2 - P3).dot(N3);
        float w31 = (P1 - P3).dot(N3);
        float w13 = (P3 - P1).dot(N1);

        b210 = (2 * P1 + P2 - w12 * N1) / 3;
        b120 = (2 * P2 + P1 - w21 * N2) / 3;
        b021 = (2 * P2 + P3 - w23 * N2) / 3;
        b012 = (2 * P3 + P2 - w32 * N3) / 3;
        b102 = (2 * P3 + P1 - w31 * N3) / 3;
        b201 = (2 * P1 + P3 - w13 * N1) / 3;

        Eigen::Vector3f E = (b210 + b120 + b021 + b012 + b102 + b201) / 6;
        Eigen::Vector3f V = (P1 + P2 + P3) / 3;

        b111 = E + (E - V) / 2;

        for (int i = 0; i < 3; i++)
        {
            // with respect to depth
            if (jacMethod == DepthJacobianMethod::depthJacobian)
                d_z_d_iz[i] = 1.0;
            // with respecto to idepth (depth = 1/idepth)
            if (jacMethod == DepthJacobianMethod::idepthJacobian)
                d_z_d_iz[i] = -(vertices[i](2) * vertices[i](2));
            // width respect to depth = exp(z)
            if (jacMethod == DepthJacobianMethod::logDepthJacobian)
                d_z_d_iz[i] = vertices[i](2);
            if (jacMethod == DepthJacobianMethod::logIdepthJacobian)
                d_z_d_iz[i] = -vertices[i](2);
        }
    }

    float barfunc(float input)
    {
        // input goes from 0 to 1
        // sigmoid input should go from -5 to 5
        // float in = input*10.0*2.0 - 10.0;
        // float output = 1.0/(1 + exp(-in));
        // float output = input*input;
        float output = std::sin(input * M_PI / 2);
        return output;
    }

    /*
    void computeNormal()
    {
        normal = ((vertices[0] - vertices[2]).cross(vertices[0] - vertices[1]));
    }
    */

    void computeBarycentric(Eigen::Vector3f &ray)
    {
        // Calculate the area of the triangle
        float denominator = (rays[1].y() - rays[2].y()) * (rays[0].x() - rays[2].x()) +
                            (rays[2].x() - rays[1].x()) * (rays[0].y() - rays[2].y());

        // Calculate the sub-areas
        barycentric(0) = ((rays[1].y() - rays[2].y()) * (ray.x() - rays[2].x()) +
                          (rays[2].x() - rays[1].x()) * (ray.y() - rays[2].y())) /
                         denominator;
        barycentric(1) = ((rays[2].y() - rays[0].y()) * (ray.x() - rays[2].x()) +
                          (rays[0].x() - rays[2].x()) * (ray.y() - rays[2].y())) /
                         denominator;
        barycentric(2) = 1.0f - barycentric(0) - barycentric(1);
    }

    Eigen::Vector3f vertices[3];
    Eigen::Vector3f rays[3];
    Eigen::Vector3f normals[3];

    Eigen::Vector3f ray;
    Eigen::Vector3f barycentric;

    Eigen::Vector3f b300;
    Eigen::Vector3f b030;
    Eigen::Vector3f b003;

    Eigen::Vector3f b210;
    Eigen::Vector3f b120;
    Eigen::Vector3f b021;
    Eigen::Vector3f b012;
    Eigen::Vector3f b102;
    Eigen::Vector3f b201;

    Eigen::Vector3f b111;

    // for der
    float d_z_d_iz[3];
};
