#pragma once

#include "common/types.h"
#include "common/camera.h"

class ShapeBase
{
public:
    virtual float getArea() = 0;
    virtual inline std::array<int, 4> getScreenBounds(camera &cam) = 0;
    virtual inline void prepareForRay(vec3<float> &r) = 0;
    virtual inline bool rayHitsShape() = 0;
    virtual inline bool isEdge() = 0;
    virtual inline float getRayDepth() = 0;
    virtual inline vec3<float> getRay(ShapeBase *shape) = 0;
    virtual inline std::vector<float> getJacobian(float &d_f_i_d_kf_depth) = 0;
};

class ShapePatch : public ShapeBase
{
public:
    ShapePatch(vec3<float> vert, float w, float h, DepthJacobianMethod jacMethod)
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
        vec3<float> ray = vertice / vertice(2);
        vec2<float> pix = cam.rayToPix(ray);

        std::array<int, 4> minmax;
        minmax[0] = pix(0) - width / 2;
        minmax[1] = pix(0) + width / 2;
        minmax[2] = pix(1) - height / 2;
        minmax[3] = pix(1) + height / 2;

        return minmax;
    }

    inline void prepareForRay(vec3<float> &r) override
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
        switch (jacMethod)
        {
        case DepthJacobianMethod::depthJacobian:
            d_z_d_param = 1.0;
            break;
        case DepthJacobianMethod::idepthJacobian:
            d_z_d_param = -(vertice(0) * vertice(0));
            break;
        case DepthJacobianMethod::logDepthJacobian:
            d_z_d_param = vertice(0);
            break;
        case DepthJacobianMethod::logIdepthJacobian:
            d_z_d_param = -vertice(0);
            break;
        default:
            d_z_d_param = 1.0;
        }
    }

    vec3<float> vertice;
    vec3<float> ray;

    float width;
    float height;

    float d_z_d_param;
};

class ShapeSurfel : public ShapeBase
{
public:
    ShapeSurfel(vec3<float> &vert, vec3<float> &norm, float rad, DepthJacobianMethod jacMethod)
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

    inline void prepareForRay(vec3<float> &r) override
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
        vec3<float> ray = vertice / vertice(2);
        vec2<float> pix = cam.rayToPix(ray);

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

    vec3<float> vertice;
    vec3<float> normal;
    float radius;

    vec3<float> ray;

    vec3<float> vert_ray;
    float vert_dot_normal;
    float area;
    float d_depth_d_theta;
};

class ShapeTriangleFlat : public ShapeBase
{
public:
    /*
      ShapeTriangleFlat(Eigen::Vector3f &vert1, Eigen::Vector3f &vert2, Eigen::Vector3f &vert3, DepthJacobianMethod jacMethod)
          : m_vertice0(vert1),
            m_vertice1(vert2),
            m_vertice2(vert3)
      {
          m_depth0 = m_vertice0(2);
          m_depth1 = m_vertice1(2);
          m_depth2 = m_vertice2(2);

          m_ray0 = m_vertice0 / m_depth0;
          m_ray1 = m_vertice1 / m_depth1;
          m_ray2 = m_vertice2 / m_depth2;

          //computeNormal();
          prepareForMapJacobian(jacMethod);

          // Calculate the area of the triangle
          denominator = (m_ray1.y() - m_ray2.y()) * (m_ray0.x() - m_ray2.x()) +
                        (m_ray2.x() - m_ray1.x()) * (m_ray0.y() - m_ray2.y());
      };
      */

    // ShapeTriangleFlat() {};

    ShapeTriangleFlat(vec3<float> &ray0, vec3<float> &ray1, vec3<float> &ray2,
                      float &depth0, float &depth1, float &depth2, DepthJacobianMethod jacMethod)
        : m_depth0(depth0),
          m_depth1(depth1),
          m_depth2(depth2)
    {
        m_ray0 = &ray0;
        m_ray1 = &ray1;
        m_ray2 = &ray2;

        // computeNormal();
        prepareForMapJacobian(jacMethod);

        // Calculate the area of the triangle
        denominator = ((*m_ray1)(1) - (*m_ray2)(1)) * ((*m_ray0)(0) - (*m_ray2)(0)) +
                      ((*m_ray2)(0) - (*m_ray1)(0)) * ((*m_ray0)(1) - (*m_ray2)(1));
    };

    /*
    ShapeTriangleFlat(Eigen::Vector3f &ray0, Eigen::Vector3f &ray1, Eigen::Vector3f &ray2,
                      Eigen::Vector2f &pix0, Eigen::Vector2f &pix1, Eigen::Vector2f &pix2,
                      float &depth0, float &depth1, float &depth2, DepthJacobianMethod jacMethod)
        : m_ray0(ray0),
          m_ray1(ray1),
          m_ray2(ray2),
          m_pix0(pix0),
          m_pix1(pix1),
          m_pix2(pix2),
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
    */

    void set(vec3<float> &ray0, vec3<float> &ray1, vec3<float> &ray2,
             float &depth0, float &depth1, float &depth2, DepthJacobianMethod jacMethod)
    {
        m_ray0 = &ray0;
        m_ray1 = &ray1;
        m_ray2 = &ray2;

        m_depth0 = depth0;
        m_depth1 = depth1;
        m_depth2 = depth2;
        // computeNormal();
        prepareForMapJacobian(jacMethod);

        m1(0) = (*m_ray1)(1) - (*m_ray2)(1);
        m1(1) = (*m_ray2)(0) - (*m_ray1)(0);
        m2(0) = (*m_ray2)(1) - (*m_ray0)(1);
        m2(1) = (*m_ray0)(0) - (*m_ray2)(0);

        // Calculate the area of the triangle
        denominator = ((*m_ray1)(1) - (*m_ray2)(1)) * ((*m_ray0)(0) - (*m_ray2)(0)) +
                      ((*m_ray2)(0) - (*m_ray1)(0)) * ((*m_ray0)(1) - (*m_ray2)(1));
    }

    float getArea() override
    {
        return 1.0;
        // return normal.norm() / 2.0;
    }

    inline void prepareForRay(vec3<float> &r) override
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

    inline vec3<float> getRay(ShapeBase *shape)
    {
        ShapeTriangleFlat *sh = (ShapeTriangleFlat *)shape;
        return *sh->m_ray0 * barycentric(0) + *sh->m_ray1 * barycentric(1) + *sh->m_ray2 * barycentric(2);
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

    inline std::array<int, 4> getScreenBounds(camera &cam) override
    {
        vec2<float> screencoords[3];
        screencoords[0] = cam.rayToPix(*m_ray0);
        screencoords[1] = cam.rayToPix(*m_ray1);
        screencoords[2] = cam.rayToPix(*m_ray2);

        std::array<int, 4> minmax;
        minmax[0] = std::min(std::min((int)screencoords[0](0), (int)screencoords[1](0)), (int)screencoords[2](0)) - 1;
        minmax[1] = std::max(std::max((int)screencoords[0](0), (int)screencoords[1](0)), (int)screencoords[2](0)) + 1;
        minmax[2] = std::min(std::min((int)screencoords[0](1), (int)screencoords[1](1)), (int)screencoords[2](1)) - 1;
        minmax[3] = std::max(std::max((int)screencoords[0](1), (int)screencoords[1](1)), (int)screencoords[2](1)) + 1;

        return minmax;
    };

    inline std::vector<float> getJacobian(float &d_f_i_d_kf_depth) override
    {
        vec3<float> d_kf_depth_d_z = barycentric;
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
        d_z_d_iz[0] = -(m_depth0 * m_depth0);
        d_z_d_iz[1] = -(m_depth1 * m_depth1);
        d_z_d_iz[2] = -(m_depth2 * m_depth2);

        /*
        switch (jacMethod)
        {
        case DepthJacobianMethod::depthJacobian:
            d_z_d_iz[0] = 1.0;
            d_z_d_iz[1] = 1.0;
            d_z_d_iz[2] = 1.0;
            break;
        case DepthJacobianMethod::idepthJacobian:
            d_z_d_iz[0] = -(m_depth0 * m_depth0);
            d_z_d_iz[1] = -(m_depth1 * m_depth1);
            d_z_d_iz[2] = -(m_depth2 * m_depth2);
            break;
        case DepthJacobianMethod::logDepthJacobian:
            d_z_d_iz[0] = m_depth0;
            d_z_d_iz[1] = m_depth1;
            d_z_d_iz[2] = m_depth2;
            break;
        case DepthJacobianMethod::logIdepthJacobian:
            d_z_d_iz[0] = -m_depth0;
            d_z_d_iz[1] = -m_depth1;
            d_z_d_iz[2] = -m_depth2;
            break;
        default:
            d_z_d_iz[0] = 1.0;
            d_z_d_iz[1] = 1.0;
            d_z_d_iz[2] = 1.0;
        }
        */
    }

    void computeNormal()
    {
        // normal = ((vertices[0] - vertices[2]).cross(vertices[0] - vertices[1]));
        // normal = ((vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]));
    }

    inline void computeBarycentric(vec3<float> &ray)
    {
         //barycentric(0) = (((*m_ray1)(1) - (*m_ray2)(1)) * (ray(0) - (*m_ray2)(0)) +
         //                 ((*m_ray2)(0) - (*m_ray1)(0)) * (ray(1) - (*m_ray2)(1))) /
         //                denominator;
         //barycentric(1) = (((*m_ray2)(1) - (*m_ray0)(1)) * (ray(0) - (*m_ray2)(0)) +
         //                 ((*m_ray0)(0) - (*m_ray2)(0)) * (ray(1) - (*m_ray2)(1))) /
         //                denominator;
        vec2<float> diff;
        diff(0) = ray(0) - (*m_ray2)(0);
        diff(1) = ray(1) - (*m_ray2)(1);
        barycentric(0) = m1.dot(diff) / denominator;
        barycentric(1) = m2.dot(diff) / denominator;
        barycentric(2) = 1.0f - barycentric(0) - barycentric(1);
    }

    vec2<float> m1;
    vec2<float> m2;

    vec3<float> m_vertice0;
    vec3<float> m_vertice1;
    vec3<float> m_vertice2;

    float m_depth0;
    float m_depth1;
    float m_depth2;

    vec3<float> *m_ray0;
    vec3<float> *m_ray1;
    vec3<float> *m_ray2;

    vec2<float> m_pix0;
    vec2<float> m_pix1;
    vec2<float> m_pix2;

    // Eigen::Vector3f normal;

    vec3<float> ray;
    vec3<float> barycentric;

    float denominator;

    // for der
    float d_z_d_iz[3];
};

class ShapeTriangleSmooth : public ShapeBase
{
public:
    ShapeTriangleSmooth(vec3<float> vert1, vec3<float> vert2, vec3<float> vert3,
                        vec3<float> norm1, vec3<float> norm2, vec3<float> norm3,
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

    inline void prepareForRay(vec3<float> &r) override
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

    inline vec3<float> getRay(ShapeBase *shape)
    {
        ShapeTriangleSmooth *sh = (ShapeTriangleSmooth *)shape;
        return sh->rays[0] * barycentric(0) + sh->rays[1] * barycentric(1) + sh->rays[2] * barycentric(2);
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
        vec2<float> screencoords[3];
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
        vec3<float> d_kf_depth_d_z;
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
        vec3<float> P1 = vertices[0];
        vec3<float> P2 = vertices[1];
        vec3<float> P3 = vertices[2];

        vec3<float> N1 = normals[0] / normals[0].norm();
        vec3<float> N2 = normals[1] / normals[1].norm();
        vec3<float> N3 = normals[2] / normals[2].norm();

        b300 = P1;
        b030 = P2;
        b003 = P3;

        float w12 = (P2 - P1).dot(N1);
        float w21 = (P1 - P2).dot(N2);
        float w23 = (P3 - P2).dot(N2);
        float w32 = (P2 - P3).dot(N3);
        float w31 = (P1 - P3).dot(N3);
        float w13 = (P3 - P1).dot(N1);

        b210 = (P1 * 2 + P2 - N1 * w12) / 3;
        b120 = (P2 * 2 + P1 - N2 * w21) / 3;
        b021 = (P2 * 2 + P3 - N2 * w23) / 3;
        b012 = (P3 * 2 + P2 - N3 * w32) / 3;
        b102 = (P3 * 2 + P1 - N3 * w31) / 3;
        b201 = (P1 * 2 + P3 - N1 * w13) / 3;

        vec3<float> E = (b210 + b120 + b021 + b012 + b102 + b201) / 6;
        vec3<float> V = (P1 + P2 + P3) / 3;

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

    void computeBarycentric(vec3<float> &ray)
    {
        // Calculate the area of the triangle
        float denominator = (rays[1](1) - rays[2](1)) * (rays[0](0) - rays[2](0)) +
                            (rays[2](0) - rays[1](0)) * (rays[0](1) - rays[2](1));

        // Calculate the sub-areas
        barycentric(0) = ((rays[1](1) - rays[2](1)) * (ray(0) - rays[2](0)) +
                          (rays[2](0) - rays[1](0)) * (ray(1) - rays[2](1))) /
                         denominator;
        barycentric(1) = ((rays[2](1) - rays[0](1)) * (ray(0) - rays[2](0)) +
                          (rays[0](0) - rays[2](0)) * (ray(1) - rays[2](1))) /
                         denominator;
        barycentric(2) = 1.0f - barycentric(0) - barycentric(1);
    }

    vec3<float> vertices[3];
    vec3<float> rays[3];
    vec3<float> normals[3];

    vec3<float> ray;
    vec3<float> barycentric;

    vec3<float> b300;
    vec3<float> b030;
    vec3<float> b003;

    vec3<float> b210;
    vec3<float> b120;
    vec3<float> b021;
    vec3<float> b012;
    vec3<float> b102;
    vec3<float> b201;

    vec3<float> b111;

    // for der
    float d_z_d_iz[3];
};
