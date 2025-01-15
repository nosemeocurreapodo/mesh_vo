#pragma once

#include "common/types.h"
#include "common/camera.h"
#include "common/window.h"
#include "common/depthParam.h"

/*
template <typename jacType, typename idsType>
class ShapeBase
{
public:
    virtual float getScreenArea() = 0;
    virtual window getScreenBounds() = 0;
    virtual vec2<float> getCenterPix() = 0;
    virtual bool isPixInShape(vec2<float> p) = 0;
    virtual bool isEdge(vec2<float> p) = 0;
    virtual float getDepth(vec2<float> p) = 0;
    virtual float getWeight(vec2<float> p) = 0;
    virtual vec2<float> getPix(vec2<float> p, ShapeBase *shape) = 0;
    virtual float getDepth(vec2<float> p, ShapeBase *shape) = 0;
    virtual jacType getParamJacobian(vec2<float> p) = 0;
    virtual idsType getParamIds() = 0;
};
*/

class ShapePatch // : public ShapeBase<float, int>
{
public:
    ShapePatch(vertex vert, int i, float w, float h)
    {
        set(vert, i, w, h);
    };

    void set(vertex vert, int i, float w, float h)
    {
        // centerRay = r;
        m_vert = vert;
        paramId = i;
        width = w;
        height = h;
        d_z_d_param = d_depth_d_param(m_vert.ver(2));
    }

    float getScreenArea()
    {
        return width * height;
    }

    window getScreenBounds()
    {
        int min_x = m_vert.pix(0) - width / 2;
        int max_x = m_vert.pix(0) + width / 2;
        int min_y = m_vert.pix(1) - height / 2;
        int max_y = m_vert.pix(1) + height / 2;

        window win(min_x, max_x, min_y, max_y);

        return win;
    }

    vec2f getCenterPix()
    {
        return m_vert.pix;
    }

    // void prepareForRay(vec3<float> r) override
    //{
    //     ray_diff = r - ray;
    // }

    bool isPixInShape(vec2f p)
    {
        return true;
    }

    bool isEdge(vec2f p)
    {
        return false;
    }

    // vec3<float> getRay(ShapeBase *shape)
    //{
    //     ShapePatch *sh = (ShapePatch *)shape;
    //     return sh->ray + ray_diff;
    // }

    // void prepareForPix(vec2<float> p)
    //{
    //     pix_diff = p - centerPix;
    // }

    vec2f getPix(vec2f p, ShapePatch &shape)
    {
        if (!(currentPix == p))
        {
            currentPix = p;
            pix_diff = currentPix - m_vert.pix;
        }

        return shape.m_vert.pix + pix_diff;
    }

    float getDepth(vec2f p)
    {
        return m_vert.ver(2);
    }

    float getWeight(vec2f p)
    {
        return weight;
    }

    float getDepth(vec2f p, ShapePatch &shape)
    {
        return shape.m_vert.ver(2);
    }

    float getParamJacobian(vec2f p)
    {
        return d_z_d_param;
    }

    int getParamIds()
    {
        return paramId;
    }

private:
    vertex m_vert;
    float weight;
    int paramId;

    float width;
    float height;

    vec2f currentPix;
    // vec3<float> ray_diff;
    vec2f pix_diff;

    float d_z_d_param;
};

/*
class ShapeSurfel : public ShapeBase
{
public:
    ShapeSurfel(vec3<float> &v, vec3<float> &r, vec2<float> &p, vec3<float> &n, float rad)
    {
        centerVert = v;
        centerRay = r;
        centerPix = p;
        normal = n;
        radius = rad;

        centerVert_dot_normal = centerVert.dot(normal);
        area = M_PI * radius * radius;

        d_depth_d_theta = d_depth_d_param(vert(2));
    };

    float getArea() override
    {
        return area;
    }

    //void prepareForRay(vec3<float> r) override
    //{
    //    ray = r;
    //}

    float getDepth(vec2<float> p) override
    {
        float ray_depth = centerVert_dot_normal / ray.dot(normal);
        return ray_depth;
    }


    //float getScreenArea()
    //{
    //    float area = rays[0](0) * (rays[1](1) - rays[2](1));
    //    area += rays[1](0) * (rays[2](1) - rays[0](1));
    //    area += rays[2](0) * (rays[0](1) - rays[1](1));
    //    return area;
    //}

    bool hitsShape(vec2<float> p) override
    {
        // float depth = getRayDepth(ray);
        // Eigen::Vector3f point = ray * depth;
        // if ((point - vertice).norm() > radius)
        //     return false;
        return true;
    };

    bool isEdge(vec2<float> p) override
    {
        return false;
    }


    //window getScreenBounds(camera cam) override
   // {
     //   vec3<float> ray = vertice / vertice(2);
     //   vec2<float> pix = cam.rayToPix(ray);

  //      int min_x = pix(0) - radius;
  //      int max_x = pix(0) + radius;
    //    int min_y = pix(1) - radius;
      //  int max_y = pix(1) + radius;

    //    window win(min_x, max_x, min_y, max_y);

  //      return win;
//    };


    window getScreenBounds() override
    {
        vec3<float> ray = vertice / vertice(2);
        vec2<float> pix = cam.rayToPix(ray);

        int min_x = pix(0) - radius;
        int max_x = pix(0) + radius;
        int min_y = pix(1) - radius;
        int max_y = pix(1) + radius;

        window win(min_x, max_x, min_y, max_y);

        return win;
    };

    vec3<float> getJacobian()
    {
        float kf_ray_dot_normal = ray.dot(normal);

        float d_depth_d_surfel_depth = vert_ray.dot(normal) / kf_ray_dot_normal;
        float d_depth_d_normal_x = vertice(0) / kf_ray_dot_normal - ray(0) * vert_dot_normal / (kf_ray_dot_normal * kf_ray_dot_normal);
        float d_depth_d_normal_y = vertice(1) / kf_ray_dot_normal - ray(1) * vert_dot_normal / (kf_ray_dot_normal * kf_ray_dot_normal);

        vec3<float> J;

        // with respect to d
        J(0) = d_depth_d_surfel_depth * d_depth_d_theta;
        J(1) = d_depth_d_normal_x;
        J(2) = d_depth_d_normal_y;

        return J;
    }

private:
    vec3<float> centerVert;
    vec2<float> centerPix;
    vec3<float> centerNorm;
    float radius;

    vec2<float> currentPix;
    vec2<float> currentRay;

    vec3<float> vert_ray;
    float vert_dot_normal;
    float area;
    float d_depth_d_theta;
};
*/

class ShapeTriangleFlat // : public ShapeBase<vec3<float>, vec3<int>>
{
public:
    // ShapeTriangleFlat() {};

    ShapeTriangleFlat(vertex vert0, vertex vert1, vertex vert2, int id0, int id1, int id2)
    {
        set(vert0, vert1, vert2, id0, id1, id2);
    }

    void set(vertex vert0, vertex vert1, vertex vert2, int id0, int id1, int id2)
    {
        m_vert0 = vert0;
        m_vert1 = vert1;
        m_vert2 = vert2;

        m_paramId0 = id0;
        m_paramId1 = id1;
        m_paramId2 = id2;

        // computeNormal();
        m_d_kf_depth_d_param(0) = d_depth_d_param(m_vert0.ver(2));
        m_d_kf_depth_d_param(1) = d_depth_d_param(m_vert1.ver(2));
        m_d_kf_depth_d_param(2) = d_depth_d_param(m_vert2.ver(2));

        /*
        r_m1(0) = (*m_ray1)(1) - (*m_ray2)(1);
        r_m1(1) = (*m_ray2)(0) - (*m_ray1)(0);
        r_m2(0) = (*m_ray2)(1) - (*m_ray0)(1);
        r_m2(1) = (*m_ray0)(0) - (*m_ray2)(0);

        // Calculate the area of the triangle
        r_denominator = ((*m_ray1)(1) - (*m_ray2)(1)) * ((*m_ray0)(0) - (*m_ray2)(0)) +
                        ((*m_ray2)(0) - (*m_ray1)(0)) * ((*m_ray0)(1) - (*m_ray2)(1));
        */

        m_p_m1(0) = m_vert1.pix(1) - m_vert2.pix(1);// (*m_pix1)(1) - (*m_pix2)(1);
        m_p_m1(1) = m_vert2.pix(0) - m_vert1.pix(0);// (*m_pix2)(0) - (*m_pix1)(0);
        m_p_m2(0) = m_vert2.pix(1) - m_vert0.pix(1);// (*m_pix2)(1) - (*m_pix0)(1);
        m_p_m2(1) = m_vert0.pix(0) - m_vert2.pix(0);// (*m_pix0)(0) - (*m_pix2)(0);

        // Calculate the area of the triangle
        m_p_denominator = (m_vert1.pix(1) - m_vert2.pix(1)) * (m_vert0.pix(0) - m_vert2.pix(0)) + // ((*m_pix1)(1) - (*m_pix2)(1)) * ((*m_pix0)(0) - (*m_pix2)(0)) +
                        (m_vert2.pix(0) - m_vert1.pix(0)) * (m_vert0.pix(1) - m_vert2.pix(1)); //((*m_pix2)(0) - (*m_pix1)(0)) * ((*m_pix0)(1) - (*m_pix2)(1));
    }

    float getScreenArea()
    {
        return -m_p_denominator;
        // return normal.norm() / 2.0;
    }

    // void prepareForRay(vec3<float> r) override
    //{
    //     computeBarycentric(r);
    // }

    // void prepareForPix(vec2<float> p) override
    //{
    //     computeBarycentric(p);
    // }

    float getDepth(vec2f p)
    {
        if (!(m_currentPix == p))
        {
            m_currentPix = p;
            computeBarycentric(m_currentPix);
        }
        // float ray_depth = vertices[0].dot(normal) / ray.dot(normal);
        float ray_depth = m_barycentric(0) * m_vert0.ver(2) +
                          m_barycentric(1) * m_vert1.ver(2) +
                          m_barycentric(2) * m_vert2.ver(2);

        return ray_depth;
    }

    // vec3<float> getRay(ShapeBase *shape) override
    //{
    //     ShapeTriangleFlat *sh = (ShapeTriangleFlat *)shape;
    //     return *sh->m_ray0 * barycentric(0) + *sh->m_ray1 * barycentric(1) + *sh->m_ray2 * barycentric(2);
    // }

    vec2f getPix(vec2f p, ShapeTriangleFlat &shape)
    {
        if (!(m_currentPix == p))
        {
            m_currentPix = p;
            computeBarycentric(m_currentPix);
        }

        return shape.m_vert0.pix * m_barycentric(0) + shape.m_vert1.pix * m_barycentric(1) + shape.m_vert2.pix * m_barycentric(2);
    }

    float getDepth(vec2f p, ShapeTriangleFlat &shape)
    {
        if (!(m_currentPix == p))
        {
            m_currentPix = p;
            computeBarycentric(m_currentPix);
        }

        return shape.m_vert0.ver(2) * m_barycentric(0) + shape.m_vert1.ver(2) * m_barycentric(1) + shape.m_vert2.ver(2) * m_barycentric(2);
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

    bool isPixInShape(vec2f p)
    {
        if (!(m_currentPix == p))
        {
            m_currentPix = p;
            computeBarycentric(m_currentPix);
        }

        if (m_barycentric(0) < -0.0 || m_barycentric(1) < -0.0 || m_barycentric(2) < -0.0)
            return false;
        if (m_barycentric(0) > 1.0 || m_barycentric(1) > 1.0 || m_barycentric(2) > 1.0)
            return false;
        return true;
    };

    bool isEdge(vec2f p)
    {
        if (!(m_currentPix == p))
        {
            m_currentPix = p;
            computeBarycentric(m_currentPix);
        }

        if (m_barycentric(0) < 0.05 || m_barycentric(1) < 0.05 || m_barycentric(2) < 0.05)
            return true;
        if (m_barycentric(0) > 0.95 || m_barycentric(1) > 0.95 || m_barycentric(2) > 0.95)
            return true;
        return false;
    }

    /*
    window getScreenBounds(camera cam) override
    {
        vec2<float> screencoords[3];
        screencoords[0] = cam.rayToPix(*m_ray0);
        screencoords[1] = cam.rayToPix(*m_ray1);
        screencoords[2] = cam.rayToPix(*m_ray2);

        int min_x = std::min(std::min((int)screencoords[0](0), (int)screencoords[1](0)), (int)screencoords[2](0)) - 1;
        int max_x = std::max(std::max((int)screencoords[0](0), (int)screencoords[1](0)), (int)screencoords[2](0)) + 1;
        int min_y = std::min(std::min((int)screencoords[0](1), (int)screencoords[1](1)), (int)screencoords[2](1)) - 1;
        int max_y = std::max(std::max((int)screencoords[0](1), (int)screencoords[1](1)), (int)screencoords[2](1)) + 1;

        window win(min_x, max_x, min_y, max_y);

        return win;
    };
    */

    window getScreenBounds()
    {
        int min_x = std::min(std::min((int)m_vert0.pix(0), (int)m_vert1.pix(0)), (int)m_vert2.pix(0)) - 1;
        int max_x = std::max(std::max((int)m_vert0.pix(0), (int)m_vert1.pix(0)), (int)m_vert2.pix(0)) + 1;
        int min_y = std::min(std::min((int)m_vert0.pix(1), (int)m_vert1.pix(1)), (int)m_vert2.pix(1)) - 1;
        int max_y = std::max(std::max((int)m_vert0.pix(1), (int)m_vert1.pix(1)), (int)m_vert2.pix(1)) + 1;

        window win(min_x, max_x, min_y, max_y);

        return win;
    };

    vec2f getCenterPix()
    {
        vec2f center = (m_vert0.pix + m_vert1.pix + m_vert2.pix)/3.0;
        return center;
    };

    vec3f getParamJacobian(vec2f p)
    {
        if (!(m_currentPix == p))
        {
            m_currentPix = p;
            computeBarycentric(m_currentPix);
        }

        vec3f jac;

        vec3f d_depth_d_kf_depth = m_barycentric;

        jac(0) = d_depth_d_kf_depth(0) * m_d_kf_depth_d_param(0);
        jac(1) = d_depth_d_kf_depth(1) * m_d_kf_depth_d_param(1);
        jac(2) = d_depth_d_kf_depth(2) * m_d_kf_depth_d_param(2);

        return jac;
    }

    vec3i getParamIds()
    {
        vec3i ids;

        ids(0) = m_paramId0;
        ids(1) = m_paramId1;
        ids(2) = m_paramId2;

        return ids;
    }

    vec3f getBarycentric()
    {
        return m_barycentric;
    }

private:
    /*
    void computeNormal()
    {
        // normal = ((vertices[0] - vertices[2]).cross(vertices[0] - vertices[1]));
        // normal = ((vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]));
    }
    */
    /*
    void computeBarycentric(vec3<float> ray)
    {
        // barycentric(0) = (((*m_ray1)(1) - (*m_ray2)(1)) * (ray(0) - (*m_ray2)(0)) +
        //                  ((*m_ray2)(0) - (*m_ray1)(0)) * (ray(1) - (*m_ray2)(1))) /
        //                 denominator;
        // barycentric(1) = (((*m_ray2)(1) - (*m_ray0)(1)) * (ray(0) - (*m_ray2)(0)) +
        //                  ((*m_ray0)(0) - (*m_ray2)(0)) * (ray(1) - (*m_ray2)(1))) /
        //                 denominator;
        vec2<float> diff;
        diff(0) = ray(0) - m_vert2.ray(0);
        diff(1) = ray(1) - m_vert2.ray(1);
        m_barycentric(0) = m_r_m1.dot(diff) / m_r_denominator;
        m_barycentric(1) = m_r_m2.dot(diff) / m_r_denominator;
        m_barycentric(2) = 1.0f - m_barycentric(0) - m_barycentric(1);
    }
    */

    void computeBarycentric(vec2f pix)
    {
        // barycentric(0) = (((*m_ray1)(1) - (*m_ray2)(1)) * (ray(0) - (*m_ray2)(0)) +
        //                  ((*m_ray2)(0) - (*m_ray1)(0)) * (ray(1) - (*m_ray2)(1))) /
        //                 denominator;
        // barycentric(1) = (((*m_ray2)(1) - (*m_ray0)(1)) * (ray(0) - (*m_ray2)(0)) +
        //                  ((*m_ray0)(0) - (*m_ray2)(0)) * (ray(1) - (*m_ray2)(1))) /
        //                 denominator;
        vec2f diff = pix - m_vert2.pix;
        m_barycentric(0) = m_p_m1.dot(diff) / m_p_denominator;
        m_barycentric(1) = m_p_m2.dot(diff) / m_p_denominator;
        m_barycentric(2) = 1.0f - m_barycentric(0) - m_barycentric(1);
    }

    vertex m_vert0;
    vertex m_vert1;
    vertex m_vert2;

    int m_paramId0;
    int m_paramId1;
    int m_paramId2;

    vec2f m_currentPix;

    vec3f m_barycentric;

    //vec2<float> r_m1;
    //vec2<float> r_m2;
    //float r_denominator;

    vec2f m_p_m1;
    vec2f m_p_m2;
    float m_p_denominator;

    vec3f m_d_kf_depth_d_param;
};

/*
class ShapeTriangleSmooth : public ShapeBase
{
public:
    ShapeTriangleSmooth(vec3<float> &vert1, vec3<float> &vert2, vec3<float> &vert3,
                        vec3<float> &norm1, vec3<float> &norm2, vec3<float> &norm3)
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
        prepareForMapJacobian();
    };

    float getArea() override
    {
        return ((vertices[1] - vertices[0]).cross(vertices[2] - vertices[0])).norm() / 2.0;
    }

    //void prepareForRay(vec3<float> r) override
    //{
    //    ray = r;
    //    computeBarycentric(ray);
    //}

    float getDepth() override
    {

        //float depth_0 = vertices[0].dot(normals[0]) / ray.dot(normals[0]);
        //float depth_1 = vertices[1].dot(normals[1]) / ray.dot(normals[1]);
        //float depth_2 = vertices[2].dot(normals[2]) / ray.dot(normals[2]);

        // float depth = barycentric(0)*depth_0 + barycentric(1)*depth_1 + barycentric(2)*depth_2;
        //float depth = barfunc(barycentric(0)) * depth_0 +
        //              barfunc(barycentric(1)) * depth_1 +
        //              barfunc(barycentric(2)) * depth_2;


        float u = barycentric(1);
        float v = barycentric(2);
        float w = barycentric(0);

        float depth = b300(2) * pow(w, 3) + b030(2) * pow(u, 3) + b003(2) * pow(v, 3) +
                      b210(2) * 3 * pow(w, 2) * u + b120(2) * 3 * w * pow(u, 2) + b201(2) * 3 * pow(w, 2) * v +
                      b021(2) * 3 * pow(u, 2) * v + b102(2) * 3 * w * pow(v, 2) + b012(2) * 3 * pow(v, 2) * u +
                      b111(2) * 6 * w * u * v;

        return depth;
    }

    vec3<float> getRay(ShapeBase *shape)
    {
        ShapeTriangleSmooth *sh = (ShapeTriangleSmooth *)shape;
        return sh->rays[0] * barycentric(0) + sh->rays[1] * barycentric(1) + sh->rays[2] * barycentric(2);
    }


    //float getScreenArea()
    //{
    //    float area = rays[0](0) * (rays[1](1) - rays[2](1));
    //    area += rays[1](0) * (rays[2](1) - rays[0](1));
    //    area += rays[2](0) * (rays[0](1) - rays[1](1));
    //    return area;
    //}


    // std::array<float, 3> getScreenAngles()
    // {
    //     Eigen::Vector2f a = (rays[1] - rays[0]).normalized();
    //     Eigen::Vector2f b = (rays[2] - rays[0]).normalized();

    //     float cosalpha = a.dot(b);
    //     float alpha = acos(cosalpha);

    //     a = (rays[0] - rays[1]).normalized();
    //     b = (rays[2] - rays[1]).normalized();

    //     float cosbeta = a.dot(b);
    //     float beta = acos(cosbeta);

    //     float gamma = M_PI - alpha - beta;

    //     return {alpha, beta, gamma};
    // }


    bool hitsShape() override
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

    window getScreenBounds(camera cam) override
    {
        vec2<float> screencoords[3];
        screencoords[0] = cam.rayToPix(rays[0]);
        screencoords[1] = cam.rayToPix(rays[1]);
        screencoords[2] = cam.rayToPix(rays[2]);

        int min_x = (int)std::min(std::min(screencoords[0](0), screencoords[1](0)), screencoords[2](0)) - 1;
        int max_x = (int)std::max(std::max(screencoords[0](0), screencoords[1](0)), screencoords[2](0)) + 1;
        int min_y = (int)std::min(std::min(screencoords[0](1), screencoords[1](1)), screencoords[2](1)) - 1;
        int max_y = (int)std::max(std::max(screencoords[0](1), screencoords[1](1)), screencoords[2](1)) + 1;

        window win(min_x, max_x, min_y, max_y);

        return win;
    };

    std::vector<float> getJacobian(float d_f_i_d_kf_depth) override
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
    void prepareForMapJacobian()
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

        d_z_d_iz[0] = d_depth_d_param(vertices[0](2));
        d_z_d_iz[1] = d_depth_d_param(vertices[1](2));
        d_z_d_iz[2] = d_depth_d_param(vertices[2](2));
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

    //void computeNormal()
    //{
    //    normal = ((vertices[0] - vertices[2]).cross(vertices[0] - vertices[1]));
    //}

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
*/
