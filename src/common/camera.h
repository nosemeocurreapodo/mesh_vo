#pragma once

#include <cassert>
#include <vector>
#include "common/types.h"

class pinholeCamera
{
public:
    pinholeCamera(float _fx, float _fy, float _cx, float _cy, int _width, int _height)
    {
        fx = _fx / _width;
        fy = _fy / _height;
        cx = _cx / _width;
        cy = _cy / _height;
    }

    pinholeCamera(float _fx, float _fy, float _cx, float _cy)
    {
        fx = _fx;
        fy = _fy;
        cx = _cx;
        cy = _cy;
    }

    bool isPixVisible(vec2f pix)
    {
        // the idea here is that if we have 3 pixels
        // the first goes from 0 to 1, the second 1 to 2, the third 2 to 3, and the forth from 3 to 4
        // so here the max is one more than the last pixel
        // if (pix(0) < window_min_x || pix(0) > window_max_x || pix(1) < window_min_y || pix(1) > window_max_y)
        //    return false;
        if (pix(0) < 0 || pix(0) > 1.0 || pix(1) < 0 || pix(1) > 1.0)
            return false;
        return true;
    }

    /*
    vec2f pointToPix(vec3f point)
    {
        vec2f pix;
        pix(0) = fx * point(0) / point(2) + cx;
        pix(1) = fy * point(1) / point(2) + cy;
        return pix;
    }
    */

    vec2f rayToPix(vec3f ray)
    {
        vec2f pix;
        pix(0) = fx * ray(0) + cx;
        pix(1) = fy * ray(1) + cy;
        return pix;
        // return vec2<float>(fx * ray(0) + cx, fy * ray(1) + cy);
    }

    mat<float, 2, 3> d_pix_d_ver(vec3f ver)
    {
        mat<float, 2, 3> d_pix_d_ver;

        d_pix_d_ver(0, 0) = fx / ver(2);
        d_pix_d_ver(0, 1) = 0;
        d_pix_d_ver(0, 2) = -fx * ver(0) / (ver(2) * ver(2));

        d_pix_d_ver(1, 0) = 0;
        d_pix_d_ver(1, 1) = fy / ver(2);
        d_pix_d_ver(1, 2) = -fy * ver(1) / (ver(2) * ver(2));

        return d_pix_d_ver;
    }

    mat<float, 2, 4> d_pix_d_intrinsics(vec3f ray)
    {
        mat<float, 2, 4> d_pix_d_int;

        d_pix_d_int(0, 0) = ray(0);
        d_pix_d_int(0, 1) = 0;
        d_pix_d_int(0, 2) = 1.0;
        d_pix_d_int(0, 3) = 0;

        d_pix_d_int(1, 0) = 0;
        d_pix_d_int(1, 1) = ray(1);
        d_pix_d_int(1, 2) = 0;
        d_pix_d_int(1, 3) = 1.0;

        return d_pix_d_int;
    }

    /*
    vec3f pixToRay(vec2f pix)
    {
        vec3f ray;
        ray(0) = (pix(0) - cx) / fx;
        ray(1) = (pix(1) - cy) / fy;
        ray(2) = 1.0;
        return ray;
    }
    */

    /*
    mat<float, 3, 4> d_ray_d_intrinsics(vec2f pix)
    {
        mat<float, 3, 4> d_ray_d_int;

        d_ray_d_int(0, 0) = -(pix(0) - cx) / (fx * fx);
        d_ray_d_int(0, 1) = 0.0;
        d_ray_d_int(0, 2) = -1.0 / fx;
        d_ray_d_int(0, 3) = 0.0;

        d_ray_d_int(1, 0) = 0.0;
        d_ray_d_int(1, 1) = -(pix(1) - cy) / (fy * fy);
        d_ray_d_int(1, 2) = 0.0;
        d_ray_d_int(1, 3) = -1.0 / fy;

        d_ray_d_int(2, 0) = 0.0;
        d_ray_d_int(2, 1) = 0.0;
        d_ray_d_int(2, 2) = 0.0;
        d_ray_d_int(2, 3) = 0.0;

        return d_ray_d_int;
    }
    */

    vec4f getParams()
    {
        return vec4f(fx, fy, cx, cy);
    }

    void setParams(vec4f params)
    {
        fx = params(0);
        fy = params(1);
        cx = params(2);
        cy = params(3);
    }

    /*
    bool operator==(pinholeCamera c)
    {
        if (fx == c.fx && fy == c.fy && cx == c.cx && cy == c.cy)
            return true;
        return false;
    }
    */

private:
    float fx;
    float fy;
    float cx;
    float cy;
};

class pinholeDistortedCamera
{
public:
    pinholeDistortedCamera(float _fx, float _fy, float _cx, float _cy, int _width, int _height)
    {
        fx = _fx / _width;
        fy = _fy / _height;
        cx = _cx / _width;
        cy = _cy / _height;
        k1 = 0.0;
        // k2 = 0.0;
        // p1 = 0.0;
        // p2 = 0.0;
        // k3 = 0.0;
    }

    pinholeDistortedCamera(float _fx, float _fy, float _cx, float _cy)
    {
        fx = _fx;
        fy = _fy;
        cx = _cx;
        cy = _cy;
        k1 = 0.0;
        // k2 = 0.0;
        // p1 = 0.0;
        // p2 = 0.0;
        // k3 = 0.0;
    }

    bool isPixVisible(vec2f pix)
    {
        // the idea here is that if we have 3 pixels
        // the first goes from 0 to 1, the second 1 to 2, the third 2 to 3, and the forth from 3 to 4
        // so here the max is one more than the last pixel
        // if (pix(0) < window_min_x || pix(0) > window_max_x || pix(1) < window_min_y || pix(1) > window_max_y)
        //    return false;
        if (pix(0) < 0 || pix(0) > 1.0 || pix(1) < 0 || pix(1) > 1.0)
            return false;
        return true;
    }

    vec2f rayToPix(vec3f ray)
    {
        vec3f distRay = distortRay(ray);
        vec2f pixel(fx * distRay(0) + cx, fy * distRay(1) + cy);
        return pixel;
    }
    /*
    mat<float, 2, 3> d_pix_d_ver(vec3f ver)
    {
        mat<float, 2, 3> d_pix_d_ver;

        d_pix_d_ver(0, 0) = fx / ver(2);
        d_pix_d_ver(0, 1) = 0;
        d_pix_d_ver(0, 2) = -fx * ver(0) / (ver(2) * ver(2));

        d_pix_d_ver(1, 0) = 0;
        d_pix_d_ver(1, 1) = fy / ver(2);
        d_pix_d_ver(1, 2) = -fy * ver(1) / (ver(2) * ver(2));

        return d_pix_d_ver;
    }
    */
    mat<float, 2, 3> d_pix_d_ver(vec3f ver)
    {
        vec3f ray = ver / ver(2);

        mat<float, 2, 3> d_pix_d_dist = d_pix_d_distRay();
        mat3f d_dis_d_ray = d_distray_d_ray(ray);
        mat3f d_ray_d_v = d_ray_d_ver(ver);

        mat<float, 2, 3> d_pix_d_ray = d_pix_d_dist * d_dis_d_ray;
        mat<float, 2, 3> d_pix_d_v = d_pix_d_ray * d_ray_d_v;

        return d_pix_d_v;
    }

    mat<float, 2, 5> d_pix_d_intrinsics(vec3f ray)
    {
        mat<float, 2, 5> d_pix_d_int;

        vec3f distRay = distortRay(ray);
        float r2 = ray(0) * ray(0) + ray(1) * ray(1);

        d_pix_d_int(0, 0) = distRay(0);
        d_pix_d_int(0, 1) = 0;
        d_pix_d_int(0, 2) = 1.0;
        d_pix_d_int(0, 3) = 0;
        d_pix_d_int(0, 4) = fx * ray(0) * r2;
        // d_pix_d_int(0, 5) = fx * 2 * ray(0) * ray(1);

        d_pix_d_int(1, 0) = 0;
        d_pix_d_int(1, 1) = distRay(1);
        d_pix_d_int(1, 2) = 0;
        d_pix_d_int(1, 3) = 1.0;
        d_pix_d_int(1, 4) = fy * ray(1) * r2;
        // d_pix_d_int(1, 5) = fy * (r2 + 2 * ray(1) * ray(1));

        return d_pix_d_int;
    }

    /*
    vec3f pixToRay(vec2f pix)
    {
        vec3f distRay;
        distRay(0) = (pix(0) - cx) / fx;
        distRay(1) = (pix(1) - cy) / fy;
        distRay(2) = 1.0;

        vec3f ray = correctRay(distRay);
        return ray;
    }
    */

    /*
    mat<float, 3, 5> d_ray_d_intrinsics(vec2f pix)
    {
        // Step 2: Compute radius squared from the optical axis
        float r2 = ray(0) * ray(0) + ray(1) * ray(1);
        // float r4 = r2 * r2;
        // float r6 = r4 * r2;

        // Step 3: Compute the radial distortion factor
        float radial = 1 + k1 * r2; // + k2 * r4 + k3 * r6;

        // Step 4: Apply radial and tangential distortion
        float xDistorted = ray(0) * radial; // + 2 * p1 * ray(0) * ray(1);        // + p2 * (r2 + 2 * ray(0) * ray(0));
        float yDistorted = ray(1) * radial; // + p1 * (r2 + 2 * ray(1) * ray(1)); // + 2 * p2 * ray(0) * ray(1);

        mat<float, 3, 5> d_ray_d_int;

        d_ray_d_int(0, 0) = -(pix(0) - cx) / (fx * fx);
        d_ray_d_int(0, 1) = 0.0;
        d_ray_d_int(0, 2) = -1.0 / fx;
        d_ray_d_int(0, 3) = 0.0;
        d_ray_d_int(0, 4) = 0.0;

        d_ray_d_int(1, 0) = 0.0;
        d_ray_d_int(1, 1) = -(pix(1) - cy) / (fy * fy);
        d_ray_d_int(1, 2) = 0.0;
        d_ray_d_int(1, 3) = -1.0 / fy;
        d_ray_d_int(1, 4) = 0.0;

        d_ray_d_int(2, 0) = 0.0;
        d_ray_d_int(2, 1) = 0.0;
        d_ray_d_int(2, 2) = 0.0;
        d_ray_d_int(2, 3) = 0.0;
        d_ray_d_int(2, 4) = 0.0;

        return d_ray_d_int;
    }
    */

    vec5f getParams()
    {
        return vec5f(fx, fy, cx, cy, k1);
    }

    void setParams(vec5f params)
    {
        fx = params(0);
        fy = params(1);
        cx = params(2);
        cy = params(3);
        k1 = params(4);
        // p1 = params(5);
    }

    /*
    bool operator==(cameraDist c)
    {
        if (fx == c.fx && fy == c.fy && cx == c.cx && cy == c.cy)
            return true;
        return false;
    }
    */

private:
    vec3f distortRay(vec3f ray)
    {
        // Step 2: Compute radius squared from the optical axis
        float r2 = ray(0) * ray(0) + ray(1) * ray(1);
        // float r4 = r2 * r2;
        // float r6 = r4 * r2;

        // Step 3: Compute the radial distortion factor
        float radial = 1 + k1 * r2; // + k2 * r4 + k3 * r6;

        // Step 4: Apply radial and tangential distortion
        float xDistorted = ray(0) * radial; // + 2 * p1 * ray(0) * ray(1);        // + p2 * (r2 + 2 * ray(0) * ray(0));
        float yDistorted = ray(1) * radial; // + p1 * (r2 + 2 * ray(1) * ray(1)); // + 2 * p2 * ray(0) * ray(1);

        // Step 5: Convert to pixel coordinates using the focal length and principal point
        vec3f distRay(xDistorted, yDistorted, 1.0);

        return distRay;
    }

    mat<float, 2, 3> d_pix_d_distRay()
    {
        mat<float, 2, 3> d_pix_d_dist;

        d_pix_d_dist(0, 0) = fx;
        d_pix_d_dist(0, 1) = 0;
        d_pix_d_dist(0, 2) = 1.0;

        d_pix_d_dist(1, 0) = 0;
        d_pix_d_dist(1, 1) = fy;
        d_pix_d_dist(1, 2) = 1.0;

        return d_pix_d_dist;
    }

    mat3f d_distray_d_ray(vec3f ray)
    {
        mat3f d_dist_d_ray;

        float r2 = ray(0) * ray(0) + ray(1) * ray(1);
        float radial = 1.0 + k1 * r2;

        d_dist_d_ray(0, 0) = radial + ray(0) * k1 * ray(0);
        d_dist_d_ray(0, 1) = 0.0;
        d_dist_d_ray(0, 2) = 0.0;

        d_dist_d_ray(1, 0) = 0.0;
        d_dist_d_ray(1, 1) = radial + ray(1) * k1 * ray(1);
        d_dist_d_ray(1, 2) = 0.0;

        d_dist_d_ray(2, 0) = 0.0;
        d_dist_d_ray(2, 1) = 0.0;
        d_dist_d_ray(2, 2) = 0.0;

        return d_dist_d_ray;
    }

    mat3f d_ray_d_ver(vec3f ver)
    {
        mat3f d_ray_d_v;

        d_ray_d_v(0, 0) = 1.0 / ver(2);
        d_ray_d_v(0, 1) = 0.0;
        d_ray_d_v(0, 2) = -ver(0) / (ver(2) * ver(2));

        d_ray_d_v(1, 0) = 0.0;
        d_ray_d_v(1, 1) = 1.0 / ver(2);
        d_ray_d_v(1, 2) = -ver(1) / (ver(2) * ver(2));

        d_ray_d_v(2, 0) = 0.0;
        d_ray_d_v(2, 1) = 0.0;
        d_ray_d_v(2, 2) = 0.0;

        return d_ray_d_v;
    }

    /*
    vec3f corrRay(vec3f distRay)
    {
        // Step 2: Compute radius squared from the optical axis
        float r2 = ray(0) * ray(0) + ray(1) * ray(1);
        // float r4 = r2 * r2;
        // float r6 = r4 * r2;

        // Step 3: Compute the radial distortion factor
        float radial = 1 + k1 * r2; // + k2 * r4 + k3 * r6;

        // Step 4: Apply radial and tangential distortion
        float xDistorted = ray(0) * radial; // + 2 * p1 * ray(0) * ray(1);        // + p2 * (r2 + 2 * ray(0) * ray(0));
        float yDistorted = ray(1) * radial; // + p1 * (r2 + 2 * ray(1) * ray(1)); // + 2 * p2 * ray(0) * ray(1);

        // Step 5: Convert to pixel coordinates using the focal length and principal point
        vec3f distRay(xDistorted, yDistorted, 1.0);

        return distRay;
    }
    */

    float fx;
    float fy;
    float cx;
    float cy;

    float k1;
    // float k2;
    // float p1;
    // float p2;
    // float k3;
};
