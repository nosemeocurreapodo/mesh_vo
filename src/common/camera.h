#pragma once

#include <cassert>
#include <vector>
#include "common/types.h"

class camera
{
public:
    camera(float _fx, float _fy, float _cx, float _cy, int _width, int _height)
    {
        fx = _fx / _width;
        fy = _fy / _height;
        cx = _cx / _width;
        cy = _cy / _height;

        computeKinv();
    }

    camera(float _fx, float _fy, float _cx, float _cy)
    {
        fx = _fx;
        fy = _fy;
        cx = _cx;
        cy = _cy;

        computeKinv();
    }

    void computeKinv()
    {
        mat3f K = mat3f::Zero();
        K(0, 0) = fx;
        K(1, 1) = fy;
        K(2, 2) = 1.0f;
        K(0, 2) = cx;
        K(1, 2) = cy;

        mat3f KInv = K.inverse();

        fxinv = KInv(0, 0);
        fyinv = KInv(1, 1);
        cxinv = KInv(0, 2);
        cyinv = KInv(1, 2);
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

    vec2f pointToPix(vec3f point)
    {
        vec2f pix;
        pix(0) = fx * point(0) / point(2) + cx;
        pix(1) = fy * point(1) / point(2) + cy;
        return pix;
    }

    vec2f rayToPix(vec3f ray)
    {
        vec2f pix;
        pix(0) = fx * ray(0) + cx;
        pix(1) = fy * ray(1) + cy;
        return pix;
        // return vec2<float>(fx * ray(0) + cx, fy * ray(1) + cy);
    }

    vec3f pixToRay(vec2f pix)
    {
        vec3f ray;
        ray(0) = fxinv * pix(0) + cxinv;
        ray(1) = fyinv * pix(1) + cyinv;
        ray(2) = 1.0;
        return ray;
    }

    matxf d_pix_d_ver(vec3f ver)
    {
        matxf d_pix_d_ver(2, 3);

        d_pix_d_ver(0, 0) = fx / ver(2);
        d_pix_d_ver(0, 1) = 0;
        d_pix_d_ver(0, 2) = -fx * ver(0) / (ver(2) * ver(2));

        d_pix_d_ver(1, 0) = 0;
        d_pix_d_ver(1, 1) = fy / ver(2);
        d_pix_d_ver(1, 2) = -fy * ver(1) / (ver(2) * ver(2));

        return d_pix_d_ver;
    }

    matxf d_pix_d_intrinsics(vec3f ray)
    {
        matxf d_pix_d_intrinsic(2, 4);

        d_pix_d_intrinsic(0, 0) = ray(0);
        d_pix_d_intrinsic(0, 1) = 0;
        d_pix_d_intrinsic(0, 2) = 1.0;
        d_pix_d_intrinsic(0, 3) = 0;

        d_pix_d_intrinsic(1, 0) = 0;
        d_pix_d_intrinsic(1, 1) = ray(1);
        d_pix_d_intrinsic(1, 2) = 0;
        d_pix_d_intrinsic(1, 3) = 1.0;

        return d_pix_d_intrinsic;
    }

    /*
    vec3f d_f_i_d_f_ver(vec2f d_f_i_d_pix, vec3f f_ver)
    {
        vec3f d_f_i_d_f_ver;
        d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * fx / f_ver(2);
        d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * fy / f_ver(2);
        d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

        return d_f_i_d_f_ver;
    }
    */

    bool operator==(camera c)
    {
        if (fx == c.fx && fy == c.fy && cx == c.cx && cy == c.cy)
            return true;
        return false;
    }

    float fx;
    float fy;
    float cx;
    float cy;

    float fxinv;
    float fyinv;
    float cxinv;
    float cyinv;
};

class cameraDist
{
public:
    cameraDist(float _fx, float _fy, float _cx, float _cy, int _width, int _height)
    {
        fx = _fx / _width;
        fy = _fy / _height;
        cx = _cx / _width;
        cy = _cy / _height;
        k1 = 0.0;
        // k2 = 0.0;
        p1 = 0.0;
        // p2 = 0.0;
        // k3 = 0.0;

        computeKinv();
    }

    cameraDist(float _fx, float _fy, float _cx, float _cy)
    {
        fx = _fx;
        fy = _fy;
        cx = _cx;
        cy = _cy;
        k1 = 0.0;
        // k2 = 0.0;
        p1 = 0.0;
        // p2 = 0.0;
        // k3 = 0.0;

        computeKinv();
    }

    void computeKinv()
    {
        mat3f K = mat3f::Zero();
        K(0, 0) = fx;
        K(1, 1) = fy;
        K(2, 2) = 1.0f;
        K(0, 2) = cx;
        K(1, 2) = cy;

        mat3f KInv = K.inverse();

        fxinv = KInv(0, 0);
        fyinv = KInv(1, 1);
        cxinv = KInv(0, 2);
        cyinv = KInv(1, 2);
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

    vec3f correctRay(vec3f ray)
    {
        // Step 2: Compute radius squared from the optical axis
        float r2 = ray(0) * ray(0) + ray(1) * ray(1);
        // float r4 = r2 * r2;
        // float r6 = r4 * r2;

        // Step 3: Compute the radial distortion factor
        float radial = 1 + k1 * r2; // + k2 * r4 + k3 * r6;

        // Step 4: Apply radial and tangential distortion
        float xDistorted = ray(0) * radial + 2 * p1 * ray(0) * ray(1);        // + p2 * (r2 + 2 * ray(0) * ray(0));
        float yDistorted = ray(1) * radial + p1 * (r2 + 2 * ray(1) * ray(1)); // + 2 * p2 * ray(0) * ray(1);

        // Step 5: Convert to pixel coordinates using the focal length and principal point
        vec3f corray(xDistorted, yDistorted, 1.0);

        return corray;
    }

    vec2f rayToPix(vec3f ray)
    {
        vec3f corray = correctRay(ray);
        vec2f pixel(fx * corray(0) + cx, fy * corray(1) + cy);
        return pixel;
    }

    vec3f pixToRay(vec2f pix)
    {
        vec3f ray;
        ray(0) = fxinv * pix(0) + cxinv;
        ray(1) = fyinv * pix(1) + cyinv;
        ray(2) = 1.0;
        return ray;
    }

    matxf d_pix_d_ver(vec3f ver)
    {
        vec3f ray = ver / ver(2);
        vec3f corray = correctRay(ray);

        // Step 2: Compute radius squared from the optical axis
        float r2 = ray(0) * ray(0) + ray(1) * ray(1);
        // float r4 = r2 * r2;
        // float r6 = r4 * r2;

        // Step 3: Compute the radial distortion factor
        float radial = 1 + k1 * r2; // + k2 * r4 + k3 * r6;

        // Step 4: Apply radial and tangential distortion
        float xDistorted = (ver(0) / ver(2)) * radial + 2 * p1 * (ver(0) / ver(2)) * (ver(1) / ver(2));        // + p2 * (r2 + 2 * ray(0) * ray(0));
        float yDistorted = (ver(1) / ver(2)) * radial + p1 * (r2 + 2 * ver(1) * ver(1) / (ver(2) * ver(2))); // + 2 * p2 * ray(0) * ray(1);

        vec2f pix(fx * xDistorted + cx, fy * yDistorted + cy);

        matxf d_pix_d_ver(2, 3);

        d_pix_d_ver(0, 0) = fx / ver(2);
        d_pix_d_ver(0, 1) = 0;
        d_pix_d_ver(0, 2) = -fx * ver(0) / (ver(2) * ver(2));

        d_pix_d_ver(1, 0) = 0;
        d_pix_d_ver(1, 1) = fy / ver(2);
        d_pix_d_ver(1, 2) = -fy * ver(1) / (ver(2) * ver(2));

        return d_pix_d_ver;
    }

    matxf d_pix_d_intrinsics(vec3f ray)
    {
        vec3f corrRay = correctRay(ray);
        float r2 = ray(0) * ray(0) + ray(1) * ray(1);

        d_pix_d_intrinsic(0, 0) = corrRay(0);
        d_pix_d_intrinsic(0, 1) = 0;
        d_pix_d_intrinsic(0, 2) = 1.0;
        d_pix_d_intrinsic(0, 3) = 0;
        d_pix_d_intrinsic(0, 4) = fx * ray(0) * r2;
        d_pix_d_intrinsic(0, 5) = fx * 2 * ray(0) * ray(1);

        d_pix_d_intrinsic(1, 0) = 0;
        d_pix_d_intrinsic(1, 1) = corrRay(1);
        d_pix_d_intrinsic(1, 2) = 0;
        d_pix_d_intrinsic(1, 3) = 1.0;
        d_pix_d_intrinsic(1, 4) = fy * ray(1) * r2;
        d_pix_d_intrinsic(1, 5) = fy * (r2 + 2 * ray(1) * ray(1));

        // vec3f d_f_i_d_f_ver;
        // d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * fx / f_ver(2);
        // d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * fy / f_ver(2);
        // d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

        return d_pix_d_intrinsic;
    }

    /*
    vec3f d_f_i_d_f_ver(vec2f d_f_i_d_pix, vec3f f_ver)
    {
        vec3f d_f_i_d_f_ver;
        d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * fx / f_ver(2);
        d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * fy / f_ver(2);
        d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

        return d_f_i_d_f_ver;
    }
    */

    bool operator==(cameraDist c)
    {
        if (fx == c.fx && fy == c.fy && cx == c.cx && cy == c.cy)
            return true;
        return false;
    }

    float fx;
    float fy;
    float cx;
    float cy;

    float k1;
    // float k2;
    float p1;
    // float p2;
    // float k3;

    float fxinv;
    float fyinv;
    float cxinv;
    float cyinv;
};
