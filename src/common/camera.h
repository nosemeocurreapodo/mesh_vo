#pragma once

#include <cassert>
#include <vector>
#include "common/types.h"

class camera
{
public:
    camera(float _fx, float _fy, float _cx, float _cy, int _width, int _height)
    {
        width = _width;
        height = _height;

        fx = _fx;
        fy = _fy;
        cx = _cx;
        cy = _cy;

        fx_norm = fx / width;
        fy_norm = fy / height;
        cx_norm = cx / width;
        cy_norm = cy / height;

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

        fxinv_norm = fxinv * width;
        fyinv_norm = fyinv * height;
        cxinv_norm = cxinv * width;
        cyinv_norm = cyinv * height;
    }

    void resize(int _width, int _height)
    {
        float scale_x = float(_width) / width;
        float scale_y = float(_height) / height;

        width = _width;
        height = _height;

        fx = fx * scale_x;
        fy = fy * scale_y;
        cx = cx * scale_x;
        cy = cy * scale_y;

        computeKinv();
    }

    void resize(float scale)
    {
        int new_width = int(width * scale);
        int new_height = int(height * scale);
        resize(new_width, new_height);
    }

    bool isPixVisible(vec2f pix)
    {
        // the idea here is that if we have 3 pixels
        // the first goes from 0 to 1, the second 1 to 2, the third 2 to 3, and the forth from 3 to 4
        // so here the max is one more than the last pixel
        // if (pix(0) < window_min_x || pix(0) > window_max_x || pix(1) < window_min_y || pix(1) > window_max_y)
        //    return false;
        if (pix(0) < 0 || pix(0) >= width || pix(1) < 0 || pix(1) >= height)
            return false;
        return true;
    }

    bool isPixVisible(int x, int y)
    {
        // the idea here is that if we have 3 pixels
        // the first goes from 0 to 1, the second 1 to 2, the third 2 to 3, and the forth from 3 to 4
        // so here the max is one more than the last pixel
        // if (pix(0) < window_min_x || pix(0) > window_max_x || pix(1) < window_min_y || pix(1) > window_max_y)
        //    return false;
        if (x < 0 || x >= width || y < 0 || y >= height)
            return false;
        return true;
    }

    /*
    bool isPixVisibleNormalized(Eigen::Vector2f &norm_pix)
    {
        if (norm_pix(0) < 0.0 || norm_pix(0) > 1.0 || norm_pix(1) < 0.0 || norm_pix(1) > 1.0)
            return false;
        return true;
    }
    */

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

    vec3f pixToRay(int x, int y)
    {
        // vec3<float> ray;
        // ray(0) = fxinv * x + cxinv;
        // ray(1) = fyinv * y + cyinv;
        // ray(2) = 1.0;
        return vec3f(fxinv * x + cxinv, fyinv * y + cyinv, 1.0);
    }

    vec3f d_f_i_d_f_ver(vec2f d_f_i_d_pix, vec3f f_ver)
    {
        vec3f d_f_i_d_f_ver;
        d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * fx / f_ver(2);
        d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * fy / f_ver(2);
        d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

        return d_f_i_d_f_ver;
    }

    /*
    Eigen::Vector3f pixToRayNormalized(Eigen::Vector2f &normPix)
    {
        // fxinv_norm is already multiplyed by width
        Eigen::Vector3f ray;
        ray(0) = fxinv_norm * normPix[0] + cxinv;
        ray(1) = fyinv_norm * normPix[1] + cyinv;
        ray(2) = 1.0;
        return ray;
    }
    */

    bool operator==(camera c)
    {
        if (fx == c.fx && fy == c.fy && cx == c.cx && cy == c.cy &&
            width == c.width && height == c.height)
            return true;
        return false;
    }

    int width, height;

    float fx;
    float fy;
    float cx;
    float cy;

    float fxinv;
    float fyinv;
    float cxinv;
    float cyinv;

    float fx_norm;
    float fy_norm;
    float cx_norm;
    float cy_norm;

    float fxinv_norm;
    float fyinv_norm;
    float cxinv_norm;
    float cyinv_norm;
};

class cameraMipMap
{
public:
    cameraMipMap(float _fx, float _fy, float _cx, float _cy, int _width, int _height)
    {
        int width = _width;
        int height = _height;
        float fx = _fx;
        float fy = _fy;
        float cx = _cx;
        float cy = _cy;

        while (true)
        {
            camera __cam(fx, fy, cx, cy, width, height);
            width = int(width / 2);
            height = int(height / 2);
            fx = fx / 2;
            fy = fy / 2;
            cx = cx / 2;
            cy = cy / 2;

            cam.push_back(__cam);

            if (width < 1 || height < 1)
                break;
        }
    }

    cameraMipMap(camera _cam)
    {
        int width = _cam.width;
        int height = _cam.height;
        float fx = _cam.fx;
        float fy = _cam.fy;
        float cx = _cam.cx;
        float cy = _cam.cy;

        while (true)
        {
            camera __cam(fx, fy, cx, cy, width, height);
            width = int(width / 2);
            height = int(height / 2);
            fx = fx / 2;
            fy = fy / 2;
            cx = cx / 2;
            cy = cy / 2;

            cam.push_back(__cam);

            if (width < 1 || height < 1)
                break;
        }
    }

    camera &operator[](int c)
    {
        assert(c >= 0 && c < cam.size());
        return cam[c];
    }

private:
    std::vector<camera> cam;
};
