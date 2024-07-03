#pragma once

#include <Eigen/Core>
#include "params.h"

class camera
{
public:
    camera()
    {
    }
    camera(float _fx, float _fy, float _cx, float _cy, int _width, int _height)
    {
        width = _width;
        height = _height;

        fx = _fx;
        fy = _fy;
        cx = _cx;
        cy = _cy;

        fx_norm = fx/width;
        fy_norm = fy/height;
        cx_norm = cx/width;
        cy_norm = cy/height;

        computeKinv();
    }

    void computeKinv()
    {
        Eigen::Matrix3f K = Eigen::Matrix3f::Zero();
        K(0, 0) = fx;
        K(1, 1) = fy;
        K(2, 2) = 1.0f;
        K(0, 2) = cx;
        K(1, 2) = cy;

        Eigen::Matrix3f KInv = K.inverse();

        fxinv = KInv(0, 0);
        fyinv = KInv(1, 1);
        cxinv = KInv(0, 2);
        cyinv = KInv(1, 2);

        fxinv_norm = fxinv*width;
        fyinv_norm = fyinv*height;
        cxinv_norm = cxinv*width;
        cyinv_norm = cyinv*height;
    }

    void resize(int _width, int _height)
    {
        float scale_x = float(_width)/width;
        float scale_y = float(_height)/height;

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

    bool isPixVisible(Eigen::Vector2f &pix)
    {
        if (pix(0) < 0.0 || pix(0) >= width || pix(1) < 0.0 || pix(1) >= height)
            return false;
        return true;
    }

    bool isPixVisibleNormalized(Eigen::Vector2f &norm_pix)
    {
        if (norm_pix(0) < 0.0 || norm_pix(0) > 1.0 || norm_pix(1) < 0.0 || norm_pix(1) > 1.0)
            return false;
        return true;
    }

    Eigen::Vector2f rayToPix(Eigen::Vector3f &ray)
    {
        Eigen::Vector2f pix;
        pix(0) = fx * ray(0) + cx;
        pix(1) = fy * ray(1) + cy;
        return pix;
    }

    Eigen::Vector2f rayToPixNormalized(Eigen::Vector3f &ray)
    {
        Eigen::Vector2f pix;
        pix(0) = (fx_norm * ray(0) + cx_norm);
        pix(1) = (fy_norm * ray(1) + cy_norm);
        return pix;
    }

    Eigen::Vector3f pixToRay(Eigen::Vector2f &pix)
    {
        Eigen::Vector3f ray;
        ray(0) = fxinv * pix[0] + cxinv;
        ray(1) = fyinv * pix[1] + cyinv;
        ray(2) = 1.0;
        return ray;
    }

    Eigen::Vector3f pixToRayNormalized(Eigen::Vector2f &normPix)
    {
        //fxinv_norm is already multiplyed by width
        Eigen::Vector3f ray;
        ray(0) = fxinv_norm * normPix[0] + cxinv;
        ray(1) = fyinv_norm * normPix[1] + cyinv;
        ray(2) = 1.0;
        return ray;
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

private:
};
