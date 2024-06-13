#pragma once

#include <Eigen/Core>
#include "params.h"

class camera_cpu
{
public:
    camera_cpu(float _fx, float _fy, float _cx, float _cy, int _width, int _height)
    {
        float xp = float(MAX_WIDTH) / _width;
        float yp = float(MAX_HEIGHT) / _height;

        float out_fx, out_fy, out_cx, out_cy;
        out_fx = _fx * xp;
        out_fy = _fy * yp;
        out_cx = _cx * xp;
        out_cy = _cy * yp;

        for (int lvl = 0; lvl < MAX_LEVELS; lvl++)
        {
            float scale = std::pow(2.0f, float(lvl));

            width[lvl] = int(MAX_WIDTH / scale);
            height[lvl] = int(MAX_HEIGHT / scale);

            dx[lvl] = 1.0 / width[lvl];
            dy[lvl] = 1.0 / height[lvl];

            Eigen::Matrix3f K = Eigen::Matrix3f::Zero();
            K(0, 0) = out_fx / scale;
            K(1, 1) = out_fy / scale;
            K(2, 2) = 1.0f;
            K(0, 2) = out_cx / scale;
            K(1, 2) = out_cy / scale;

            fx[lvl] = K(0, 0);
            fy[lvl] = K(1, 1);
            cx[lvl] = K(0, 2);
            cy[lvl] = K(1, 2);

            Eigen::Matrix3f KInv = K.inverse();

            fxinv[lvl] = KInv(0, 0);
            fyinv[lvl] = KInv(1, 1);
            cxinv[lvl] = KInv(0, 2);
            cyinv[lvl] = KInv(1, 2);
        }
    };

    float fx[MAX_LEVELS];
    float fy[MAX_LEVELS];
    float cx[MAX_LEVELS];
    float cy[MAX_LEVELS];

    float fxinv[MAX_LEVELS];
    float fyinv[MAX_LEVELS];
    float cxinv[MAX_LEVELS];
    float cyinv[MAX_LEVELS];

    int width[MAX_LEVELS], height[MAX_LEVELS];
    float dx[MAX_LEVELS], dy[MAX_LEVELS];
};
