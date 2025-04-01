#pragma once

#include <glm/glm.hpp>

#include "params.h"

glm::mat4 create_glm_prj_matrix(float fx, float fy, float cx, float cy, float w, float h, float znear, float zfar)
{
    glm::mat4 projmat = glm::mat4(0.0f);

    projmat[0][0] = 2.0f*fx/w;
    projmat[1][1] = 2.0f*fy/h;
    projmat[2][0] = 1.0f - 2.0f*cx/w;
    projmat[2][1] = -1.0f + 2.0f*cy/h;
    projmat[2][2] = -(zfar + znear) / (zfar - znear);
    projmat[2][3] = -1.0f;
    projmat[3][2] = -2.0f * zfar * znear / (zfar - znear);

    return projmat;
}

glm::mat4 eigen2glm_mat4(Eigen::Matrix4f eigen_m)
{
    glm::mat4 glm_m;
    glm_m[0][0] = float(eigen_m(0,0));
    glm_m[1][0] = float(eigen_m(0,1));
    glm_m[2][0] = float(eigen_m(0,2));
    glm_m[3][0] = float(eigen_m(0,3));
    glm_m[0][1] = float(eigen_m(1,0));
    glm_m[1][1] = float(eigen_m(1,1));
    glm_m[2][1] = float(eigen_m(1,2));
    glm_m[3][1] = float(eigen_m(1,3));
    glm_m[0][2] = float(eigen_m(2,0));
    glm_m[1][2] = float(eigen_m(2,1));
    glm_m[2][2] = float(eigen_m(2,2));
    glm_m[3][2] = float(eigen_m(2,3));
    glm_m[0][3] = float(eigen_m(3,0));
    glm_m[1][3] = float(eigen_m(3,1));
    glm_m[2][3] = float(eigen_m(3,2));
    glm_m[3][3] = float(eigen_m(3,3));

    return glm_m;
}

glm::mat3 eigen2glm_mat3(Eigen::Matrix3f eigen_m)
{
    glm::mat3 glm_m;
    glm_m[0][0] = eigen_m(0,0);
    glm_m[1][0] = eigen_m(0,1);
    glm_m[2][0] = eigen_m(0,2);
    glm_m[0][1] = eigen_m(1,0);
    glm_m[1][1] = eigen_m(1,1);
    glm_m[2][1] = eigen_m(1,2);
    glm_m[0][2] = eigen_m(2,0);
    glm_m[1][2] = eigen_m(2,1);
    glm_m[2][2] = eigen_m(2,2);

    return glm_m;
}


class camera
{
public:

    camera(float _fx, float _fy, float _cx, float _cy, int _width, int _height);

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

camera::camera(float _fx, float _fy, float _cx, float _cy, int _width, int _height)
{
    float xp = float(MAX_WIDTH)/_width;
    float yp = float(MAX_HEIGHT)/_height;

    float out_fx, out_fy, out_cx, out_cy;
    out_fx = _fx*xp; out_fy = _fy*yp; out_cx = _cx*xp; out_cy = _cy*yp;

    max_idepth = 100.0;
    min_idepth = 0.001;

    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        float scale = std::pow(2.0f, float(lvl));

        width[lvl] = int(MAX_WIDTH/scale);
        height[lvl] = int(MAX_HEIGHT/scale);

        dx[lvl] = 1.0/width[lvl];
        dy[lvl] = 1.0/height[lvl];

        Eigen::Matrix3f K = Eigen::Matrix3f::Zero();
        K(0,0) = out_fx/scale;
        K(1,1) = out_fy/scale;
        K(2,2) = 1.0f;
        K(0,2) = out_cx/scale;
        K(1,2) = out_cy/scale;

        fx[lvl] = K(0,0);
        fy[lvl] = K(1,1);
        cx[lvl] = K(0,2);
        cy[lvl] = K(1,2);

        Eigen::Matrix3f KInv = K.inverse();

        fxinv[lvl] = KInv(0,0);
        fyinv[lvl] = KInv(1,1);
        cxinv[lvl] = KInv(0,2);
        cyinv[lvl] = KInv(1,2);
    }
}

//    float max_idepth;
//    float min_idepth;

//    glm::mat4 projMat[MAX_LEVELS];
//    glm::mat4 opencv2opengl;
