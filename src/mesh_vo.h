#pragma once

#include <opencv2/core.hpp>

#include <iostream>
#include <fstream>

//#include "Common/se3.h"
#include <Eigen/Core>
#include "sophus/se3.hpp"

#include "Utils/tictoc.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

//#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
//#include <learnopengl/feedback_shader.h>
//#include <learnopengl/camera.h>

#define MAX_LEVELS 6

class mesh_vo
{
public:
    mesh_vo(float fx, float fy, float cx, float cy, int _width, int _height);

    void setKeyframeRandomIdepth(cv::Mat _keyFrame);
    void setKeyframeWithIdepth(cv::Mat _keyFrame, cv::Mat _idepth);

    Sophus::SE3f updatePose(cv::Mat frame);

    Sophus::SE3f framePose;

private:

    GLFWwindow* frameWindow;

    int vwidth, vheight;
    std::vector<float> scene_vertices;
    std::vector<unsigned int> scene_indices;

    unsigned int scene_VBO, scene_VAO, scene_EBO;
    unsigned int frame_VBO, frame_VAO;

    unsigned int framebuffer;
    unsigned int rbo;
    unsigned int feedbackrbo;

    unsigned int keyframeTexture;
    GLfloat* keyframe_cpu_data;

    unsigned int keyframeDerivativeTexture;

    unsigned int frameTexture;
    GLfloat* frame_cpu_data;

    unsigned int frameDerivativeTexture;
    GLfloat* frameDer_cpu_data;

    unsigned int idepthTexture;
    GLfloat* idepth_cpu_data;

    unsigned int residualTexture;
    GLfloat* residual_cpu_data;

    unsigned int traTexture;
    GLfloat* tra_cpu_data;

    unsigned int rotTexture;
    GLfloat* rot_cpu_data;

    Shader frameDerivativeShader;
    Shader residualShader;
    Shader calcHJShader;
    Shader showTextureShader;
    Shader debugShader;
    Shader idepthShader;

    Eigen::Matrix3f K[MAX_LEVELS];
    Eigen::Matrix3f KInv[MAX_LEVELS];

    int width[MAX_LEVELS], height[MAX_LEVELS];

    glm::mat4 projMat[MAX_LEVELS];
    glm::mat4 opencv2opengl;

    Eigen::Matrix<float, 6, 1> acc_J_pose;
    Eigen::Matrix<float, 6, 6> acc_H_pose;

    void calcIdepth(Sophus::SE3f framePose, int lvl);

    float calcResidual(unsigned int frame, Sophus::SE3f framePose, int lvl);
    float calcResidual_CPU(unsigned int frame, Sophus::SE3f framePose, int lvl);

    void calcHJ(unsigned int frame, unsigned int frameDer, Sophus::SE3f framePose, int lvl);
    void calcHJ_CPU(unsigned int frame, unsigned int frameDer, Sophus::SE3f framePose, int lvl);

    void frameDerivative(unsigned int frame, unsigned int frameDerivative);

    void showTexture(unsigned int texture, int lvl);

    float reduce_residual_CPU(unsigned int residualTexture, int lvl);

    void showDebug(unsigned int frame, Sophus::SE3f framePose, int lvl);
};
