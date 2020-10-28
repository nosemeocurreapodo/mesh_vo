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

#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/feedback_shader.h>
#include <learnopengl/camera.h>

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
    unsigned int keyframeDerivativeTexture;
    unsigned int currentframeTexture;
    unsigned int currentframeDerivativeTexture;

    Shader frameDerivativeShader;



    Eigen::Matrix3f K[MAX_LEVELS];
    Eigen::Matrix3f KInv[MAX_LEVELS];

    int width[MAX_LEVELS], height[MAX_LEVELS];

    Eigen::Matrix<float, 6, 1> acc_J_pose;
    Eigen::Matrix<float, 6, 6> acc_H_pose;

    float calcResidual(unsigned int frame, Sophus::SE3f framePose, int lvl);
    //void calcHJ(cv::Mat frame, cv::Mat frameDer, Sophus::SE3f framePose, int lvl);
    //void calcHJ_2(cv::Mat frame, cv::Mat frameDer, Sophus::SE3f framePose, int lvl);
    //void calcHJ_3(cv::Mat frame, cv::Mat frameDer, Sophus::SE3f framePose, int lvl);

    void frameDerivative(unsigned int frame, unsigned int frameDerivative);
};
