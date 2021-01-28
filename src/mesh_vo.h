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

#define MAX_LEVELS 10 //para una imagen de entrada de 512x512
#define MAX_FRAMES 3

class mesh_vo
{
public:
    mesh_vo(float _fx, float _fy, float _cx, float _cy, int _width, int _height);

    void setKeyframeRandomIdepth(cv::Mat _keyFrame);
    void setKeyframeWithIdepth(cv::Mat _keyFrame, cv::Mat _idepth);
    void setKeyframe(cv::Mat _keyframe, Sophus::SE3f _keyframePose);

    Sophus::SE3f calcPose(cv::Mat _frame, Sophus::SE3f initialGuessPose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));

    void addFrameToStack(cv::Mat _frame, Sophus::SE3f _framePose);

    void updateMap();

    void visual_odometry(cv::Mat _frame);

    Sophus::SE3f trackedPose;
    float superpositionPercentaje;

private:

    GLFWwindow* frameWindow;

    int vwidth, vheight;
    std::vector<float> scene_vertices;
    std::vector<float> scene_vertices_updated;
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
    unsigned int frameDerivativeTexture;

    unsigned int frameTextureStack[MAX_FRAMES];
    unsigned int frameDerivativeTextureStack[MAX_FRAMES];
    Sophus::SE3f framePoseStack[MAX_FRAMES];

    GLfloat* frame_cpu_data;
    GLfloat* frameDer_cpu_data;


    int lastFrameAdded;

    unsigned int reduceFloatTexture;
    unsigned int reduceVec4Texture;

    unsigned int idepthTexture;
    GLfloat* idepth_cpu_data;

    unsigned int errorTexture;
    GLfloat* error_cpu_data;

    unsigned int residualTexture;
    GLfloat* residual_cpu_data;

    unsigned int traTexture;
    GLfloat* tra_cpu_data;

    unsigned int rotTexture;
    GLfloat* rot_cpu_data;

    unsigned int JposeTexture1;
    GLfloat* j_pose_data1;
    unsigned int JposeTexture2;
    GLfloat* j_pose_data2;
    unsigned int JposeTexture3;
    GLfloat* j_pose_data3;
    unsigned int JposeTexture4;
    GLfloat* j_pose_data4;
    unsigned int JposeTexture5;
    GLfloat* j_pose_data5;
    unsigned int JposeTexture6;
    GLfloat* j_pose_data6;
    unsigned int JposeTexture7;
    GLfloat* j_pose_data7;


    unsigned int vertexID_Texture;
    GLfloat* vertexID_cpu_data;

    unsigned int primitiveID_Texture;
    GLfloat* primitiveID_cpu_data;

    unsigned int d_I_d_p0_Texture;
    GLfloat* d_I_d_p0_cpu_data;

    unsigned int d_I_d_p1_Texture;
    GLfloat* d_I_d_p1_cpu_data;

    unsigned int d_I_d_p2_Texture;
    GLfloat* d_I_d_p2_cpu_data;

    Shader frameDerivativeShader;
    Shader errorShader;
    Shader reduceFloatShader;
    Shader reduceVec4Shader;
    Shader calcJShader;
    Shader calcHJShader;
    Shader calcHJShader2;
    Shader calcHJMapShader;
    Shader showTextureShader;
    Shader debugShader;
    Shader idepthShader;

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

    glm::mat4 projMat[MAX_LEVELS];
    glm::mat4 opencv2opengl;

    Eigen::Matrix<float, 6, 1> acc_J_pose;
    Eigen::Matrix<float, 6, 6> acc_H_pose;

    Eigen::MatrixXf acc_H_map;
    Eigen::VectorXf acc_J_map;
    Eigen::VectorXf inc;

    void calcIdepth(Sophus::SE3f framePose, int lvl);
    float calcSuperposition(Sophus::SE3f framePose, int lvl);

    float calcError(unsigned int keyframe, unsigned int frame, Sophus::SE3f framePose, int lvl);
    float calcError_CPU(unsigned int frame, Sophus::SE3f framePose, int lvl);


    void calcHJPose(unsigned int keyframe, unsigned int keyframeDer, unsigned int frame, unsigned int frameDer, Sophus::SE3f framePose, int lvl);
    void calcHJPose2(unsigned int keyframe, unsigned int keyframeDer, unsigned int frame, unsigned int frameDer, Sophus::SE3f framePose, int lvl);
    void calcHJPose_CPU(unsigned int frame, unsigned int frameDer, Sophus::SE3f framePose, int lvl);

    void calcHJMap(unsigned int keyframe, unsigned int keyframeDer, unsigned int frame, unsigned int frameDer, Sophus::SE3f framePose, int lvl);

    void frameDerivative(unsigned int frame, unsigned int frameDerivative);

    void showTexture(unsigned int texture, int lvl);

    void showDebug(unsigned int frame, Sophus::SE3f framePose, int lvl);

    void reduceFloat(unsigned int texture, int src_lvl, int dst_lvl);
    void reduceVec4(unsigned int texture, int src_lvl, int dst_lvl);

    //for profiling
    float calcPoseTime;
};
