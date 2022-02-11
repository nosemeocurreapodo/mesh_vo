#pragma once

#include <opencv2/core.hpp>

#include <iostream>
#include <fstream>

//#include "Common/se3.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "sophus/se3.hpp"

#include "Utils/tictoc.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

//#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
//#include <learnopengl/feedback_shader.h>
//#include <learnopengl/camera.h>

#include "data.h"
#include "frame.h"

class mesh_vo
{
public:
    mesh_vo(float _fx, float _fy, float _cx, float _cy, int _width, int _height);

    void initWithRandomIdepth(cv::Mat _keyFrame, Sophus::SE3f _pose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));
    void initWithIdepth(cv::Mat _keyFrame, cv::Mat _idepth, Sophus::SE3f _pose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));

    void visual_odometry(cv::Mat _frame);
    void mapping(cv::Mat _frame, Sophus::SE3f _globalPose);

private:

    //camera data
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

    float max_idepth;
    float min_idepth;

    glm::mat4 projMat[MAX_LEVELS];
    glm::mat4 opencv2opengl;

    //opengl data
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

    //frames
    frame keyframeData;
    frame frameData;
    frame frameDataStack[MAX_FRAMES];

    int lastFrameAdded;
    float occupancy;

    //data
    data reduceFloatData;
    data reduceVec4Data;

    data occuData;
    data errorData;
    data residualData;
    data traData;
    data rotData;

    data Jpose1Data;
    data Jpose2Data;
    data Jpose3Data;
    data Jpose4Data;
    data Jpose5Data;
    data Jpose6Data;
    data Jpose7Data;

    data vertexIdData;
    data primitiveIdData;
    data d_I_d_p0Data;
    data d_I_d_p1Data;
    data d_I_d_p2Data;

    //shaders
    Shader idepthShader;
    Shader occupancyShader;

    Shader errorShader;
    Shader reduceFloatShader;
    Shader reduceVec4Shader;
    Shader calcJShader;
    Shader calcHJShader;
    Shader calcHJShader2;
    Shader calcHJMapShader;
    Shader frameDerivativeShader;

    Shader copyShader;
    Shader showTextureShader;
    Shader debugShader;

    //for invertion
    Eigen::Matrix<float, 6, 1> acc_J_pose;
    Eigen::Matrix<float, 6, 6> acc_H_pose;
    Eigen::Matrix<float, 6, 1> inc_pose;

    //Eigen::MatrixXf acc_H_depth;
    //Eigen::VectorXf acc_J_depth;
    //Eigen::VectorXf inc_depth;

    Eigen::SparseMatrix<float> acc_H_depth;
    Eigen::VectorXf acc_J_depth;
    Eigen::VectorXf inc_depth;
    Eigen::VectorXi acc_count;

    void changeKeyframe(frame newkeyFrame);
    void updateMap();
    void calcPose(frame &_frame, Sophus::SE3f initialGuessPose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));
    void addFrameToStack(frame &_frame);

    void calcHJMapGPU(frame &_frame, int dstlvl, int srclvl);
    void calcHJPoseGPU(frame _frame, int lvl);
    void calcHJPoseGPU2(frame &_frame, int lvl);
    void calcHJPoseCPU(frame &_frame, int lvl);

    float calcErrorGPU(frame &_frame, int lvl);
    float calcErrorCPU(frame &_frame, int lvl);

    //float calcOccupancyGPU(frame _frame, int lvl);
    float calcOccupancyCPU(frame &_frame, int lvl);

    void calcIdepthGPU(frame &_frame, int lvl);

    //for reduce
    void reduceFloat(unsigned int texture, int src_lvl, int dst_lvl);
    void reduceVec4(unsigned int texture, int src_lvl, int dst_lvl);

    void showCPU(data &_data, int lvl);
    void showGPU(data &_data, int lvl);

    void copyGPU(data &_src, data &_dst, int lvl);
    void copyCPU(data &_src, data &_dst, int lvl);

    void calcDerivativeCPU(frame &_frame, int lvl);
    void calcDerivativeGPU(frame &_frame, int lvl);

    //for profiling
    float calcPoseTime;

};
