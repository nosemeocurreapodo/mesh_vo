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
#include <learnopengl/compute.h>
//#include <learnopengl/feedback_shader.h>
//#include <learnopengl/camera.h>

#include "data.h"
#include "frame.h"
#include "params.h"

class mesh_vo
{
public:
    mesh_vo(float _fx, float _fy, float _cx, float _cy, int _width, int _height);

    void initWithRandomIdepth(cv::Mat _keyFrame, Sophus::SE3f _pose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));
    void initWithIdepth(cv::Mat _keyFrame, cv::Mat _idepth, Sophus::SE3f _pose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));

    void visual_odometry(cv::Mat _frame);
    void mapping(cv::Mat _frame, Sophus::SE3f _globalPose);
    void localization(cv::Mat _frame);

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

    std::vector<float> scene_vertices;
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

    //data
    data vertexIdData;
    data debugData;
    data view3DData;

    //shaders
    Shader idepthShader;
    Shader occupancyShader;
    Shader vertexViewCountShader;

    Shader errorShader;
    Shader errorVertexShader;
    Shader frameDerivativeShader;
    Shader jacobianPoseShader;
    Shader jacobianPoseMapShader;
    Shader jacobianMapShader;

    Shader copyShader;
    Shader showTextureShader;
    Shader view3DShader;

    Compute errorReduceShader;
    Compute reduceErrorShader;
    Compute reduceHJPoseShader;

    Eigen::SparseMatrix<float> H_depth;
    Eigen::VectorXf J_depth;
    Eigen::VectorXf inc_depth;
    Eigen::VectorXi count_depth;

    Eigen::MatrixXf H_joint;
    Eigen::VectorXf J_joint;
    Eigen::VectorXf inc_joint;
    Eigen::VectorXi count_joint;

    void changeKeyframe(frame &newkeyFrame, int lvl, float min_occupancy);
    void addFrameToStack(frame &_frame);
    void setTriangles();

    void optPose(frame &_frame);
    void optPose2(frame &_frame);
    void optMapJoint();
    //void optMapVertex();
    void optPoseMapJoint();

    void HJPoseCPU(frame &_frame, int lvl);
    void HJPoseGPU(frame &_frame, int lvl);
    void jacobianPoseTextureGPU(frame &_frame, int lvl);
    void reduceHJPoseGPU(frame &_frame, int lvl);

    void HJPoseMapStackGPU(int lvl);
    void HJPoseMapGPU(frame &_frame, int lvl);
    void jacobianPoseMapTextureGPU(frame &_frame, int lvl);
    void reduceHJPoseMapGPU(int frameIndex, int lvl);

    void HJMapStackGPU(int lvl);
    void HJMapGPU(frame &_frame, int lvl);
    void jacobianMapTextureGPU(frame &_frame, int lvl);
    void reduceHJMapGPU(frame &_frame, int lvl);

    float errorMesh();
    void HJMesh();

    float errorCPU(frame &_frame, int lvl);
    float errorStackCPU(int lvl);

    float errorGPU(frame &_frame, int lvl);
    float errorStackGPU(int lvl);
    void errorTextureGPU(frame &_frame, int lvl);
    float reduceErrorGPU(frame _frame, int lvl, bool useCountData = false);
    float reduceErrorComputeGPU(frame _frame, int lvl);
    void errorVertexGPU(frame &_frame, int lvl);
    float reduceErrorVertexGPU(frame _frame, int lvl);

    float goodVertexViewPercentage(frame &_frame);
    void vertexViewCountGPU(frame &_frame, int lvl);
    void vertexViewCountTextureGPU(frame &_frame, int lvl);
    void reduceVertexViewCountGPU(frame &_frame, int lvl);

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

    void view3DTexture(Sophus::SE3f pose, int lvl);

};
