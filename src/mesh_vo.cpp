#include "mesh_vo.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/OrderingMethods>
#include <Eigen/IterativeLinearSolvers>

#include <map>

meshVO::meshVO(float _fx, float _fy, float _cx, float _cy, int _width, int _height)
    : scene(_fx, _fy, _cx, _cy, _width, _height)
{
    // H_depth = Eigen::SparseMatrix<float>(VERTEX_HEIGH*VERTEX_WIDTH, VERTEX_HEIGH*VERTEX_WIDTH);
    // J_depth = Eigen::VectorXf::Zero(VERTEX_HEIGH*VERTEX_WIDTH);
    // inc_depth = Eigen::VectorXf::Zero(VERTEX_HEIGH*VERTEX_WIDTH);
    // count_depth = Eigen::VectorXi::Zero(VERTEX_HEIGH*VERTEX_WIDTH);
}

void meshVO::initScene(cv::Mat image, Sophus::SE3f pose)
{
    frameCPU keyframe;
    keyframe.set(image, pose);
    dataCPU<float> idepth = getRandomIdepth(0);
    scene.init(keyframe, idepth);
}

void meshVO::initScene(cv::Mat image, cv::Mat idepth, Sophus::SE3f pose)
{
    frameCPU keyframe;
    keyframe.set(image, pose);
    dataCPU<float> keyframeIdepth(-1.0);
    keyframeIdepth.set(idepth);
    scene.init(keyframe, keyframeIdepth);
}

void meshVO::visualOdometry(cv::Mat image)
{
    tic_toc t;

    lastFrame.set(image); //*keyframeData.pose.inverse();
    scene.optPose(lastFrame);

    std::cout << "estimated pose" << std::endl;
    std::cout << lastFrame.pose.matrix() << std::endl;
    
    frames.clear();
    frames.push_back(lastFrame);

    t.tic();
    // optMapVertex();
    // optMapJoint();
    scene.optPoseMap(frames);
    std::cout << "update pose map time " << t.toc() << std::endl;

    dataCPU<float> idepth = scene.computeFrameIdepth(lastFrame, 1);
    dataCPU<float> error = scene.computeErrorImage(lastFrame, 1);

    lastFrame.image.show("lastFrame image", 1);
    //lastFrame.dx.show("lastFrame dx", 1);
    //lastFrame.dy.show("lastFrame dy", 1);
    error.show("lastFrame error", 1);
    idepth.show("lastFrame idepth", 1);
}

void meshVO::localization(cv::Mat image)
{
    lastFrame.set(image);

    // cpu.computeFrameIdepth(lastframe, cam, scene, 1);
    // cpu.computeError(lastframe, keyframe, cam, scene, 1);
    // cpu.computeHGPose(lastframe, keyframe, cam, scene, 1);

    // frameData.pose = _globalPose*keyframeData.pose.inverse();
    scene.optPose(lastFrame); //*Sophus::SE3f::exp(inc_pose).inverse());

    std::cout << "estimated pose" << std::endl;
    std::cout << lastFrame.pose.matrix() << std::endl;

    dataCPU<float> idepth = scene.computeFrameIdepth(lastFrame, 1);
    dataCPU<float> error = scene.computeErrorImage(lastFrame, 1);

    lastFrame.image.show("lastFrame image", 1);
    //lastFrame.dx.show("lastFrame dx", 1);
    //lastFrame.dy.show("lastFrame dy", 1);
    error.show("lastFrame error", 1);
    idepth.show("lastFrame idepth", 1);
}

void meshVO::mapping(cv::Mat image, Sophus::SE3f pose)
{
    tic_toc t;

    lastFrame.set(image, pose); //*keyframeData.pose.inverse();

    frames.clear();
    frames.push_back(lastFrame);

    t.tic();
    // optMapVertex();
    // optMapJoint();
    scene.optMap(frames);
    std::cout << "update map time " << t.toc() << std::endl;

    dataCPU<float> idepth = scene.computeFrameIdepth(lastFrame, 1);
    dataCPU<float> error = scene.computeErrorImage(lastFrame, 1);

    lastFrame.image.show("lastFrame image", 1);
    //lastFrame.dx.show("lastFrame dx", 1);
    //lastFrame.dy.show("lastFrame dy", 1);
    error.show("lastFrame error", 1);
    idepth.show("lastFrame idepth", 1);
}
