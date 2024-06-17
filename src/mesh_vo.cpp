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
    :scene(_fx, _fy, _cx, _cy, _width, _height)
{
    // H_depth = Eigen::SparseMatrix<float>(VERTEX_HEIGH*VERTEX_WIDTH, VERTEX_HEIGH*VERTEX_WIDTH);
    // J_depth = Eigen::VectorXf::Zero(VERTEX_HEIGH*VERTEX_WIDTH);
    // inc_depth = Eigen::VectorXf::Zero(VERTEX_HEIGH*VERTEX_WIDTH);
    // count_depth = Eigen::VectorXi::Zero(VERTEX_HEIGH*VERTEX_WIDTH);
}

void meshVO::initScene(cv::Mat frame, Sophus::SE3f pose = Sophus::SE3f())
{
    keyframe.set(frame, pose);
    dataCPU<float> idepth = getRandomIdepth(0);
    scene.init(keyframe, idepth);
}

void meshVO::initScene(cv::Mat frame, cv::Mat idepth, Sophus::SE3f pose = Sophus::SE3f())
{
    keyframe.set(frame, pose);
    dataCPU<float> keyframeIdepth(-1.0);
    keyframeIdepth.set(idepth);
    scene.init(keyframe, keyframeIdepth);
}

void meshVO::visualOdometry(cv::Mat frame)
{
    lastframe.set(frame);

    if (!keyframe.init)
    {
        lastframe.copyTo(keyframe);
        return;
    }

    tic_toc t;
    t.tic();
    // frameData.pose = _globalPose*keyframeData.pose.inverse();
    optPose(lastframe); //*Sophus::SE3f::exp(inc_pose).inverse());

    std::cout << "estimated pose " << std::endl;
    std::cout << lastframe.pose.matrix() << std::endl;
    std::cout << "clacPose time " << t.toc() << std::endl;

    {
        // optPoseMapJoint();
    }
}

void meshVO::localization(cv::Mat frame)
{
    lastframe.set(frame);

    // cpu.computeFrameIdepth(lastframe, cam, scene, 1);
    // cpu.computeError(lastframe, keyframe, cam, scene, 1);
    // cpu.computeHGPose(lastframe, keyframe, cam, scene, 1);

    // frameData.pose = _globalPose*keyframeData.pose.inverse();
    optPose(lastframe); //*Sophus::SE3f::exp(inc_pose).inverse());

    scene.computeFrameIdepth(lastframe, 1);

    lastframe.image.show("lastframe image", 1);
    // lastframe.der.show("lastframe der", 1);
    lastframe.error.show("lastframe error", 1);
    lastframe.idepth.show("lastframe idepth", 1);
}

void meshVO::mapping(cv::Mat frame, Sophus::SE3f pose)
{
    tic_toc t;

    lastframe.set(frame, pose); //*keyframeData.pose.inverse();

    t.tic();
    // optMapVertex();
    // optMapJoint();
    optMap();
    std::cout << "update map time " << t.toc() << std::endl;

    lastframe.image.show("lastframe image", 1);
    // lastframe.der.show("lastframe der", 1);
    lastframe.error.show("lastframe error", 1);
    lastframe.idepth.show("lastframe idepth", 1);
}
