#include "visualOdometry.h"

// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>

// #include <Eigen/SparseCholesky>
// #include <Eigen/SparseLU>
// #include <Eigen/SparseQR>
// #include <Eigen/OrderingMethods>
// #include <Eigen/IterativeLinearSolvers>

#include "utils/tictoc.h"

visualOdometry::visualOdometry(camera &_cam)
    : meshOptimizer(_cam),
    lastFrame(_cam.width, _cam.height)
{
    cam = _cam;
}

void visualOdometry::initScene(cv::Mat image, Sophus::SE3f pose)
{
    lastFrame.set(image, pose);
    dataCPU<float> idepth = getRandomIdepth();
    meshOptimizer.init(lastFrame, idepth);
}

void visualOdometry::initScene(cv::Mat image, cv::Mat idepth, Sophus::SE3f pose)
{
    lastFrame.set(image, pose);
    dataCPU<float> keyframeIdepth(cam.width, cam.height, -1.0);
    keyframeIdepth.set(idepth);
    meshOptimizer.init(lastFrame, keyframeIdepth);
}

void visualOdometry::locAndMap(cv::Mat image)
{
    tic_toc t;

    dataCPU<float> idepth(cam.width, cam.height, -1.0);
    dataCPU<float> error(cam.width, cam.height, -1.0);
    dataCPU<float> sceneImage(cam.width, cam.height, -1.0);
    dataCPU<float> debug(cam.width, cam.height, -1.0);

    lastFrame.set(image); //*keyframeData.pose.inverse();
    meshOptimizer.optPose(lastFrame);

    std::cout << "estimated pose" << std::endl;
    std::cout << lastFrame.pose.matrix() << std::endl;

    frames.clear();
    frames.push_back(lastFrame);

    t.tic();

    meshOptimizer.optPoseMap(frames);
    std::cout << "update pose map time " << t.toc() << std::endl;

    meshOptimizer.renderIdepth(lastFrame.pose, idepth, 1);
    meshOptimizer.renderError(lastFrame, error, 1);
    meshOptimizer.renderImage(lastFrame.pose, sceneImage, 1);
    meshOptimizer.renderDebug(lastFrame.pose, debug, 0);

    debug.show("lastFrame debug", 0);
    lastFrame.image.show("lastFrame image", 1);
    error.show("lastFrame error", 1);
    idepth.show("lastFrame idepth", 1);
    sceneImage.show("lastFrame scene", 1);

    float scenePercentNoData = sceneImage.getPercentNoData(1);

    if (scenePercentNoData > 0.25)
    {
        meshOptimizer.changeKeyframe(lastFrame);
    }
}

void visualOdometry::localization(cv::Mat image)
{
    lastFrame.set(image);

    // cpu.computeFrameIdepth(lastframe, cam, scene, 1);
    // cpu.computeError(lastframe, keyframe, cam, scene, 1);
    // cpu.computeHGPose(lastframe, keyframe, cam, scene, 1);

    // frameData.pose = _globalPose*keyframeData.pose.inverse();
    meshOptimizer.optPose(lastFrame); //*Sophus::SE3f::exp(inc_pose).inverse());

    std::cout << "estimated pose" << std::endl;
    std::cout << lastFrame.pose.matrix() << std::endl;

    dataCPU<float> idepth(cam.width, cam.height, -1.0);
    dataCPU<float> error(cam.width, cam.height, -1.0);
    dataCPU<float> sceneImage(cam.width, cam.height, -1.0);
    dataCPU<float> debug(cam.width, cam.height, -1.0);

    meshOptimizer.renderIdepth(lastFrame.pose, idepth, 1);
    meshOptimizer.renderError(lastFrame, error, 1);
    meshOptimizer.renderImage(lastFrame.pose, sceneImage, 1);
    meshOptimizer.renderDebug(lastFrame.pose, debug, 1);

    lastFrame.image.show("lastFrame image", 1);
    // lastFrame.dx.show("lastFrame dx", 1);
    // lastFrame.dy.show("lastFrame dy", 1);
    error.show("lastFrame error", 1);
    idepth.show("lastFrame idepth", 1);
    sceneImage.show("lastFrame scene", 1);
    debug.show("lastFrame debug", 1);
}

void visualOdometry::mapping(cv::Mat image, Sophus::SE3f pose)
{
    tic_toc t;

    dataCPU<float> idepth(cam.width, cam.height, -1.0);
    dataCPU<float> error(cam.width, cam.height, -1.0);
    dataCPU<float> sceneImage(cam.width, cam.height, -1.0);
    dataCPU<float> debug(cam.width, cam.height, -1.0);

    // lastFrame.set(image, pose);
    lastFrame.set(image, pose);

    frames.clear();
    frames.push_back(lastFrame);

    t.tic();
    meshOptimizer.optMap(frames);
    std::cout << "update map time " << t.toc() << std::endl;

    meshOptimizer.renderIdepth(lastFrame.pose, idepth, 1);
    meshOptimizer.renderError(lastFrame, error, 1);
    meshOptimizer.renderImage(lastFrame.pose, sceneImage, 1);
    meshOptimizer.renderDebug(lastFrame.pose, debug, 0);

    debug.show("lastFrame debug", 0);
    lastFrame.image.show("lastFrame image", 1);
    error.show("lastFrame error", 1);
    idepth.show("lastFrame idepth", 1);
    sceneImage.show("lastFrame scene", 1);

    float scenePercentNoData = sceneImage.getPercentNoData(1);

    if (scenePercentNoData > 0.25)
    {
        meshOptimizer.changeKeyframe(lastFrame);
    }
}
