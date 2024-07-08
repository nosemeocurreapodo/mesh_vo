#include "visualOdometry.h"
#include "utils/tictoc.h"

visualOdometry::visualOdometry(camera &_cam)
    : meshOptimizer(_cam),
      lastFrame(_cam.width, _cam.height)
{
    cam = _cam;
}

void visualOdometry::initScene(dataCPU<float> &image, Sophus::SE3f pose)
{
    lastFrame.set(image, pose);
    dataCPU<float> idepth = getRandomIdepth();
    dataCPU<float> invVar(cam.width, cam.height, -1.0);
    invVar.set(10.0 * 10.0, 0);
    invVar.generateMipmaps();

    meshOptimizer.initKeyframe(lastFrame, idepth, invVar);
}

void visualOdometry::initScene(dataCPU<float> &image, dataCPU<float> &idepth, Sophus::SE3f pose)
{
    lastFrame.set(image, pose);
    dataCPU<float> invVar(cam.width, cam.height, -1.0);
    invVar.set(10.0 * 10.0, 0);
    meshOptimizer.initKeyframe(lastFrame, idepth, invVar);
}

void visualOdometry::locAndMap(dataCPU<float> &image)
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

    show(debug, "lastFrame debug", 0);
    show(lastFrame.image, "lastFrame image", 1);
    show(error, "lastFrame error", 1);
    show(idepth, "lastFrame idepth", 1);
    show(sceneImage, "lastFrame scene", 1);

    float scenePercentNoData = sceneImage.getPercentNoData(1);

    if (scenePercentNoData > 0.10)
    {
        meshOptimizer.changeKeyframe(lastFrame);
    }
}

void visualOdometry::localization(dataCPU<float> &image)
{
    lastFrame.set(image);
    Sophus::SE3f iniPose = lastFrame.pose;
    lastFrame.pose = lastMovement * lastFrame.pose;
    meshOptimizer.optPose(lastFrame);
    lastMovement = lastFrame.pose * iniPose.inverse();

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

    show(lastFrame.image, "lastFrame image", 1);
    // lastFrame.dx.show("lastFrame dx", 1);
    // lastFrame.dy.show("lastFrame dy", 1);
    show(error, "lastFrame error", 1);
    show(idepth, "lastFrame idepth", 1);
    show(sceneImage, "lastFrame scene", 1);
    show(debug, "lastFrame debug", 1);
}

void visualOdometry::mapping(dataCPU<float> &image, Sophus::SE3f pose)
{
    tic_toc t;

    dataCPU<float> idepth(cam.width, cam.height, -1.0);
    dataCPU<float> error(cam.width, cam.height, -1.0);
    dataCPU<float> sceneImage(cam.width, cam.height, -1.0);
    dataCPU<float> debug(cam.width, cam.height, -1.0);
    dataCPU<float> idepthVar(cam.width, cam.height, -1.0);

    // lastFrame.set(image);
    lastFrame.set(image, pose);

    //frames.clear();
    if (frames.size() >= 1)
        frames.erase(frames.begin());

    frames.push_back(lastFrame);

    t.tic();
    meshOptimizer.optMap(frames);
    std::cout << "update map time " << t.toc() << std::endl;

    meshOptimizer.renderImage(lastFrame.pose, sceneImage, 1);

    float scenePercentNoData = sceneImage.getPercentNoData(1);

    if (scenePercentNoData > 0.1)
    {
        meshOptimizer.changeKeyframe(lastFrame);
    }

    meshOptimizer.renderIdepth(lastFrame.pose, idepth, 1);
    meshOptimizer.renderError(lastFrame, error, 1);
    meshOptimizer.renderDebug(lastFrame.pose, debug, 0);
    meshOptimizer.renderInvVar(lastFrame.pose, idepthVar, 1);
    meshOptimizer.renderImage(lastFrame.pose, sceneImage, 1);

    // show(debug, "lastFrame debug", 0);
    show(lastFrame.image, "lastFrame image", 1);
    show(meshOptimizer.keyframe.image, "keyframe image", 1);
    show(error, "lastFrame error", 1);
    show(idepth, "lastFrame idepth", 1);
    show(idepthVar, "lastFrame invVar", 1);
    show(sceneImage, "lastFrame scene", 1);
}
