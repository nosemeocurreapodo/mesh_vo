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
    dataCPU<float> idepth(cam.width, cam.height, -1.0);
    // idepth.setRandom(0);
    idepth.setSmooth(0);
    idepth.generateMipmaps();
    dataCPU<float> invVar(cam.width, cam.height, -1.0);
    invVar.set(1.0 / INITIAL_VAR, 0);
    invVar.generateMipmaps();

    meshOptimizer.initKeyframe(lastFrame, idepth, invVar, 0);
}

void visualOdometry::initScene(dataCPU<float> &image, dataCPU<float> &idepth, Sophus::SE3f pose)
{
    lastFrame.set(image, pose);
    dataCPU<float> invVar(cam.width, cam.height, -1.0);
    invVar.set(1.0 / INITIAL_VAR, 0);
    meshOptimizer.initKeyframe(lastFrame, idepth, invVar, 0);
}

void visualOdometry::locAndMap(dataCPU<float> &image)
{
    tic_toc t;

    lastFrame.set(image); //*keyframeData.pose.inverse();
    lastFrame.id += 1;
    Sophus::SE3f iniPose = lastFrame.pose;
    lastFrame.pose = lastMovement * lastFrame.pose;
    meshOptimizer.optPose(lastFrame);
    lastMovement = lastFrame.pose * iniPose.inverse();

    // float imagePercentNoData = meshOptimizer.getImage(lastFrame.pose, 1).getPercentNoData(1);

    meshOptimizer.changeKeyframe(lastFrame);
    if(frames.size() > 0)
    {
        meshOptimizer.optPoseMap(frames);
        meshOptimizer.plotDebug(frames[0]);
    }

    frames.push_back(lastFrame);
    if (frames.size() > 7)
        frames.erase(frames.begin());
}

void visualOdometry::localization(dataCPU<float> &image)
{
    lastFrame.set(image);
    lastFrame.id += 1;
    Sophus::SE3f iniPose = lastFrame.pose;
    lastFrame.pose = lastMovement * lastFrame.pose;
    meshOptimizer.optPose(lastFrame);
    lastMovement = lastFrame.pose * iniPose.inverse();

    std::cout << "estimated pose" << std::endl;
    std::cout << lastFrame.pose.matrix() << std::endl;

    meshOptimizer.plotDebug(lastFrame);
}

void visualOdometry::mapping(dataCPU<float> &image, Sophus::SE3f pose)
{
    tic_toc t;

    // lastFrame.set(image);
    lastFrame.set(image, pose);
    lastFrame.id += 1;

    float imagePercentNoData = meshOptimizer.getImage(lastFrame.pose, 1).getPercentNoData(1);

    if (imagePercentNoData > 0.15)
    {
        meshOptimizer.changeKeyframe(lastFrame);
    }
    else
    {
        frames.push_back(lastFrame);
        if (frames.size() > 3)
            frames.erase(frames.begin());
    }

    meshOptimizer.optMap(frames);

    meshOptimizer.plotDebug(frames[0]);
}
