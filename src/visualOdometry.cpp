#include "visualOdometry.h"
#include "utils/tictoc.h"

visualOdometry::visualOdometry(camera &_cam)
    : lastFrame(_cam.width, _cam.height),
      meshOptimizer(_cam)
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

    frameCPU newFrame(cam.width, cam.height);
    newFrame.set(image); //*keyframeData.pose.inverse();
    newFrame.id = lastFrame.id + 1;
    newFrame.pose = lastMovement * lastFrame.pose;
    meshOptimizer.optPose(newFrame);

    lastMovement = newFrame.pose * lastFrame.pose.inverse();
    lastFrame = newFrame;
    frames.push_back(newFrame);

    meshOptimizer.optPoseMap(frames);

    if (frames.size() > 3)
    {
        meshOptimizer.changeKeyframe(frames[0]);
        //frames.erase(frames.end());
        frames.erase(frames.begin());
    }

    meshOptimizer.plotDebug(frames[0]);
}

void visualOdometry::localization(dataCPU<float> &image)
{
    frameCPU newFrame(cam.width, cam.height);
    newFrame.set(image);
    newFrame.id = lastFrame.id + 1;
    newFrame.pose = lastMovement * lastFrame.pose;
    tic_toc t;
    t.tic();
    meshOptimizer.optPose(newFrame);

    std::cout << "estimated pose " << t.toc() << std::endl;
    std::cout << newFrame.pose.matrix() << std::endl;

    if(newFrame.id % 1 == 0)
        meshOptimizer.plotDebug(newFrame);

    lastMovement = newFrame.pose * lastFrame.pose.inverse();
    lastFrame = newFrame;
}

void visualOdometry::mapping(dataCPU<float> &image, Sophus::SE3f pose)
{
    frameCPU newFrame(cam.width, cam.height);
    newFrame.set(image); //*keyframeData.pose.inverse();
    newFrame.id = lastFrame.id + 1;
    newFrame.pose = pose;
    //meshOptimizer.optPose(newFrame);

    lastMovement = newFrame.pose * lastFrame.pose.inverse();
    lastFrame = newFrame;
    frames.push_back(newFrame);

    //meshOptimizer.optMap(frames);

    if (frames.size() > 3)
    {
        //meshOptimizer.changeKeyframe(frames[0]);
        //frames.erase(frames.end());
        frames.erase(frames.begin());
    }

    meshOptimizer.plotDebug(frames[0]);
}
