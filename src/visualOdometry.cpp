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
    meshOptimizer.optPose(lastFrame, newFrame);

    // float imagePercentNoData = meshOptimizer.getImage(lastFrame.pose, 1).getPercentNoData(1);
    // meshOptimizer.changeKeyframe(lastFrame);
    if (frames.size() > 0)
    {
        meshOptimizer.optPoseMap(newFrame, frames);
        meshOptimizer.plotDebug(newFrame, frames[0]);
    }
    lastMovement = newFrame.pose * lastFrame.pose.inverse();
    lastFrame = newFrame;
    frames.push_back(newFrame);
    if (frames.size() > 3)
        frames.erase(frames.begin());
}

void visualOdometry::localization(dataCPU<float> &image)
{
    frameCPU newFrame(cam.width, cam.height);
    newFrame.set(image);
    newFrame.id = lastFrame.id + 1;
    newFrame.pose = lastMovement * lastFrame.pose;
    meshOptimizer.optPose(lastFrame, newFrame);

    std::cout << "estimated pose" << std::endl;
    std::cout << newFrame.pose.matrix() << std::endl;

    meshOptimizer.plotDebug(lastFrame, newFrame);

    lastMovement = newFrame.pose * lastFrame.pose.inverse();
    lastFrame = newFrame;
}

void visualOdometry::mapping(dataCPU<float> &image, Sophus::SE3f pose)
{
    tic_toc t;

    frameCPU newFrame(cam.width, cam.height);
    newFrame.set(image, pose);
    newFrame.id = lastFrame.id + 1;

    if (frames.size() > 0)
    {
        /*
        // use new frame as keyframe
        std::vector<frameCPU> framesOpt;
        framesOpt.push_back(frames[0]);

        meshOptimizer.optMap(newFrame, framesOpt);
        meshOptimizer.plotDebug(frames[0]);

        // use last frame as keyframe
        framesOpt.clear();
        framesOpt.push_back(newFrame);

        meshOptimizer.optMap(frames[0], framesOpt);
        meshOptimizer.plotDebug(newFrame);
        */

        meshOptimizer.optMap(newFrame, frames);
        meshOptimizer.plotDebug(newFrame, frames[0]);
    }

    lastFrame = newFrame;
    frames.push_back(newFrame);
    if (frames.size() > 3)
        frames.erase(frames.begin());
    /*
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
    */
}
