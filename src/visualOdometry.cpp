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
    frameCPU newFrame(cam.width, cam.height);
    newFrame.set(image); //*keyframeData.pose.inverse();
    newFrame.id = lastFrame.id + 1;
    newFrame.pose = lastMovement * lastFrame.pose;
    tic_toc t;
    t.tic();
    meshOptimizer.optPose(newFrame);
    std::cout << "estimated pose " << t.toc() << std::endl;
    std::cout << newFrame.pose.matrix() << std::endl;
    lastMovement = newFrame.pose * lastFrame.pose.inverse();
    lastFrame = newFrame;

    dataCPU<float> idepth = meshOptimizer.getIdepth(newFrame.pose, 1);
    float percentNoData = idepth.getPercentNoData(1);

    bool updateMap = false;

    if (percentNoData > 0.20)
    {
        meshOptimizer.changeKeyframe(newFrame);
        updateMap = true;

        //std::vector<frameCPU> newFrames;
        //newFrames.push_back(frames[1]);
        //newFrames.push_back(lastFrame);

        //frames = newFrames;
    }
    else
    {
        frames.push_back(newFrame);
        if(frames.size() > 3)
        {
            frames.erase(frames.begin());
        }
        updateMap = true;
        /*
        if (frames.size() < 3)
        {
            frames.push_back(newFrame);
            updateMap = true;
        }
        else
        {
            frames[frames.size()-1] = newFrame;
            updateMap = true;
        }
        */
    }

    if (updateMap)
    {
        t.tic();
        meshOptimizer.optPoseMap(frames);
        std::cout << "optposemap time " << t.toc() << std::endl;
    }

    meshOptimizer.plotDebug(newFrame);
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
    tic_toc t;
    //t.tic();
    //meshOptimizer.optPose(newFrame);
    //std::cout << "estimated pose " << t.toc() << std::endl;
    //std::cout << newFrame.pose.matrix() << std::endl;
    //lastMovement = newFrame.pose * lastFrame.pose.inverse();
    lastFrame = newFrame;

    dataCPU<float> idepth = meshOptimizer.getIdepth(newFrame.pose, 1);
    float percentNoData = idepth.getPercentNoData(1);

    bool updateMap = false;

    if (percentNoData > 0.20)
    {
        meshOptimizer.changeKeyframe(newFrame);
        updateMap = true;

        std::vector<frameCPU> newFrames;
        newFrames.push_back(frames[1]);
        newFrames.push_back(lastFrame);

        frames = newFrames;
    }
    else
    {
        if (frames.size() < 3)
        {
            frames.push_back(newFrame);
            updateMap = true;
        }
        else
        {
            frames[frames.size()-1] = newFrame;
            updateMap = true;
        }
    }

    if (updateMap)
    {
        t.tic();
        meshOptimizer.optMap(frames);
        std::cout << "optmap time " << t.toc() << std::endl;
    }

    meshOptimizer.plotDebug(newFrame);
}
