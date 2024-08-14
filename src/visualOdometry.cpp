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

    // float info = meshOptimizer.checkInfo(newFrame);
    // std::cout << "info: " << info << std::endl;

    dataCPU<float> idepth = meshOptimizer.getIdepth(newFrame.pose, 1);
    float percentNoData = idepth.getPercentNoData(1);

    bool updateMap = false;

    if (percentNoData > 0.20)
    {
        meshOptimizer.changeKeyframe(newFrame);
        updateMap = true;

        for (int i = 0; i < framesInfo.size(); i++)
            framesInfo[i] = 10000.0;
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
            if (newFrame.id % 2)
            {
                frames.erase(frames.begin());
                frames.push_back(newFrame);
                updateMap = true;
            }
        }
        /*
        if (frames.size() < 3)
        {
            frames.push_back(newFrame);
            framesInfo.push_back(info);
            updateMap = true;
            // frames.erase(frames.begin());
        }
        else
        {
            float maxInfo = 0.0;
            int maxI = -1;
            for (int i = 0; i < framesInfo.size(); i++)
            {
                if (framesInfo[i] > maxInfo)
                {
                    maxInfo = framesInfo[i];
                    maxI = i;
                }
            }
            if (info < maxInfo)
            {
                frames[maxI] = newFrame;
                framesInfo[maxI] = info;
                updateMap = true;
            }
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
    // meshOptimizer.optPose(newFrame);

    lastMovement = newFrame.pose * lastFrame.pose.inverse();
    lastFrame = newFrame;
    frames.push_back(newFrame);

    tic_toc t;
    t.tic();
    meshOptimizer.optMap(frames);
    std::cout << "opmap time " << t.toc() << std::endl;

    if (frames.size() > 3)
    {
        meshOptimizer.changeKeyframe(frames[0]);
        frames.erase(frames.begin());
    }

    // meshOptimizer.plotDebug(frames[0]);
}
