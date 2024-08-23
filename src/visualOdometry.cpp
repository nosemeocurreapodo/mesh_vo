#include "visualOdometry.h"
#include "utils/tictoc.h"

visualOdometry::visualOdometry(camera &_cam)
    : meshOptimizer(_cam)
{
    cam = _cam;
    lastId = 0;
}

void visualOdometry::initScene(dataCPU<float> &image, Sophus::SE3f pose)
{
    lastFrames.clear();
    keyFrames.clear();
    lastMovement = Sophus::SE3f();
    frameCPU newFrame(cam.width, cam.height);
    newFrame.set(image, pose);
    meshOptimizer.initKeyframe(newFrame, 0);
}

void visualOdometry::initScene(dataCPU<float> &image, dataCPU<float> &idepth, Sophus::SE3f pose)
{
    lastFrames.clear();
    keyFrames.clear();
    lastMovement = Sophus::SE3f();
    frameCPU newFrame(cam.width, cam.height);
    newFrame.set(image, pose);
    meshOptimizer.initKeyframe(newFrame, idepth, 0);
}

void visualOdometry::locAndMap(dataCPU<float> &image)
{
    tic_toc t;
    bool optimize = false;

    frameCPU newFrame(cam.width, cam.height);
    newFrame.set(image); //*keyframeData.pose.inverse();
    newFrame.id = lastId;
    lastId++;
    if (lastFrames.size() == 0)
        newFrame.pose = meshOptimizer.kscene.getPose();
    else
        newFrame.pose = lastMovement * lastFrames[lastFrames.size() - 1].pose;

    t.tic();
    meshOptimizer.optPose(newFrame);
    std::cout << "estimated pose " << t.toc() << std::endl;
    std::cout << newFrame.pose.matrix() << std::endl;

    lastFrames.push_back(newFrame);

    if (lastFrames.size() < NUM_FRAMES)
    {
        optimize = true;
        keyFrames = lastFrames;
    }

    if (lastFrames.size() > NUM_FRAMES)
    {
        lastFrames.erase(lastFrames.begin());
    }

    /*
    if (optimize)
    {
        t.tic();
        meshOptimizer.optMap(frames);
        std::cout << "optmap time " << t.toc() << std::endl;
        //newFrame.pose = frames[frames.size() - 1].pose;
    }
    */

    // dataCPU<float> idepth = meshOptimizer.getIdepth(newFrame.pose, 1);
    // float percentNoData = idepth.getPercentNoData(1);
    float viewPercent = meshOptimizer.getViewPercent(newFrame);

    std::cout << "view percent " << viewPercent << std::endl;

    if (viewPercent < 0.80)
    {
        keyFrames.clear();
        keyFrames = lastFrames;

        int newFrameIndex = int(keyFrames.size() / 2);
        meshOptimizer.changeKeyframe(lastFrames[newFrameIndex]);
        keyFrames.erase(keyFrames.begin() + newFrameIndex);
        optimize = true;
    }

    if (optimize)
    {
        t.tic();
        meshOptimizer.optPoseMap(keyFrames);
        std::cout << "optposemap time " << t.toc() << std::endl;
    }

    if (lastFrames.size() >= 2)
        lastMovement = lastFrames[lastFrames.size() - 1].pose * lastFrames[lastFrames.size() - 2].pose.inverse();

    meshOptimizer.plotDebug(newFrame);
}

void visualOdometry::localization(dataCPU<float> &image)
{
    tic_toc t;

    frameCPU newFrame(cam.width, cam.height);
    newFrame.set(image);
    newFrame.id = lastId;
    lastId++;
    if (lastFrames.size() == 0)
        newFrame.pose = meshOptimizer.kscene.getPose();
    else
        newFrame.pose = lastMovement * lastFrames[lastFrames.size() - 1].pose;

    t.tic();
    meshOptimizer.optPose(newFrame);
    std::cout << "estimated pose " << t.toc() << std::endl;
    std::cout << newFrame.pose.matrix() << std::endl;

    lastFrames.push_back(newFrame);
    if (lastFrames.size() > NUM_FRAMES)
    {
        lastFrames.erase(lastFrames.begin());
    }

    if (lastFrames.size() >= 2)
        lastMovement = lastFrames[lastFrames.size() - 1].pose * lastFrames[lastFrames.size() - 2].pose.inverse();

    meshOptimizer.plotDebug(newFrame);
}

void visualOdometry::mapping(dataCPU<float> &image, Sophus::SE3f pose)
{
    tic_toc t;
    bool optimize = false;

    frameCPU newFrame(cam.width, cam.height);
    newFrame.set(image); //*keyframeData.pose.inverse();
    newFrame.id = lastId;
    lastId++;
    newFrame.pose = pose;

    lastFrames.push_back(newFrame);

    if (lastFrames.size() < NUM_FRAMES)
    {
        optimize = true;
        keyFrames = lastFrames;
    }

    if (lastFrames.size() > NUM_FRAMES)
    {
        lastFrames.erase(lastFrames.begin());
    }

    /*
    if (optimize)
    {
        t.tic();
        meshOptimizer.optMap(frames);
        std::cout << "optmap time " << t.toc() << std::endl;
        //newFrame.pose = frames[frames.size() - 1].pose;
    }
    */

    // dataCPU<float> idepth = meshOptimizer.getIdepth(newFrame.pose, 1);
    // float percentNoData = idepth.getPercentNoData(1);
    float viewPercent = meshOptimizer.getViewPercent(newFrame);

    std::cout << "view percent " << viewPercent << std::endl;

    if (viewPercent < 0.80)
    {
        keyFrames.clear();
        keyFrames = lastFrames;

        int newFrameIndex = int(keyFrames.size() / 2);
        meshOptimizer.changeKeyframe(lastFrames[newFrameIndex]);
        keyFrames.erase(keyFrames.begin() + newFrameIndex);
        optimize = true;
    }

    if (optimize)
    {
        t.tic();
        meshOptimizer.optMap(keyFrames);
        std::cout << "optmap time " << t.toc() << std::endl;
    }

    if (lastFrames.size() >= 2)
        lastMovement = lastFrames[lastFrames.size() - 1].pose * lastFrames[lastFrames.size() - 2].pose.inverse();

    meshOptimizer.plotDebug(newFrame);
}
