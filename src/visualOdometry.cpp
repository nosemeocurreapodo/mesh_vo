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
    meshOptimizer.initKeyframe(lastFrame, 0);
}

void visualOdometry::initScene(dataCPU<float> &image, dataCPU<float> &idepth, Sophus::SE3f pose)
{
    lastFrame.set(image, pose);
    meshOptimizer.initKeyframe(lastFrame, idepth, 0);
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

    float angle = meshOptimizer.meanViewAngle(&lastFrame, &newFrame);

    // vec3<float> lastRay = vec3<float>(lastPoint_e(0), lastPoint_e(1), lastPoint_e(2));
    // vec3<float> lastRotatedRay = last

    // vec2<float> lastPix = cam.rayToPix(lastRay);
    // float pixBaseline = (lastPix - centerPix).norm();
    // std::cout << "pixBaseline " << pixBaseline << std::endl;

    // float baseline = (newFrame.pose.translation() - lastFrame.pose.translation()).norm();
    // float centerIdepth = idepth.get(int(cam.height/2), int(cam.width/2), 0);
    // float relBaseline = baseline*centerIdepth;

    // std::cout << "relBaseline " << relBaseline << " idepth " << centerIdepth << " baseline " << baseline << std::endl;

    bool optimize = false;
    if (frames.size() == 0 || angle > M_PI / 32.0)
    {
        frames.push_back(newFrame);
        optimize = true;
    }

    if (frames.size() > 3)
    {
        frames.erase(frames.begin());
    }

    if (optimize)
    {
        // t.tic();
        // meshOptimizer.optMap(frames);
        // std::cout << "optmap time " << t.toc() << std::endl;

        t.tic();
        meshOptimizer.optPoseMap(frames);
        std::cout << "optposemap time " << t.toc() << std::endl;
        newFrame.pose = frames[frames.size() - 1].pose;
    }

    lastMovement = newFrame.pose * lastFrame.pose.inverse();
    lastFrame = newFrame;

    dataCPU<float> idepth = meshOptimizer.getIdepth(newFrame.pose, 1);
    float percentNoData = idepth.getPercentNoData(1);

    if (percentNoData > 0.20)
    {
        meshOptimizer.changeKeyframe(newFrame);
        frames.erase(frames.end());
        // t.tic();
        // meshOptimizer.optPoseMap(frames);
        // std::cout << "optposemap time " << t.toc() << std::endl;
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
    // t.tic();
    // meshOptimizer.optPose(newFrame);
    // std::cout << "estimated pose " << t.toc() << std::endl;
    // std::cout << newFrame.pose.matrix() << std::endl;
    // lastMovement = newFrame.pose * lastFrame.pose.inverse();
    lastFrame = newFrame;

    dataCPU<float> idepth = meshOptimizer.getIdepth(newFrame.pose, 1);
    float percentNoData = idepth.getPercentNoData(1);

    bool updateMap = false;

    if (percentNoData > 0.20)
    {
        meshOptimizer.changeKeyframe(newFrame);
        if (frames.size() > 0)
            updateMap = true;
    }
    else
    {
        frames.push_back(newFrame);
        if (frames.size() > 3)
        {
            frames.erase(frames.begin());
        }
        updateMap = true;
    }

    if (updateMap)
    {
        t.tic();
        meshOptimizer.optMap(frames);
        std::cout << "optmap time " << t.toc() << std::endl;
    }

    meshOptimizer.plotDebug(newFrame);
}
