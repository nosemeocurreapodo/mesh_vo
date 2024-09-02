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
    lastPose = pose;
    frameCPU newFrame(cam.width, cam.height);
    newFrame.set(image, pose);
    meshOptimizer.initKeyframe(newFrame, 0);
}

void visualOdometry::initScene(dataCPU<float> &image, dataCPU<float> &idepth, dataCPU<float> &ivar, Sophus::SE3f pose)
{
    lastFrames.clear();
    keyFrames.clear();
    lastMovement = Sophus::SE3f();
    lastPose = pose;
    frameCPU newFrame(cam.width, cam.height);
    newFrame.set(image, pose);
    meshOptimizer.initKeyframe(newFrame, idepth, ivar, 0);
}

void visualOdometry::locAndMap(dataCPU<float> &image)
{
    tic_toc t;
    bool optimize = false;

    frameCPU newFrame(cam.width, cam.height);
    newFrame.set(image); //*keyframeData.pose.inverse();
    newFrame.id = lastId;
    lastId++;
    newFrame.pose = lastMovement * lastPose;

    t.tic();
    meshOptimizer.optPose(newFrame);
    std::cout << "estimated pose " << t.toc() << std::endl;
    std::cout << newFrame.pose.matrix() << std::endl;

    if (lastFrames.size() < NUM_FRAMES)
    {
        optimize = true;
        lastFrames.push_back(newFrame);
        keyFrames = lastFrames;
    }
    else
    {
        if (lastFrames.size() == NUM_FRAMES)
        {
            float meanViewAngle = meshOptimizer.meanViewAngle(&lastFrames[lastFrames.size() - 1], &newFrame);
            dataCPU<float> idepth = meshOptimizer.getIdepth(newFrame.pose, 1);
            float percentNoData = idepth.getPercentNoData(1);

            float viewPercent = meshOptimizer.getViewPercent(newFrame);
        
            std::cout << "mean viewAngle " << meanViewAngle << std::endl;
            std::cout << "view percent " << viewPercent << std::endl;
            std::cout << "percent nodata " << percentNoData << std::endl;

            if (meanViewAngle > M_PI / 32.0 || viewPercent < 0.7 || percentNoData > 0.3)
            {
                lastFrames.erase(lastFrames.begin());
                lastFrames.push_back(newFrame);

                keyFrames = lastFrames;
                optimize = true;

                int newFrameIndex = int(keyFrames.size() / 2);
                //int newFrameIndex = keyFrames.size() - 1;
                meshOptimizer.changeKeyframe(keyFrames[newFrameIndex]);
                keyFrames.erase(keyFrames.begin() + newFrameIndex);
            }
        }
        else
        {
            // should not happen
            if (lastFrames.size() > NUM_FRAMES)
            {
                lastFrames.erase(lastFrames.begin());
            }
        }
    }

    /*
    dataCPU<float> idepth = meshOptimizer.getIdepth(newFrame.pose, 1);
    float percentNoData = idepth.getPercentNoData(1);

    float viewPercent = meshOptimizer.getViewPercent(newFrame);

    std::cout << "view percent " << viewPercent << std::endl;
    std::cout << "percent nodata " << percentNoData << std::endl;

    if (viewPercent < 0.7 || percentNoData > 0.3)
    {
        int newFrameIndex = int(keyFrames.size() / 2);
        //int newFrameIndex = keyFrames.size() - 2;
        meshOptimizer.changeKeyframe(keyFrames[newFrameIndex]);
        keyFrames.erase(keyFrames.begin() + newFrameIndex);

        //meshOptimizer.changeKeyframe(newFrame);

        keyFrames = lastFrames;
        optimize = true;
    }
    */

    if (optimize)
    {
        t.tic();
        //meshOptimizer.setMeshRegu(0.0);
        //dataCPU<float> mask(cam.width, cam.height, -1);
        //meshOptimizer.optMap(keyFrames, mask);
        meshOptimizer.setMeshRegu(100.0);
        meshOptimizer.optPoseMap(keyFrames);
        std::cout << "optposemap time " << t.toc() << std::endl;

        // sync the updated keyframe poses present in lastframes
        for (auto keyframe : keyFrames)
        {
            for (int i = 0; i < lastFrames.size(); i++)
            {
                if (keyframe.id == lastFrames[i].id)
                {
                    lastFrames[i].pose = keyframe.pose;
                }
                if (newFrame.id == lastFrames[i].id)
                {
                    newFrame.pose = lastFrames[i].pose;
                }
            }
        }
    }

    lastMovement = newFrame.pose * lastPose.inverse();
    lastPose = newFrame.pose;

    meshOptimizer.plotDebug(newFrame, keyFrames);
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
        dataCPU<float> mask(cam.width, cam.height, -1);
        meshOptimizer.setMeshRegu(0.0);
        meshOptimizer.optMap(keyFrames, mask);
        std::cout << "optmap time " << t.toc() << std::endl;
    }

    meshOptimizer.plotDebug(newFrame);
}
