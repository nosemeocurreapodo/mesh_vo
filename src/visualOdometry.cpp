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
    lastAffine = vec2<float>(0.0f, 0.0f);
    frameCPU newFrame(cam.width, cam.height);
    newFrame.setImage(image, 0);
    newFrame.setPose(pose);
    newFrame.setAffine(lastAffine);
    meshOptimizer.initKeyframe(newFrame, 0);
}

void visualOdometry::initScene(dataCPU<float> &image, dataCPU<float> &idepth, dataCPU<float> &ivar, Sophus::SE3f pose)
{
    lastFrames.clear();
    keyFrames.clear();
    lastMovement = Sophus::SE3f();
    lastPose = pose;
    lastAffine = vec2<float>(0.0f, 0.0f);
    frameCPU newFrame(cam.width, cam.height);
    newFrame.setImage(image, 0);
    newFrame.setPose(pose);
    newFrame.setAffine(lastAffine);
    meshOptimizer.initKeyframe(newFrame, idepth, ivar, 0);
}

void visualOdometry::locAndMap(dataCPU<float> &image)
{
    tic_toc t;
    bool optimize = false;

    frameCPU newFrame(cam.width, cam.height);
    newFrame.setImage(image, lastId); //*keyframeData.pose.inverse();
    lastId++;
    newFrame.setPose(lastMovement * lastPose);
    newFrame.setAffine(lastAffine);

    t.tic();
    meshOptimizer.optPose(newFrame);
    std::cout << "affine " << newFrame.getAffine()(0) << " " << newFrame.getAffine()(1) << std::endl;
    std::cout << "estimated pose " << t.toc() << std::endl;
    std::cout << newFrame.getPose().matrix() << std::endl;

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
            dataCPU<float> idepth = meshOptimizer.getIdepth(newFrame.getPose(), 1);
            float percentNoData = idepth.getPercentNoData(1);

            float viewPercent = meshOptimizer.getViewPercent(newFrame);
        
            std::cout << "mean viewAngle " << meanViewAngle << std::endl;
            std::cout << "view percent " << viewPercent << std::endl;
            std::cout << "percent nodata " << percentNoData << std::endl;

            if (meanViewAngle > M_PI / 16.0 || viewPercent < 0.8 || percentNoData > 0.2)
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
        meshOptimizer.setMeshRegu(10.0);
        meshOptimizer.optPoseMap(keyFrames);
        std::cout << "optposemap time " << t.toc() << std::endl;

        // sync the updated keyframe poses present in lastframes
        for (auto keyframe : keyFrames)
        {
            for (int i = 0; i < lastFrames.size(); i++)
            {
                if (keyframe.getId() == lastFrames[i].getId())
                {
                    lastFrames[i].getPose() = keyframe.getPose();
                }
                if (newFrame.getId() == lastFrames[i].getId())
                {
                    newFrame.getPose() = lastFrames[i].getPose();
                }
            }
        }
    }

    lastMovement = newFrame.getPose() * lastPose.inverse();
    lastPose = newFrame.getPose();
    lastAffine = newFrame.getAffine();

    meshOptimizer.plotDebug(newFrame, keyFrames);
}

vec2<float> visualOdometry::lightaffine(dataCPU<float> &image, Sophus::SE3f pose)
{
    tic_toc t;

    frameCPU newFrame(cam.width, cam.height);
    newFrame.setImage(image, lastId);
    newFrame.setPose(pose);
    lastId++;

    t.tic();
    meshOptimizer.optLightAffine(newFrame);
    std::cout << "optmap time " << t.toc() << std::endl;

    meshOptimizer.plotDebug(newFrame);
    return newFrame.getAffine();
}

Sophus::SE3f visualOdometry::localization(dataCPU<float> &image)
{
    tic_toc t;

    frameCPU newFrame(cam.width, cam.height);
    newFrame.setImage(image, lastId);
    newFrame.setAffine(lastAffine);
    lastId++;
    if (lastFrames.size() == 0)
        newFrame.setPose(meshOptimizer.kscene.getPose());
    else
        newFrame.setPose(lastMovement * lastFrames[lastFrames.size() - 1].getPose());

    t.tic();
    meshOptimizer.optPose(newFrame);
    std::cout << "estimated pose " << t.toc() << std::endl;
    std::cout << newFrame.getPose().matrix() << std::endl;
    std::cout << newFrame.getAffine()(0) << " " << newFrame.getAffine()(1) << std::endl;

    lastFrames.push_back(newFrame);
    if (lastFrames.size() > NUM_FRAMES)
    {
        lastFrames.erase(lastFrames.begin());
    }

    if (lastFrames.size() >= 2)
        lastMovement = lastFrames[lastFrames.size() - 1].getPose() * lastFrames[lastFrames.size() - 2].getPose().inverse();
    lastAffine = lastFrames[lastFrames.size() - 1].getAffine();

    meshOptimizer.plotDebug(newFrame);

    return newFrame.getPose();
}

void visualOdometry::mapping(dataCPU<float> &image, Sophus::SE3f pose)
{
    tic_toc t;
    bool optimize = false;

    frameCPU newFrame(cam.width, cam.height);
    newFrame.setImage(image, lastId); //*keyframeData.pose.inverse();
    newFrame.setPose(pose);
    lastId++;

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
    /*
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
    */

    meshOptimizer.plotDebug(newFrame);
}
