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

void visualOdometry::initScene(dataCPU<float> &image, std::vector<vec2<float>> &pixels, std::vector<float> &idepths, Sophus::SE3f pose)
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
    meshOptimizer.initKeyframe(newFrame, pixels, idepths, 0);
}

void visualOdometry::locAndMap(dataCPU<float> &image)
{
    tic_toc t;
    bool optimize = false;

    frameCPU newFrame(cam.width, cam.height);
    newFrame.setImage(image, lastId);
    newFrame.setPose(lastMovement * lastPose);
    // newFrame.setPose(lastPose);
    newFrame.setAffine(lastAffine);
    lastId++;

    t.tic();
    meshOptimizer.optPose(newFrame);
    std::cout << "affine " << newFrame.getAffine()(0) << " " << newFrame.getAffine()(1) << std::endl;
    std::cout << "estimated pose " << t.toc() << std::endl;
    std::cout << newFrame.getPose().matrix() << std::endl;

    if (lastFrames.size() == 0)
    {
        lastFrames.push_back(newFrame);
    }
    else
    {
        float meanViewAngle = meshOptimizer.meanViewAngle(&lastFrames[lastFrames.size() - 1], &newFrame);
        //dataCPU<float> idepth = meshOptimizer.getIdepth(newFrame.getPose(), 1);
        //float percentNoData = idepth.getPercentNoData(1);

        float viewPercent = meshOptimizer.getViewPercent(newFrame);

        //std::cout << "mean viewAngle " << meanViewAngle << std::endl;
        //std::cout << "view percent " << viewPercent << std::endl;
        //std::cout << "percent nodata " << percentNoData << std::endl;

        if (meanViewAngle > M_PI / 64.0) //(percentNoData > 0.2) //(viewPercent < 0.8)
        {            
            lastFrames.push_back(newFrame);
            if(lastFrames.size() > NUM_FRAMES)
                lastFrames.erase(lastFrames.begin());
        }

        if(viewPercent < 0.8 && lastFrames.size() > 1)
        {
            int newKeyframeIndex = int(lastFrames.size() / 2);
            frameCPU newKeyframe = lastFrames[newKeyframeIndex];

            Sophus::SE3f newKeyframePoseInv = newKeyframe.getPose().inverse();
            vec2<float> newKeyframeAffine = newKeyframe.getAffine();

            Sophus::SE3f newPose = newFrame.getPose() * newKeyframePoseInv;
            vec2<float> lastAffine = newFrame.getAffine();
            float newAlpha = lastAffine(0) - newKeyframeAffine(0);
            float newBeta = lastAffine(1) - newKeyframeAffine(1)/std::exp(-newAlpha);
            vec2<float> newAffine(newAlpha, newBeta);

            newFrame.setPose(newPose);
            newFrame.setAffine(newAffine);

            for(int i = 0; i < lastFrames.size(); i++)
            {
                newPose = lastFrames[i].getPose() * newKeyframePoseInv;
                lastAffine = lastFrames[i].getAffine();
                newAlpha = lastAffine(0) - newKeyframeAffine(0);
                newBeta = lastAffine(1) - newKeyframeAffine(1)/std::exp(-newAlpha);
                newAffine = vec2<float>(newAlpha, newBeta);

                lastFrames[i].setPose(newPose);
                lastFrames[i].setAffine(newAffine);
            }

            keyFrames = lastFrames;
            keyFrames.erase(keyFrames.begin() + newKeyframeIndex);

            meshOptimizer.changeKeyframe(newKeyframe);

            optimize = true;
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
    /*
    // keyFrames.clear();
    if(newFrame.getId() % 10 == 0)
    {
        keyFrames.push_back(newFrame);
        if (keyFrames.size() > NUM_FRAMES)
            keyFrames.erase(keyFrames.begin());
        optimize = true;
    }
    */
    if (optimize)
    {
        /*
        t.tic();
        for(frameCPU keyframe : keyFrames)
        {
            meshOptimizer.optPose(keyframe);
        }
        std::cout << "opt poses time " << t.toc() << std::endl;

        meshOptimizer.plotDebug(newFrame, keyFrames);

        t.tic();
        dataCPU<float> mask(cam.width, cam.height, -1);
        meshOptimizer.optMap(keyFrames, mask);
        std::cout << "optmap time " << t.toc() << std::endl;

        meshOptimizer.plotDebug(newFrame, keyFrames);
        */

        t.tic();
        //meshOptimizer.setMeshRegu(200.0);
        meshOptimizer.optPoseMap(keyFrames);
        std::cout << "optposemap time " << t.toc() << std::endl;
       
        // sync the updated keyframe poses present in lastframes
        for (auto keyframe : keyFrames)
        {
            if (newFrame.getId() == keyframe.getId())
            {
                newFrame.setPose(keyframe.getPose());
                newFrame.setAffine(keyframe.getAffine());
            }
            for (int i = 0; i < lastFrames.size(); i++)
            {
                if (keyframe.getId() == lastFrames[i].getId())
                {
                    lastFrames[i].setPose(keyframe.getPose());
                    lastFrames[i].setAffine(keyframe.getAffine());
                }
            }
        }
    }

    meshOptimizer.plotDebug(newFrame, keyFrames);

    lastMovement = newFrame.getPose() * lastPose.inverse();
    lastPose = newFrame.getPose();
    lastAffine = newFrame.getAffine();
}

void visualOdometry::lightaffine(dataCPU<float> &image, Sophus::SE3f pose)
{
    tic_toc t;

    frameCPU newFrame(cam.width, cam.height);
    newFrame.setImage(image, 0);
    newFrame.setPose(pose);

    t.tic();
    meshOptimizer.optLightAffine(newFrame);
    std::cout << "optmap time " << t.toc() << std::endl;

    meshOptimizer.plotDebug(newFrame);

    lastAffine = newFrame.getAffine();
}

void visualOdometry::localization(dataCPU<float> &image)
{
    tic_toc t;

    frameCPU newFrame(cam.width, cam.height);
    newFrame.setImage(image, 0);

    t.tic();
    meshOptimizer.optPose(newFrame);
    std::cout << "estimated pose " << t.toc() << std::endl;
    std::cout << newFrame.getPose().matrix() << std::endl;
    std::cout << newFrame.getAffine()(0) << " " << newFrame.getAffine()(1) << std::endl;

    meshOptimizer.plotDebug(newFrame);

    lastPose = newFrame.getPose();
    lastAffine = newFrame.getAffine();
}

void visualOdometry::mapping(dataCPU<float> &image, Sophus::SE3f pose, vec2<float> affine)
{
    tic_toc t;
    bool optimize = false;

    frameCPU newFrame(cam.width, cam.height);
    newFrame.setImage(image, lastId); //*keyframeData.pose.inverse();
    newFrame.setPose(pose);
    newFrame.setAffine(affine);
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
        meshOptimizer.setMeshRegu(100.0);
        meshOptimizer.optMap(keyFrames, mask);
        std::cout << "optmap time " << t.toc() << std::endl;
    }

    meshOptimizer.plotDebug(newFrame);
}
