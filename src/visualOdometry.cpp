#include "visualOdometry.h"
#include "utils/tictoc.h"

template class visualOdometry<SceneMesh>;

template <typename sceneType>
visualOdometry<sceneType>::visualOdometry(camera &_cam)
    : cam(_cam),
      sceneOptimizer(_cam),
      renderer(_cam.width, _cam.height),
      kframe(_cam.width, _cam.height)
{
    lastId = 0;
}

template <typename sceneType>
void visualOdometry<sceneType>::init(dataCPU<float> &image, Sophus::SE3f pose)
{
    assert(image.width == cam[0].width && image.height == cam[0].height);

    frameCPU newFrame(cam[0].width, cam[0].height);
    newFrame.setImage(image, 0);
    newFrame.setPose(pose);
    newFrame.setAffine(vec2<float>(0.0, 0.0));

    lastFrames.clear();
    keyFrames.clear();
    lastMovement = Sophus::SE3f();
    lastPose = pose;
    lastAffine = vec2<float>(0.0f, 0.0f);

    int lvl = 1;

    dataCPU<float> buffer(cam[lvl].width, cam[lvl].height, -1.0);
    renderer.renderSmooth(cam[lvl], buffer, 0.1, 1.0);

    kframe = newFrame;
    kscene.init(buffer, cam[lvl]);
}

template <typename sceneType>
void visualOdometry<sceneType>::init(dataCPU<float> &image, dataCPU<float> &idepth, Sophus::SE3f pose)
{
    assert(image.width == idepth.width && image.height == idepth.height);

    frameCPU newFrame(cam[0].width, cam[0].height);
    newFrame.setImage(image, 0);
    newFrame.setPose(pose);
    newFrame.setAffine(vec2<float>(0.0, 0.0));

    lastFrames.clear();
    keyFrames.clear();
    lastMovement = Sophus::SE3f();
    lastPose = pose;
    lastAffine = vec2<float>(0.0f, 0.0f);

    kframe = newFrame;
    kscene.init(idepth, cam[0]);
}

template <typename sceneType>
void visualOdometry<sceneType>::init(frameCPU &frame)
{
    int lvl = 1;

    assert(frame.getRawImage(lvl).width == cam[lvl].width && frame.getRawImage(lvl).height == cam[lvl].height);

    lastFrames.clear();
    keyFrames.clear();
    lastMovement = Sophus::SE3f();
    lastPose = frame.getPose();
    lastAffine = vec2<float>(0.0f, 0.0f);

    dataCPU<float> buffer(cam[lvl].width, cam[lvl].height, -1.0);

    renderer.renderSmooth(cam[lvl], buffer, 0.1, 1.0);

    kframe = frame;
    kscene.init(buffer, cam[lvl]);
}

template <typename sceneType>
void visualOdometry<sceneType>::init(frameCPU &frame, dataCPU<float> &idepth)
{
    assert(frame.getRawImage(0).width == idepth.width && frame.getRawImage(0).height == idepth.height);

    lastFrames.clear();
    keyFrames.clear();
    lastMovement = Sophus::SE3f();
    lastPose = frame.getPose();
    lastAffine = vec2<float>(0.0f, 0.0f);

    kframe = frame;
    kscene.init(idepth, cam[0]);
}

template <typename sceneType>
void visualOdometry<sceneType>::init(frameCPU &frame, std::vector<vec2<float>> &pixels, std::vector<float> &idepths)
{
    assert(frame.getRawImage(0).width == cam[0].width && frame.getRawImage(0).height == cam[0].height);

    lastFrames.clear();
    keyFrames.clear();
    lastMovement = Sophus::SE3f();
    lastPose = frame.getPose();
    lastAffine = vec2<float>(0.0f, 0.0f);

    kframe = frame;
    kscene.init(pixels, idepths, cam[0]);
}

template <typename sceneType>
void visualOdometry<sceneType>::locAndMap(dataCPU<float> &image)
{
    assert(image.width == cam[0].width && image.height == cam[0].height);

    tic_toc t;
    bool optimize = false;

    frameCPU newFrame(cam[0].width, cam[0].height);
    newFrame.setImage(image, lastId);
    newFrame.setPose(lastMovement * lastPose);
    // newFrame.setPose(lastPose);
    newFrame.setAffine(lastAffine);
    lastId++;

    t.tic();
    sceneOptimizer.optPose(newFrame, kframe, kscene);
    std::cout << "estimate pose time: " << t.toc() << std::endl;

    // std::cout << "affine " << newFrame.getAffine()(0) << " " << newFrame.getAffine()(1) << std::endl;
    // std::cout << "estimated pose " << std::endl;
    // std::cout << newFrame.getPose().matrix() << std::endl;

    if (lastFrames.size() == 0)
    {
        lastFrames.push_back(newFrame);
        keyFrames = lastFrames;
        optimize = true;
    }
    else
    {
        float lastViewAngle = meanViewAngle(lastFrames[lastFrames.size() - 1].getPose(), newFrame.getPose());
        float keyframeViewAngle = meanViewAngle(newFrame.getPose(), Sophus::SE3f());
        // dataCPU<float> idepth = meshOptimizer.getIdepth(newFrame.getPose(), 1);
        // float percentNoData = idepth.getPercentNoData(1);

        float viewPercent = getViewPercent(newFrame);

        // std::cout << "mean viewAngle " << meanViewAngle << std::endl;
        // std::cout << "view percent " << viewPercent << std::endl;
        // std::cout << "percent nodata " << percentNoData << std::endl;

        if (lastViewAngle > LAST_MAX_ANGLE) //(percentNoData > 0.2) //(viewPercent < 0.8)
        {
            lastFrames.push_back(newFrame);
            if (lastFrames.size() > NUM_FRAMES)
                lastFrames.erase(lastFrames.begin());
        }

        if ((viewPercent < MIN_VIEW_PERC || keyframeViewAngle > KEY_MAX_ANGLE) && lastFrames.size() > 1)
        // if((viewPercent < 0.9 || meanViewAngle > M_PI / 64.0) && lastFrames.size() > 1)
        {
            // int newKeyframeIndex = int(lastFrames.size() / 2);
            int newKeyframeIndex = int(lastFrames.size() - 1);
            frameCPU newKeyframe = lastFrames[newKeyframeIndex];

            Sophus::SE3f newKeyframePoseInv = newKeyframe.getPose().inverse();
            vec2<float> newKeyframeAffine = newKeyframe.getAffine();

            Sophus::SE3f newPose = newFrame.getPose() * newKeyframePoseInv;
            vec2<float> lastAffine = newFrame.getAffine();
            float newAlpha = lastAffine(0) - newKeyframeAffine(0);
            float newBeta = lastAffine(1) - newKeyframeAffine(1) / std::exp(-newAlpha);
            vec2<float> newAffine(newAlpha, newBeta);

            newFrame.setPose(newPose);
            newFrame.setAffine(newAffine);

            lastPose = lastPose * newKeyframePoseInv;

            for (int i = 0; i < (int)lastFrames.size(); i++)
            {
                newPose = lastFrames[i].getPose() * newKeyframePoseInv;
                lastAffine = lastFrames[i].getAffine();
                newAlpha = lastAffine(0) - newKeyframeAffine(0);
                newBeta = lastAffine(1) - newKeyframeAffine(1) / std::exp(-newAlpha);
                newAffine = vec2<float>(newAlpha, newBeta);

                lastFrames[i].setPose(newPose);
                lastFrames[i].setAffine(newAffine);
            }

            keyFrames = lastFrames;
            keyFrames.erase(keyFrames.begin() + newKeyframeIndex);

            int lvl = 0;

            dataCPU<float> idepth_buffer(cam[lvl].width, cam[lvl].height, -1);

            renderer.renderIdepthParallel(kscene, newKeyframe.getPose(), cam[lvl], idepth_buffer);
            renderer.renderInterpolate(cam[lvl], idepth_buffer);

            init(newKeyframe, idepth_buffer);

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
        */

        /*
        t.tic();
        dataCPU<float> mask(cam.width, cam.height, -1);
        meshOptimizer.optMap(keyFrames, mask);
        std::cout << "optmap time " << t.toc() << std::endl;
        */

        t.tic();
        // meshOptimizer.setMeshRegu(200.0);
        // dataCPU<float> mask(cam.width, cam.height, -1);
        // meshOptimizer.optMap(keyFrames);
        // meshOptimizer.normalizeDepth();

        sceneOptimizer.optPoseMap(keyFrames, kframe, kscene);
        // meshOptimizer.normalizeDepth();
        // vec2<float> depthAffine = meshOptimizer.kDepthAffine;

        // Sophus::SE3f pose = newFrame.getPose();
        // pose.translation() = pose.translation()/depthAffine(0);
        // newFrame.setPose(pose);

        // for(int i = 0; i < keyFrames.size(); i++)
        //{
        //     Sophus::SE3f pose = keyFrames[i].getPose();
        //     pose.translation() = pose.translation()/depthAffine(0);
        //     keyFrames[i].setPose(pose);
        // }

        std::cout << "optposemap time: " << t.toc() << std::endl;

        // sync the updated keyframe poses present in lastframes
        for (auto keyframe : keyFrames)
        {
            /*
            if (newFrame.getId() == keyframe.getId())
            {
                newFrame.setPose(keyframe.getPose());
                newFrame.setAffine(keyframe.getAffine());
            }
            */
            for (int i = 0; i < (int)lastFrames.size(); i++)
            {
                if (keyframe.getId() == lastFrames[i].getId())
                {
                    lastFrames[i].setPose(keyframe.getPose());
                    lastFrames[i].setAffine(keyframe.getAffine());
                }
            }
        }
    }

    plotDebug(newFrame, keyFrames);

    lastMovement = newFrame.getPose() * lastPose.inverse();
    lastPose = newFrame.getPose();
    lastAffine = newFrame.getAffine();
}

template <typename sceneType>
void visualOdometry<sceneType>::lightaffine(dataCPU<float> &image, Sophus::SE3f pose)
{
    assert(image.width == cam[0].width && image.height == cam[0].height);

    tic_toc t;

    frameCPU newFrame(cam[0].width, cam[0].height);
    newFrame.setImage(image, 0);
    newFrame.setPose(pose);

    t.tic();
    sceneOptimizer.optLightAffine(newFrame, kframe, kscene);
    std::cout << "optmap time " << t.toc() << std::endl;

    plotDebug(newFrame);

    lastAffine = newFrame.getAffine();
}

template <typename sceneType>
void visualOdometry<sceneType>::localization(dataCPU<float> &image)
{
    assert(image.width == cam[0].width && image.height == cam[0].height);

    tic_toc t;

    frameCPU newFrame(cam[0].width, cam[0].height);
    newFrame.setImage(image, 0);

    t.tic();
    sceneOptimizer.optPose(newFrame, kframe, kscene);
    std::cout << "estimated pose " << t.toc() << std::endl;
    std::cout << newFrame.getPose().matrix() << std::endl;
    std::cout << newFrame.getAffine()(0) << " " << newFrame.getAffine()(1) << std::endl;

    plotDebug(newFrame);

    lastPose = newFrame.getPose();
    lastAffine = newFrame.getAffine();
}

template <typename sceneType>
void visualOdometry<sceneType>::mapping(dataCPU<float> &image, Sophus::SE3f globalPose, vec2<float> affine)
{
    assert(image.width == cam[0].width && image.height == cam[0].height);

    tic_toc t;
    bool optimize = false;

    frameCPU newFrame(cam[0].width, cam[0].height);
    newFrame.setImage(image, lastId);
    newFrame.setPose(globalPose);
    // newFrame.setPose(lastPose);
    newFrame.setAffine(affine);
    lastId++;

    if (lastFrames.size() == 0)
    {
        lastFrames.push_back(newFrame);
        keyFrames = lastFrames;
        optimize = true;
    }
    else
    {
        float lastViewAngle = meanViewAngle(lastFrames[lastFrames.size() - 1].getPose(), newFrame.getPose());
        float keyframeViewAngle = meanViewAngle(newFrame.getPose(), Sophus::SE3f());
        // dataCPU<float> idepth = meshOptimizer.getIdepth(newFrame.getPose(), 1);
        // float percentNoData = idepth.getPercentNoData(1);

        float viewPercent = getViewPercent(newFrame);

        // std::cout << "mean viewAngle " << meanViewAngle << std::endl;
        // std::cout << "view percent " << viewPercent << std::endl;
        // std::cout << "percent nodata " << percentNoData << std::endl;

        if (lastViewAngle > LAST_MAX_ANGLE) //(percentNoData > 0.2) //(viewPercent < 0.8)
        {
            lastFrames.push_back(newFrame);
            if (lastFrames.size() > NUM_FRAMES)
                lastFrames.erase(lastFrames.begin());
        }

        if ((viewPercent < MIN_VIEW_PERC || keyframeViewAngle > KEY_MAX_ANGLE) && lastFrames.size() > 1)
        // if((viewPercent < 0.9 || meanViewAngle > M_PI / 64.0) && lastFrames.size() > 1)
        {
            int newKeyframeIndex = 0;
            // int newKeyframeIndex = int(lastFrames.size() / 2);
            // int newKeyframeIndex = int(lastFrames.size() - 1);
            frameCPU newKeyframe = lastFrames[newKeyframeIndex];

            Sophus::SE3f newKeyframePoseInv = newKeyframe.getPose().inverse();
            vec2<float> newKeyframeAffine = newKeyframe.getAffine();

            for (int i = 0; i < (int)lastFrames.size(); i++)
            {
                Sophus::SE3f newPose = lastFrames[i].getPose() * newKeyframePoseInv;
                vec2<float> lastAffine = lastFrames[i].getAffine();
                float newAlpha = lastAffine(0) - newKeyframeAffine(0);
                float newBeta = lastAffine(1) - newKeyframeAffine(1) / std::exp(-newAlpha);
                vec2<float> newAffine = vec2<float>(newAlpha, newBeta);

                lastFrames[i].setPose(newPose);
                lastFrames[i].setAffine(newAffine);
            }

            keyFrames = lastFrames;
            keyFrames.erase(keyFrames.begin() + newKeyframeIndex);

            Sophus::SE3f newPose = newFrame.getPose() * newKeyframePoseInv;
            vec2<float> lastAffine = newFrame.getAffine();
            float newAlpha = lastAffine(0) - newKeyframeAffine(0);
            float newBeta = lastAffine(1) - newKeyframeAffine(1) / std::exp(-newAlpha);
            vec2<float> newAffine(newAlpha, newBeta);

            newFrame.setPose(newPose);
            newFrame.setAffine(newAffine);

            int lvl = 0;

            dataCPU<float> idepth_buffer(cam[lvl].width, cam[lvl].height, -1);

            renderer.renderIdepthParallel(kscene, newKeyframe.getPose(), cam[lvl], idepth_buffer);
            renderer.renderInterpolate(cam[lvl], idepth_buffer);

            init(newKeyframe, idepth_buffer);

            optimize = true;
        }

        plotDebug(newFrame, keyFrames);
    }

    if (optimize)
    {
        t.tic();
        sceneOptimizer.optMap(keyFrames, kframe, kscene);
        std::cout << "optmap time: " << t.toc() << std::endl;

        plotDebug(newFrame, keyFrames);
    }
}
