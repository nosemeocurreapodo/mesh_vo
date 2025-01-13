#include "visualOdometry.h"
#include "utils/tictoc.h"

template class visualOdometry<SceneMesh>;

template <typename sceneType>
visualOdometry<sceneType>::visualOdometry(camera &_cam)
    : cam(_cam),
      poseOptimizer(_cam),
      mapOptimizer(_cam),
      sceneOptimizer(_cam),
      renderer(_cam.width, _cam.height),
      kframe(_cam.width, _cam.height)
{
    lastId = 0;
}

template <typename sceneType>
void visualOdometry<sceneType>::init(dataCPU<float> &image, Sophus::SE3f globalPose)
{
    assert(image.width == cam[0].width && image.height == cam[0].height);

    frameCPU newFrame(cam[0].width, cam[0].height);
    newFrame.setImage(image, 0);
    newFrame.setPose(globalPose);
    newFrame.setAffine(vec2<float>(0.0, 0.0));

    int lvl = 1;

    dataCPU<float> buffer(cam[lvl].width, cam[lvl].height, -1.0);
    // renderer.renderRandom(cam[lvl], buffer, 0.1, 1.0);
    renderer.renderSmooth(cam[lvl], buffer, 0.1, 1.0);

    kframe = newFrame;
    scene.init(buffer, cam[lvl], globalPose);
    // lastPose = globalPose;
}

template <typename sceneType>
void visualOdometry<sceneType>::init(dataCPU<float> &image, dataCPU<float> &idepth, Sophus::SE3f globalPose)
{
    assert(image.width == idepth.width && image.height == idepth.height && image.width == cam[0].width && image.height == cam[0].height);

    frameCPU newFrame(cam[0].width, cam[0].height);
    newFrame.setImage(image, 0);
    newFrame.setPose(globalPose);
    newFrame.setAffine(vec2<float>(0.0, 0.0));

    kframe = newFrame;
    scene.init(idepth, cam[0], globalPose);
    // lastPose = globalPose;
}

template <typename sceneType>
void visualOdometry<sceneType>::init(frameCPU &frame)
{
    int lvl = 1;

    assert(frame.getRawImage(lvl).width == cam[lvl].width && frame.getRawImage(lvl).height == cam[lvl].height);

    dataCPU<float> buffer(cam[lvl].width, cam[lvl].height, -1.0);

    renderer.renderSmooth(cam[lvl], buffer, 0.1, 1.0);

    kframe = frame;
    scene.init(buffer, cam[lvl], frame.getPose());
    // lastPose = frame.getPose();
}

template <typename sceneType>
void visualOdometry<sceneType>::init(frameCPU &frame, dataCPU<float> &idepth)
{
    assert(frame.getRawImage(0).width == idepth.width && frame.getRawImage(0).height == idepth.height);

    kframe = frame;
    scene.init(idepth, cam[0], frame.getPose());
    // lastPose = frame.getPose();
}

template <typename sceneType>
void visualOdometry<sceneType>::init(frameCPU &frame, std::vector<vec2<float>> &pixels, std::vector<float> &idepths)
{
    assert(frame.getRawImage(0).width == cam[0].width && frame.getRawImage(0).height == cam[0].height);

    kframe = frame;
    scene.init(pixels, idepths, cam[0], frame.getPose());
    // lastPose = frame.getPose();
}

template <typename sceneType>
dataCPU<float> visualOdometry<sceneType>::getIdepth(Sophus::SE3f pose, int lvl)
{
    dataCPU<float> buffer(cam[lvl].width, cam[lvl].height, -1);

    renderer.renderIdepthParallel(scene, pose, cam[lvl], buffer);
    // renderer.renderInterpolate(cam[lvl], &idepth_buffer, lvl);
    return buffer;
}

template <typename sceneType>
float visualOdometry<sceneType>::meanViewAngle(Sophus::SE3f pose1, Sophus::SE3f pose2)
{
    int lvl = 1;

    sceneType scene1 = scene;
    scene1.transform(cam[lvl], pose1);

    sceneType scene2 = scene;
    scene2.transform(cam[lvl], pose2);

    Sophus::SE3f relativePose = pose1 * pose2.inverse();

    Sophus::SE3f frame1PoseInv = relativePose.inverse();
    Sophus::SE3f frame2PoseInv = Sophus::SE3f();

    Eigen::Vector3f frame1Tra = frame1PoseInv.translation();
    Eigen::Vector3f frame2Tra = frame2PoseInv.translation();

    vec3<float> frame1Translation(frame1Tra(0), frame1Tra(1), frame1Tra(2));
    vec3<float> frame2Translation(frame2Tra(0), frame2Tra(1), frame2Tra(2));

    std::vector<int> sIds = scene2.getShapesIds();

    float accAngle = 0;
    int count = 0;
    for (auto sId : sIds)
    {
        auto shape1 = scene1.getShape(sId);
        auto shape2 = scene2.getShape(sId);

        vec2<float> centerPix1 = shape1.getCenterPix();
        vec2<float> centerPix2 = shape2.getCenterPix();

        float centerDepth2 = shape2.getDepth(centerPix2);

        if (!cam[lvl].isPixVisible(centerPix1) || !cam[lvl].isPixVisible(centerPix2))
            continue;

        vec3<float> centerRay2 = cam[lvl].pixToRay(centerPix2);
        vec3<float> centerPoint2 = centerRay2 * centerDepth2;

        vec3<float> diff1 = frame1Translation - centerPoint2;
        vec3<float> diff2 = frame2Translation - centerPoint2;
        vec3<float> diff1Normalized = diff1 / diff1.norm();
        vec3<float> diff2Normalized = diff2 / diff2.norm();

        float cos_angle = diff1Normalized.dot(diff2Normalized);
        float angle = std::acos(cos_angle);

        accAngle += std::fabs(angle);
        count += 1;
    }

    return accAngle / count;
}

template <typename sceneType>
float visualOdometry<sceneType>::getViewPercent(frameCPU &frame)
{
    int lvl = 1;
    /*
    sceneType scene1 = scene;
    scene1.transform(cam[lvl], frame.getPose());
    std::vector<int> shapeIds = scene1.getShapesIds();

    int numVisible = 0;
    for (auto shapeId : shapeIds)
    {
        auto shape = scene1.getShape(shapeId);
        vec2<float> pix = shape.getCenterPix();
        float depth = shape.getDepth(pix);
        if (cam[lvl].isPixVisible(pix) && depth > 0.0f)
            numVisible++;
    }
    return float(numVisible) / shapeIds.size();

    */

    dataCPU<float> idepth(cam[lvl].width, cam[lvl].height, -1);
    renderer.renderIdepthParallel(scene, frame.getPose(), cam[lvl], idepth);
    float pnodata = idepth.getPercentNoData();
    return 1.0 - pnodata;
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
    newFrame.setAffine(lastAffine);
    lastId++;

    t.tic();
    poseOptimizer.optimize(newFrame, kframe, scene);
    // sceneOptimizer.optPose(newFrame, kframe, scene);
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
        float keyframeViewAngle = meanViewAngle(newFrame.getPose(), kframe.getPose());
        // dataCPU<float> idepth = meshOptimizer.getIdepth(newFrame.getPose(), 1);
        // float percentNoData = idepth.getPercentNoData(1);

        float viewPercent = getViewPercent(newFrame);

        // std::cout << "mean viewAngle " << meanViewAngle << std::endl;
        // std::cout << "view percent " << viewPercent << std::endl;
        // std::cout << "percent nodata " << percentNoData << std::endl;

        if (lastViewAngle > LAST_MIN_ANGLE) //(percentNoData > 0.2) //(viewPercent < 0.8)
        {
            lastFrames.push_back(newFrame);
            if (lastFrames.size() > NUM_FRAMES)
                lastFrames.erase(lastFrames.begin());
        }

        if ((viewPercent < MIN_VIEW_PERC || keyframeViewAngle > KEY_MAX_ANGLE) && lastFrames.size() > 1)
        // if((viewPercent < 0.9 || meanViewAngle > M_PI / 64.0) && lastFrames.size() > 1)
        {
            int newKeyframeIndex = int(lastFrames.size() / 2);
            // int newKeyframeIndex = int(lastFrames.size() - 1);
            frameCPU newKeyframe = lastFrames[newKeyframeIndex];

            keyFrames = lastFrames;
            keyFrames.erase(keyFrames.begin() + newKeyframeIndex);

            int lvl = 0;
            dataCPU<float> idepth_buffer(cam[lvl].width, cam[lvl].height, -1);
            renderer.renderIdepthParallel(scene, newKeyframe.getPose(), cam[lvl], idepth_buffer);
            renderer.renderInterpolate(cam[lvl], idepth_buffer);
            init(newKeyframe, idepth_buffer);

            optimize = true;
        }
    }

    if (optimize)
    {
        t.tic();
        /*
        sceneOptimizer.optMap(keyFrames, kframe, scene);
        for(auto &keyframe : keyFrames)
        {
            sceneOptimizer.optPose(keyframe, kframe, scene);
        }
        */
        sceneOptimizer.optPoseMap(keyFrames, kframe, scene);
        std::cout << "optposemap time: " << t.toc() << std::endl;

        // sync the updated keyframe poses present in lastframes
        for (auto keyframe : keyFrames)
        {
            if (newFrame.getId() == keyframe.getId())
            {
                newFrame.setPose(keyframe.getPose());
                newFrame.setAffine(keyframe.getAffine());
            }
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
    newFrame.setImage(image, lastId);
    newFrame.setPose(pose);
    newFrame.setAffine(lastAffine);
    lastId++;

    t.tic();
    sceneOptimizer.optLightAffine(newFrame, kframe, scene);
    std::cout << "optmap time " << t.toc() << std::endl;

    lastAffine = newFrame.getAffine();
}

template <typename sceneType>
void visualOdometry<sceneType>::localization(dataCPU<float> &image)
{
    assert(image.width == cam[0].width && image.height == cam[0].height);

    tic_toc t;

    frameCPU newFrame(cam[0].width, cam[0].height);
    newFrame.setImage(image, lastId);
    newFrame.setPose(lastMovement * lastPose);
    newFrame.setAffine(lastAffine);
    lastId++;

    t.tic();
    // sceneOptimizer.optPose(newFrame, kframe, scene);
    poseOptimizer.optimize(newFrame, kframe, scene);
    std::cout << "estimated pose " << t.toc() << std::endl;
    std::cout << newFrame.getPose().matrix() << std::endl;
    std::cout << newFrame.getAffine()(0) << " " << newFrame.getAffine()(1) << std::endl;

    lastMovement = newFrame.getPose() * lastPose.inverse();
    lastPose = newFrame.getPose();
    lastAffine = newFrame.getAffine();
}

template <typename sceneType>
void visualOdometry<sceneType>::mapping(dataCPU<float> &image, Sophus::SE3f globalPose, vec2<float> exp)
{
    assert(image.width == cam[0].width && image.height == cam[0].height);

    tic_toc t;
    bool optimize = false;

    frameCPU newFrame(cam[0].width, cam[0].height);
    newFrame.setImage(image, lastId);
    newFrame.setPose(globalPose);
    newFrame.setAffine(exp);
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
        float keyframeViewAngle = meanViewAngle(newFrame.getPose(), kframe.getPose());
        // dataCPU<float> idepth = meshOptimizer.getIdepth(newFrame.getPose(), 1);
        // float percentNoData = idepth.getPercentNoData(1);

        float viewPercent = getViewPercent(newFrame);

        // std::cout << "mean viewAngle " << meanViewAngle << std::endl;
        // std::cout << "view percent " << viewPercent << std::endl;
        // std::cout << "percent nodata " << percentNoData << std::endl;

        if (lastViewAngle > LAST_MIN_ANGLE) //(percentNoData > 0.2) //(viewPercent < 0.8)
        {
            lastFrames.push_back(newFrame);
            if (lastFrames.size() > NUM_FRAMES)
                lastFrames.erase(lastFrames.begin());
        }

        if ((viewPercent < MIN_VIEW_PERC || keyframeViewAngle > KEY_MAX_ANGLE) && lastFrames.size() > 1)
        // if((viewPercent < 0.9 || meanViewAngle > M_PI / 64.0) && lastFrames.size() > 1)
        {
            // int newKeyframeIndex = 0;
            int newKeyframeIndex = int(lastFrames.size() / 2);
            // int newKeyframeIndex = int(lastFrames.size() - 1);
            frameCPU newKeyframe = lastFrames[newKeyframeIndex];

            keyFrames = lastFrames;
            keyFrames.erase(keyFrames.begin() + newKeyframeIndex);

            int lvl = 0;
            dataCPU<float> idepth_buffer(cam[lvl].width, cam[lvl].height, -1);
            renderer.renderIdepthParallel(scene, newKeyframe.getPose(), cam[lvl], idepth_buffer);
            renderer.renderInterpolate(cam[lvl], idepth_buffer);
            init(newKeyframe, idepth_buffer);

            optimize = true;
        }
    }

    if (optimize)
    {
        t.tic();
        mapOptimizer.optimize(keyFrames, kframe, scene);
        std::cout << "optmap time: " << t.toc() << std::endl;
    }
}
