#include "visualOdometry.h"
#include "utils/tictoc.h"

visualOdometry::visualOdometry(camera &_cam)
    : cam(_cam),
      kframe(_cam.width, _cam.height),
      poseOptimizer(_cam),
      mapOptimizer(_cam),
      poseMapOptimizer(_cam),
      renderer(_cam.width, _cam.height)
{
    lastId = 0;
}

void visualOdometry::init(dataCPU<float> &image, SE3f globalPose)
{
    assert(image.width == cam[0].width && image.height == cam[0].height);

    dataCPU<float> buffer(cam[0].width, cam[0].height, -1.0);
    renderer.renderSmooth(cam[0], buffer, 0.1, 1.0);

    kframe.init(image, vec2f(0.0, 0.0), globalPose, 1.0, buffer, cam[0]);
}

void visualOdometry::init(dataCPU<float> &image, SE3f globalPose, dataCPU<float> &idepth)
{
    assert(image.width == idepth.width && image.height == idepth.height && image.width == cam[0].width && image.height == cam[0].height);

    kframe.init(image, vec2f(0.0, 0.0), globalPose, 1.0, idepth, cam[0]);
}

dataCPU<float> visualOdometry::getIdepth(SE3f pose, int lvl)
{
    dataMipMapCPU<float> buffer(cam[0].width, cam[0].height, -1);

    renderer.renderIdepthParallel(kframe, pose, buffer, cam[lvl], lvl);
    // renderer.renderInterpolate(cam[lvl], &idepth_buffer, lvl);
    return buffer.get(lvl);
}

float visualOdometry::meanViewAngle(SE3f pose1, SE3f pose2)
{
    int lvl = 1;

    geometryType scene1 = kframe.getGeometry();
    scene1.transform(pose1);
    scene1.project(cam[lvl]);

    geometryType scene2 = kframe.getGeometry();
    scene2.transform(pose2);
    scene2.project(cam[lvl]);

    SE3f relativePose = pose1 * pose2.inverse();

    SE3f frame1PoseInv = relativePose.inverse();
    SE3f frame2PoseInv = SE3f();

    vec3f frame1Translation = frame1PoseInv.translation();
    vec3f frame2Translation = frame2PoseInv.translation();

    std::vector<int> sIds = scene2.getShapesIds();

    float accAngle = 0;
    int count = 0;
    for (auto sId : sIds)
    {
        //auto shape1 = scene1.getShape(sId);
        auto shape2 = scene2.getShape(sId);

        //vec2f centerPix1 = shape1.getCenterPix();
        vec2f centerPix2 = shape2.getCenterPix();

        //if (!cam[lvl].isPixVisible(centerPix1) || !cam[lvl].isPixVisible(centerPix2))
        //    continue;

        //float centerDepth1 = shape1.getDepth(centerPix1);
        float centerDepth2 = shape2.getDepth(centerPix2);

        vec3f centerRay2 = cam[lvl].pixToRay(centerPix2);
        vec3f centerPoint2 = centerRay2 * centerDepth2;

        vec3f diff1 = centerPoint2 - frame1Translation;
        vec3f diff2 = centerPoint2 - frame2Translation;
        vec3f diff1Normalized = diff1 / diff1.norm();
        vec3f diff2Normalized = diff2 / diff2.norm();

        float cos_angle = diff1Normalized.dot(diff2Normalized);
        float angle = std::acos(cos_angle);

        accAngle += std::fabs(angle);
        count += 1;
    }

    return accAngle / count;
}

float visualOdometry::getViewPercent(frameCPU &frame)
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

    dataMipMapCPU<float> idepth(cam[0].width, cam[0].height, -1);
    renderer.renderIdepthParallel(kframe, frame.getLocalPose(), idepth, cam, lvl);
    float pnodata = idepth.getPercentNoData(lvl);
    return 1.0 - pnodata;
}

void visualOdometry::locAndMap(dataCPU<float> &image)
{
    assert(image.width == cam[0].width && image.height == cam[0].height);

    tic_toc t;
    bool optimize = false;

    frameCPU newFrame(cam[0].width, cam[0].height);
    newFrame.setImage(image, lastId);
    newFrame.setLocalPose(lastMovement * lastPose);
    newFrame.setLocalExp(lastExposure);
    lastId++;

    t.tic();
    poseOptimizer.optimize(newFrame, kframe);
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
        float lastViewAngle = meanViewAngle(lastFrames[lastFrames.size() - 1].getLocalPose(), newFrame.getLocalPose());
        float keyframeViewAngle = meanViewAngle(SE3f(), newFrame.getLocalPose());
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
            //int newKeyframeIndex = int(lastFrames.size() - 1);
            frameCPU newKeyframe = lastFrames[newKeyframeIndex];

            keyFrames = lastFrames;
            keyFrames.erase(keyFrames.begin() + newKeyframeIndex);

            int lvl = 0;
            dataMipMapCPU<float> idepth_buffer(cam[0].width, cam[0].height, -1);
            renderer.renderIdepthParallel(kframe, newKeyframe.getLocalPose(), idepth_buffer, cam, lvl);
            renderer.renderInterpolate(cam[lvl], idepth_buffer.get(lvl));

            SE3f newKeyframeGlobalPose = newKeyframe.getLocalPose() * kframe.getGlobalPose();

            updateLocalPoses(newKeyframeGlobalPose);

            kframe.init(newKeyframe, 1.0, idepth_buffer.get(lvl), cam[lvl]);

            vec2f meanStd = kframe.getGeometry().meanStdDepth();
            updateLocalScale(1.0/meanStd(0));

            kframe.scaleVertices(1.0/meanStd(0));

            optimize = true;
        }
    }

    if (optimize)
    {
        t.tic();
        poseMapOptimizer.optimize(keyFrames, kframe);
        std::cout << "optposemap time: " << t.toc() << std::endl;

        // sync the updated keyframe poses present in lastframes
        for (auto keyframe : keyFrames)
        {
            if (newFrame.getId() == keyframe.getId())
            {
                newFrame.setLocalPose(keyframe.getLocalPose());
                newFrame.setLocalExp(keyframe.getLocalExp());
            }
            for (int i = 0; i < (int)lastFrames.size(); i++)
            {
                if (keyframe.getId() == lastFrames[i].getId())
                {
                    lastFrames[i].setLocalPose(keyframe.getLocalPose());
                    lastFrames[i].setLocalExp(keyframe.getLocalExp());
                }
            }
        }
    }

    lastMovement = newFrame.getLocalPose() * lastPose.inverse();
    lastPose = newFrame.getLocalPose();
    lastExposure = newFrame.getLocalExp();
}

void visualOdometry::lightaffine(dataCPU<float> &image, SE3f pose)
{
    assert(image.width == cam[0].width && image.height == cam[0].height);

    tic_toc t;

    frameCPU newFrame(cam[0].width, cam[0].height);
    newFrame.setImage(image, lastId);
    newFrame.setLocalPose(pose);
    newFrame.setLocalExp(lastExposure);
    lastId++;

    t.tic();
    //sceneOptimizer.optLightAffine(newFrame, kframe, scene);
    std::cout << "optmap time " << t.toc() << std::endl;

    lastExposure = newFrame.getLocalExp();
}

void visualOdometry::localization(dataCPU<float> &image)
{
    assert(image.width == cam[0].width && image.height == cam[0].height);

    tic_toc t;

    frameCPU newFrame(cam[0].width, cam[0].height);
    newFrame.setImage(image, lastId);
    newFrame.setLocalPose(lastMovement * lastPose);
    newFrame.setLocalExp(lastExposure);
    lastId++;

    t.tic();
    poseOptimizer.optimize(newFrame, kframe);
    std::cout << "estimated pose " << t.toc() << std::endl;
    std::cout << newFrame.getLocalPose().matrix() << std::endl;
    std::cout << newFrame.getLocalExp()(0) << " " << newFrame.getLocalExp()(1) << std::endl;

    lastMovement = newFrame.getLocalPose() * lastPose.inverse();
    lastPose = newFrame.getLocalPose();
    lastExposure = newFrame.getLocalExp();
}

void visualOdometry::mapping(dataCPU<float> &image, SE3f globalPose, vec2f exp)
{
    assert(image.width == cam[0].width && image.height == cam[0].height);

    tic_toc t;
    bool optimize = false;

    frameCPU newFrame(cam[0].width, cam[0].height);
    newFrame.setImage(image, lastId);
    newFrame.setLocalPose(globalPose);
    newFrame.setLocalExp(exp);
    lastId++;

    if (lastFrames.size() == 0)
    {
        lastFrames.push_back(newFrame);
        keyFrames = lastFrames;
        optimize = true;
    }
    else
    {
        float lastViewAngle = meanViewAngle(lastFrames[lastFrames.size() - 1].getLocalPose(), newFrame.getLocalPose());
        float keyframeViewAngle = meanViewAngle(SE3f(), newFrame.getLocalPose());
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
            //int newKeyframeIndex = int(lastFrames.size() - 1);
            frameCPU newKeyframe = lastFrames[newKeyframeIndex];

            keyFrames = lastFrames;
            keyFrames.erase(keyFrames.begin() + newKeyframeIndex);

            int lvl = 0;
            dataMipMapCPU<float> idepth_buffer(cam[0].width, cam[0].height, -1);
            renderer.renderIdepthParallel(kframe, newKeyframe.getLocalPose(), idepth_buffer, cam[lvl], lvl);
            renderer.renderInterpolate(cam[lvl], idepth_buffer.get(lvl));

            SE3f newKeyframeGlobalPose = newKeyframe.getLocalPose() * kframe.getGlobalPose();

            updateLocalPoses(newKeyframeGlobalPose);

            kframe.init(newKeyframe, 1.0, idepth_buffer.get(lvl), cam[lvl]);

            vec2f meanStd = kframe.getGeometry().meanStdDepth();
            updateLocalScale(1.0/meanStd(0));

            kframe.scaleVertices(1.0/meanStd(0));

            optimize = true;
        }
    }

    if (optimize)
    {
        t.tic();
        mapOptimizer.optimize(keyFrames, kframe);
        std::cout << "optmap time: " << t.toc() << std::endl;
    }
}
