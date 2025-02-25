#include "visualOdometry.h"
#include "utils/tictoc.h"

visualOdometry::visualOdometry(cameraType _cam, int width, int height)
    : cam(_cam),
      kframe(width, height),
      lastFrame(width, height),
      poseOptimizer(width, height),
      mapOptimizer(width, height),
      poseMapOptimizer(width, height),
      intrinsicPoseMapOptimizer(width, height),
      renderer(width, height)
{
    lastId = 0;
    lastLocalMovement = SE3f();
}

void visualOdometry::init(dataCPU<float> &image, SE3f globalPose)
{
    kframe.init(image, vec2f(0.0, 0.0), globalPose, 1.0);
    kframe.initGeometryVerticallySmooth(cam);
    //kframe.initGeometryRandom(cam);
}

void visualOdometry::init(dataCPU<float> &image, SE3f globalPose, dataCPU<float> &depth, dataCPU<float> &weight)
{
    assert(image.width == depth.width && image.height == depth.height);

    kframe.init(image, vec2f(0.0, 0.0), globalPose, 1.0);
    kframe.initGeometryFromDepth(depth, weight, cam);
}

std::vector<frameCPU> visualOdometry::getFrames()
{
    return goodFrames;
}

keyFrameCPU visualOdometry::getKeyframe()
{
    return kframe;
}

float visualOdometry::meanViewAngle(SE3f pose1, SE3f pose2)
{
    int lvl = 1;

    geometryType scene1 = kframe.getGeometry();
    scene1.transform(pose1);
    scene1.project(cam);

    geometryType scene2 = kframe.getGeometry();
    scene2.transform(pose2);
    scene2.project(cam);

    SE3f relativePose = pose1 * pose2.inverse();

    SE3f frame1PoseInv = relativePose.inverse();
    SE3f frame2PoseInv = SE3f();

    vec3f frame1Translation = frame1PoseInv.translation();
    vec3f frame2Translation = frame2PoseInv.translation();

    std::vector<int> vIds = scene2.getVerticesIds();

    float accAngle = 0;
    int count = 0;
    for (int vId : vIds)
    {
        vertex vert = scene2.getVertex(vId);

        vec3f diff1 = vert.ver - frame1Translation;
        vec3f diff2 = vert.ver - frame2Translation;

        assert(diff1.norm() > 0 && diff2.norm() > 0);

        vec3f diff1Normalized = diff1 / diff1.norm();
        vec3f diff2Normalized = diff2 / diff2.norm();

        float cos_angle = diff1Normalized.dot(diff2Normalized);
        cos_angle = std::clamp(cos_angle, -1.0f, 1.0f);
        float angle = std::acos(cos_angle);

        assert(!std::isnan(angle));

        accAngle += angle;
        count += 1;
    }

    return accAngle / count;
}

float visualOdometry::getViewPercent(frameCPU &frame)
{
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

    dataMipMapCPU<float> depth(frame.getRawImage(0).width, frame.getRawImage(0).height, -1);
    int lvl = 1;
    renderer.renderImageParallel(kframe, frame.getLocalPose(), depth, cam, lvl);
    float pnodata = depth.get(lvl).getPercentNoData();
    return 1.0 - pnodata;
}

int visualOdometry::locAndMap(dataCPU<float> &image)
{
    tic_toc t;
    bool optimize = false;

    std::vector<frameCPU> keyFrames;

    SE3f lastLocalPose = lastFrame.getLocalPose();

    lastFrame.setImage(image, lastId);
    lastFrame.setLocalPose(lastLocalMovement * lastLocalPose);
    // lastFrame.setLocalExp(lastLocalExp);
    lastId++;

    t.tic();
    poseOptimizer.optimize(lastFrame, kframe, cam);
    // sceneOptimizer.optPose(newFrame, kframe, scene);
    std::cout << "estimate pose time: " << t.toc() << std::endl;

    // std::cout << "affine " << newFrame.getAffine()(0) << " " << newFrame.getAffine()(1) << std::endl;
    // std::cout << "estimated pose " << std::endl;
    // std::cout << newFrame.getPose().matrix() << std::endl;

    if (goodFrames.size() == 0)
    {
        goodFrames.push_back(lastFrame);
        keyFrames = goodFrames;
        optimize = true;
    }
    else
    {
        float keyframeViewAngle = meanViewAngle(SE3f(), lastFrame.getLocalPose());

        float lastMinViewAngle = M_PI;
        for(size_t i = 0; i < goodFrames.size(); i++)
        {
            float lastViewAngle = meanViewAngle(goodFrames[i].getLocalPose(), lastFrame.getLocalPose());
            if(lastViewAngle < lastMinViewAngle)
                lastMinViewAngle = lastViewAngle;
        }

        float viewPercent = getViewPercent(lastFrame);

        std::cout << "last min viewAngle " << lastMinViewAngle << std::endl;
        std::cout << "keyframe viewAngle " << keyframeViewAngle << std::endl;
        std::cout << "viewPercent " << viewPercent << std::endl;

        if (lastMinViewAngle > mesh_vo::last_min_angle) //(percentNoData > 0.2) //(viewPercent < 0.8)
        {
            goodFrames.push_back(lastFrame);
            if (goodFrames.size() > mesh_vo::num_frames)
                goodFrames.erase(goodFrames.begin());
        }

        if ((viewPercent < mesh_vo::min_view_perc || keyframeViewAngle > mesh_vo::key_max_angle) && goodFrames.size() > 1)
        {
            // select new keyframe
            // int newKeyframeIndex = 0;
            int newKeyframeIndex = int(goodFrames.size() / 2);
            //int newKeyframeIndex = int(goodFrames.size() - 1);
            frameCPU newKeyframe = goodFrames[newKeyframeIndex];

            // render its idepth
            int lvl = 0;
            dataMipMapCPU<float> depth_buffer(lastFrame.getRawImage(lvl).width, lastFrame.getRawImage(lvl).height, -1);
            dataMipMapCPU<float> weight_buffer(lastFrame.getRawImage(lvl).width, lastFrame.getRawImage(lvl).height, -1);
            renderer.renderDepthParallel(kframe, newKeyframe.getLocalPose(), depth_buffer, cam, lvl);
            renderer.renderWeightParallel(kframe, newKeyframe.getLocalPose(), weight_buffer, cam, lvl);
            renderer.renderInterpolate(depth_buffer.get(lvl));

            // save local frames global params
            std::vector<SE3f> goodFramesGlobalPoses;
            std::vector<vec2f> goodFramesGlobalExp;
            for (int i = 0; i < goodFrames.size(); i++)
            {
                SE3f globalPose = kframe.localPoseToGlobal(goodFrames[i].getLocalPose());
                vec2f globalExp = kframe.localExpToGlobal(goodFrames[i].getLocalExp());

                goodFramesGlobalPoses.push_back(globalPose);
                goodFramesGlobalExp.push_back(globalExp);
            }

            SE3f lastFrameGlobalPose = kframe.localPoseToGlobal(lastFrame.getLocalPose());
            vec2f lastFrameGlobalExp = kframe.localExpToGlobal(lastFrame.getLocalExp());

            SE3f lastGlobalPose = kframe.localPoseToGlobal(lastLocalPose);

            SE3f newKeyframeGlobalPose = kframe.localPoseToGlobal(newKeyframe.getLocalPose());
            vec2f newKeyframeGlobalExp = kframe.localExpToGlobal(newKeyframe.getLocalExp());

            kframe.init(newKeyframe.getRawImage(0), newKeyframeGlobalExp, newKeyframeGlobalPose, kframe.getGlobalScale());
            kframe.initGeometryFromDepth(depth_buffer.get(lvl), weight_buffer.get(lvl), cam);

            vec2f meanStd = kframe.getGeometry().meanStdDepth();
            // vec2f minMax = kframe.getGeometry().minMaxDepthParams();
            // vec2f minMax = kframe.getGeometry().minMaxDepthVertices();
            float scale = 1.0 / meanStd(0);

            kframe.scaleVerticesAndWeights(scale);

            for (int i = 0; i < goodFrames.size(); i++)
            {
                SE3f localPose = kframe.globalPoseToLocal(goodFramesGlobalPoses[i]);
                vec2f localExp = kframe.globalExpToLocal(goodFramesGlobalExp[i]);

                goodFrames[i].setLocalPose(localPose);
                goodFrames[i].setLocalExp(localExp);
            }

            SE3f localPose = kframe.globalPoseToLocal(lastFrameGlobalPose);
            vec2f localExp = kframe.globalExpToLocal(lastFrameGlobalExp);

            lastFrame.setLocalPose(localPose);
            lastFrame.setLocalExp(localExp);

            lastLocalPose = kframe.globalPoseToLocal(lastGlobalPose);

            keyFrames = goodFrames;
            keyFrames.erase(keyFrames.begin() + newKeyframeIndex);

            optimize = true;
        }
    }

    if (optimize)
    {
        t.tic();
        //mapOptimizer.optimize(keyFrames, kframe, cam);
        poseMapOptimizer.optimize(keyFrames, kframe, cam);
        std::cout << "optposemap time: " << t.toc() << std::endl;

        // sync the updated keyframe poses present in lastframes
        for (auto keyframe : keyFrames)
        {
            for (int i = 0; i < (int)goodFrames.size(); i++)
            {
                if (keyframe.getId() == goodFrames[i].getId())
                {
                    goodFrames[i].setLocalPose(keyframe.getLocalPose());
                    goodFrames[i].setLocalExp(keyframe.getLocalExp());
                }
            }
        }
    }

    lastLocalMovement = lastFrame.getLocalPose() * lastLocalPose.inverse();

    return int(optimize);
}

void visualOdometry::intrinsicAndLocAndMap(dataCPU<float> &image)
{
    tic_toc t;
    bool optimize = false;

    std::vector<frameCPU> keyFrames;

    SE3f lastLocalPose = lastFrame.getLocalPose();

    lastFrame.setImage(image, lastId);
    lastFrame.setLocalPose(lastLocalMovement * lastLocalPose);
    // lastFrame.setLocalExp(lastLocalExp);
    lastId++;

    t.tic();
    poseOptimizer.optimize(lastFrame, kframe, cam);
    // sceneOptimizer.optPose(newFrame, kframe, scene);
    std::cout << "estimate pose time: " << t.toc() << std::endl;

    // std::cout << "affine " << newFrame.getAffine()(0) << " " << newFrame.getAffine()(1) << std::endl;
    // std::cout << "estimated pose " << std::endl;
    // std::cout << newFrame.getPose().matrix() << std::endl;

    if (goodFrames.size() == 0)
    {
        goodFrames.push_back(lastFrame);
        keyFrames = goodFrames;
        optimize = true;
    }
    else
    {
        float keyframeViewAngle = meanViewAngle(SE3f(), lastFrame.getLocalPose());

        float lastMinViewAngle = M_PI;
        for(size_t i = 0; i < goodFrames.size(); i++)
        {
            float lastViewAngle = meanViewAngle(goodFrames[i].getLocalPose(), lastFrame.getLocalPose());
            if(lastViewAngle < lastMinViewAngle)
                lastMinViewAngle = lastViewAngle;
        }

        float viewPercent = getViewPercent(lastFrame);

        std::cout << "last viewAngle " << lastMinViewAngle << std::endl;
        std::cout << "keyframe viewAngle " << keyframeViewAngle << std::endl;
        std::cout << "viewPercent " << viewPercent << std::endl;

        if (lastMinViewAngle > mesh_vo::last_min_angle) //(percentNoData > 0.2) //(viewPercent < 0.8)
        {
            goodFrames.push_back(lastFrame);
            if (goodFrames.size() > mesh_vo::num_frames)
                goodFrames.erase(goodFrames.begin());
        }

        if ((viewPercent < mesh_vo::min_view_perc || keyframeViewAngle > mesh_vo::key_max_angle) && goodFrames.size() > 1)
        // if((viewPercent < 0.9 || meanViewAngle > M_PI / 64.0) && lastFrames.size() > 1)
        {
            // select new keyframe
            // int newKeyframeIndex = 0;
            int newKeyframeIndex = int(goodFrames.size() / 2);
            // int newKeyframeIndex = int(lastFrames.size() - 1);
            frameCPU newKeyframe = goodFrames[newKeyframeIndex];

            // render its idepth
            int lvl = 0;
            dataMipMapCPU<float> depth_buffer(lastFrame.getRawImage(lvl).width, lastFrame.getRawImage(lvl).height, -1);
            dataMipMapCPU<float> weight_buffer(lastFrame.getRawImage(lvl).width, lastFrame.getRawImage(lvl).height, -1);
            renderer.renderDepthParallel(kframe, newKeyframe.getLocalPose(), depth_buffer, cam, lvl);
            renderer.renderWeightParallel(kframe, newKeyframe.getLocalPose(), weight_buffer, cam, lvl);
            renderer.renderInterpolate(depth_buffer.get(lvl));

            // save local frames global params
            std::vector<SE3f> goodFramesGlobalPoses;
            std::vector<vec2f> goodFramesGlobalExp;
            for (int i = 0; i < goodFrames.size(); i++)
            {
                SE3f globalPose = kframe.localPoseToGlobal(goodFrames[i].getLocalPose());
                vec2f globalExp = kframe.localExpToGlobal(goodFrames[i].getLocalExp());

                goodFramesGlobalPoses.push_back(globalPose);
                goodFramesGlobalExp.push_back(globalExp);
            }

            SE3f lastFrameGlobalPose = kframe.localPoseToGlobal(lastFrame.getLocalPose());
            vec2f lastFrameGlobalExp = kframe.localExpToGlobal(lastFrame.getLocalExp());

            SE3f lastGlobalPose = kframe.localPoseToGlobal(lastLocalPose);

            SE3f newKeyframeGlobalPose = kframe.localPoseToGlobal(newKeyframe.getLocalPose());
            vec2f newKeyframeGlobalExp = kframe.localExpToGlobal(newKeyframe.getLocalExp());

            kframe.init(newKeyframe.getRawImage(0), newKeyframeGlobalExp, newKeyframeGlobalPose, kframe.getGlobalScale());
            kframe.initGeometryFromDepth(depth_buffer.get(lvl), weight_buffer.get(lvl), cam);

            vec2f meanStd = kframe.getGeometry().meanStdDepth();
            // vec2f minMax = kframe.getGeometry().minMaxDepthParams();
            // vec2f minMax = kframe.getGeometry().minMaxDepthVertices();
            float scale = 1.0 / meanStd(0);

            kframe.scaleVerticesAndWeights(scale);

            for (int i = 0; i < goodFrames.size(); i++)
            {
                SE3f localPose = kframe.globalPoseToLocal(goodFramesGlobalPoses[i]);
                vec2f localExp = kframe.globalExpToLocal(goodFramesGlobalExp[i]);

                goodFrames[i].setLocalPose(localPose);
                goodFrames[i].setLocalExp(localExp);
            }

            SE3f localPose = kframe.globalPoseToLocal(lastFrameGlobalPose);
            vec2f localExp = kframe.globalExpToLocal(lastFrameGlobalExp);

            lastFrame.setLocalPose(localPose);
            lastFrame.setLocalExp(localExp);

            lastLocalPose = kframe.globalPoseToLocal(lastGlobalPose);

            keyFrames = goodFrames;
            keyFrames.erase(keyFrames.begin() + newKeyframeIndex);

            optimize = true;
        }
    }

    if (optimize)
    {
        t.tic();
        intrinsicPoseMapOptimizer.optimize(keyFrames, kframe, cam);
        std::cout << "optposemap time: " << t.toc() << std::endl;

        // sync the updated keyframe poses present in lastframes
        for (auto keyframe : keyFrames)
        {
            for (int i = 0; i < (int)goodFrames.size(); i++)
            {
                if (keyframe.getId() == goodFrames[i].getId())
                {
                    goodFrames[i].setLocalPose(keyframe.getLocalPose());
                    goodFrames[i].setLocalExp(keyframe.getLocalExp());
                }
            }
        }
    }

    lastLocalMovement = lastFrame.getLocalPose() * lastLocalPose.inverse();
}

void visualOdometry::lightaffine(dataCPU<float> &image, SE3f globalPose)
{
    tic_toc t;

    lastFrame.setImage(image, lastId);
    lastFrame.setLocalPose(globalPose * kframe.getGlobalPose().inverse());
    // lastFrame.setLocalExp(lastLocalExp);
    lastId++;

    t.tic();
    // sceneOptimizer.optLightAffine(newFrame, kframe, scene);
    std::cout << "optmap time " << t.toc() << std::endl;
}

void visualOdometry::localization(dataCPU<float> &image)
{
    tic_toc t;

    SE3f lastLocalPose = lastFrame.getLocalPose();

    lastFrame.setImage(image, lastId);
    lastFrame.setLocalPose(lastLocalMovement * lastLocalPose);
    // lastFrame.setLocalExp(lastLocalExp);
    lastId++;

    t.tic();
    poseOptimizer.optimize(lastFrame, kframe, cam);
    std::cout << "estimated pose " << t.toc() << std::endl;
    std::cout << lastFrame.getLocalPose().matrix() << std::endl;
    std::cout << lastFrame.getLocalExp()(0) << " " << lastFrame.getLocalExp()(1) << std::endl;

    lastLocalMovement = lastFrame.getLocalPose() * lastLocalPose.inverse();
}

void visualOdometry::mapping(dataCPU<float> &image, SE3f globalPose, vec2f exp)
{
    tic_toc t;
    bool optimize = false;

    std::vector<frameCPU> keyFrames;

    SE3f localPose = globalPose * kframe.getGlobalPose().inverse();
    localPose.translation() *= kframe.getGlobalScale();

    lastFrame.setImage(image, lastId);
    lastFrame.setLocalPose(localPose);
    lastFrame.setLocalExp(exp);
    lastId++;

    if (goodFrames.size() == 0)
    {
        goodFrames.push_back(lastFrame);
        keyFrames = goodFrames;
        optimize = true;
    }
    else
    {
        float keyframeViewAngle = meanViewAngle(SE3f(), lastFrame.getLocalPose());

        float lastMinViewAngle = M_PI;
        for(size_t i = 0; i < goodFrames.size(); i++)
        {
            float lastViewAngle = meanViewAngle(goodFrames[i].getLocalPose(), lastFrame.getLocalPose());
            if(lastViewAngle < lastMinViewAngle)
                lastMinViewAngle = lastViewAngle;
        }

        float viewPercent = getViewPercent(lastFrame);

        std::cout << "last viewAngle " << lastMinViewAngle << std::endl;
        std::cout << "keyframe viewAngle " << keyframeViewAngle << std::endl;
        std::cout << "viewPercent " << viewPercent << std::endl;

        if (lastMinViewAngle > mesh_vo::last_min_angle) //(percentNoData > 0.2) //(viewPercent < 0.8)
        {
            goodFrames.push_back(lastFrame);
            if (goodFrames.size() > mesh_vo::num_frames)
                goodFrames.erase(goodFrames.begin());
        }

        if ((viewPercent < mesh_vo::min_view_perc || keyframeViewAngle > mesh_vo::key_max_angle) && goodFrames.size() > 1)
        // if((viewPercent < 0.9 || meanViewAngle > M_PI / 64.0) && lastFrames.size() > 1)
        {
            // select new keyframe
            // int newKeyframeIndex = 0;
            int newKeyframeIndex = int(goodFrames.size() / 2);
            // int newKeyframeIndex = int(lastFrames.size() - 1);
            frameCPU newKeyframe = goodFrames[newKeyframeIndex];

            // render its idepth
            int lvl = 0;
            dataMipMapCPU<float> depth_buffer(lastFrame.getRawImage(lvl).width, lastFrame.getRawImage(lvl).height, -1);
            dataMipMapCPU<float> weight_buffer(lastFrame.getRawImage(lvl).width, lastFrame.getRawImage(lvl).height, -1);
            renderer.renderDepthParallel(kframe, newKeyframe.getLocalPose(), depth_buffer, cam, lvl);
            // renderer.renderWeightParallel(kframe, newKeyframe.getLocalPose(), weight_buffer, cam, lvl);
            renderer.renderInterpolate(depth_buffer.get(lvl));

            // save local frames global params
            std::vector<SE3f> goodFramesGlobalPoses;
            std::vector<vec2f> goodFramesGlobalExp;
            for (int i = 0; i < goodFrames.size(); i++)
            {
                SE3f globalPose = kframe.localPoseToGlobal(goodFrames[i].getLocalPose());
                vec2f globalExp = kframe.localExpToGlobal(goodFrames[i].getLocalExp());

                goodFramesGlobalPoses.push_back(globalPose);
                goodFramesGlobalExp.push_back(globalExp);
            }

            SE3f lastFrameGlobalPose = kframe.localPoseToGlobal(lastFrame.getLocalPose());
            vec2f lastFrameGlobalExp = kframe.localExpToGlobal(lastFrame.getLocalExp());

            SE3f newKeyframeGlobalPose = kframe.localPoseToGlobal(newKeyframe.getLocalPose());
            vec2f newKeyframeGlobalExp = kframe.localExpToGlobal(newKeyframe.getLocalExp());

            kframe.init(newKeyframe.getRawImage(0), newKeyframeGlobalExp, newKeyframeGlobalPose, kframe.getGlobalScale());
            kframe.initGeometryFromDepth(depth_buffer.get(lvl), weight_buffer.get(lvl), cam);

            vec2f meanStd = kframe.getGeometry().meanStdDepth();
            // vec2f minMax = kframe.getGeometry().minMaxDepthParams();
            // vec2f minMax = kframe.getGeometry().minMaxDepthVertices();
            float scale = 1.0 / meanStd(0);

            kframe.scaleVerticesAndWeights(scale);

            for (int i = 0; i < goodFrames.size(); i++)
            {
                SE3f localPose = kframe.globalPoseToLocal(goodFramesGlobalPoses[i]);
                vec2f localExp = kframe.globalExpToLocal(goodFramesGlobalExp[i]);

                goodFrames[i].setLocalPose(localPose);
                goodFrames[i].setLocalExp(localExp);
            }

            SE3f localPose = kframe.globalPoseToLocal(lastFrameGlobalPose);
            vec2f localExp = kframe.globalExpToLocal(lastFrameGlobalExp);

            lastFrame.setLocalPose(localPose);
            lastFrame.setLocalExp(localExp);

            keyFrames = goodFrames;
            keyFrames.erase(keyFrames.begin() + newKeyframeIndex);

            optimize = true;
        }
    }

    if (optimize)
    {
        t.tic();
        mapOptimizer.optimize(keyFrames, kframe, cam);
        std::cout << "optmap time: " << t.toc() << std::endl;
    }
}
