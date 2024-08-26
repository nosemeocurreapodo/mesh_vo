#pragma once

#include <Eigen/Core>
#include <thread>

#include "common/camera.h"
#include "common/types.h"
#include "common/HGEigenDense.h"
#include "common/HGEigenSparse.h"
// #include "common/HGMapped.h"
#include "common/Error.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/reduceCPU.h"
#include "cpu/SceneBase.h"
#include "cpu/ScenePatches.h"
#include "cpu/SceneMesh.h"
// #include "cpu/SceneSurfels.h"
//  #include "cpu/SceneMeshSmooth.h"
#include "cpu/OpenCVDebug.h"
#include "params.h"

class meshOptimizerCPU
{
public:
    meshOptimizerCPU(camera &cam);

    void initKeyframe(frameCPU &frame, int lvl);
    void initKeyframe(frameCPU &frame, dataCPU<float> &idepth, dataCPU<float> &ivar, int lvl);

    void optPose(frameCPU &frame);
    void optMap(std::vector<frameCPU> &frames);
    void optPoseMap(std::vector<frameCPU> &frame);

    void setMeshRegu(float mr)
    {
        meshRegularization = mr;
    }

    float meanViewAngle(frameCPU *kframe, frameCPU *frame)
    {
        int lvl = 1;

        scene->transform(kframe->pose);
        scene->project(cam[lvl]);

        std::vector<int> sIds = scene->getShapesIds();

        Sophus::SE3f fromkframeToframe = frame->pose * kframe->pose.inverse();
        Sophus::SE3f fromframeTokframe = fromkframeToframe.inverse();

        float accAngle = 0;
        int count = 0;
        for (auto sId : sIds)
        {
            auto shape = scene->getShape(cam[lvl], sId);
            vec2<float> centerPix = shape->getCenterPix();
            if (!cam[lvl].isPixVisible(centerPix))
                continue;

            vec3<float> centerRay = cam[lvl].pixToRay(centerPix);

            shape->prepareForPix(centerPix);
            float centerDepth = shape->getDepth();

            vec3<float> centerPoint = centerRay * centerDepth;

            Eigen::Vector3f lastPoint_e = fromkframeToframe * Eigen::Vector3f(centerPoint(0), centerPoint(1), centerPoint(2));
            Eigen::Vector3f lastRay_e = lastPoint_e / lastPoint_e(2);
            vec3<float> lastRay(lastRay_e(0), lastRay_e(1), lastRay_e(2));
            vec2<float> lastPix = cam[lvl].rayToPix(lastRay);
            if (!cam[lvl].isPixVisible(lastPix))
                continue;

            Eigen::Vector3f lastRotatedRay_e = fromframeTokframe.inverse().rotationMatrix() * lastRay_e;
            vec3<float> lastRotatedRay = vec3<float>(lastRotatedRay_e(0), lastRotatedRay_e(1), lastRotatedRay_e(2));
            vec3<float> centerRayNormalized = centerRay / centerRay.norm();
            vec3<float> lastRoratedRayNormalized = lastRotatedRay / lastRotatedRay.norm();
            float cos_angle = centerRayNormalized.dot(lastRoratedRayNormalized);
            float angle = std::acos(cos_angle);

            accAngle += std::fabs(angle);
            count += 1;
        }

        return accAngle / count;
    }

    float checkInfo(frameCPU &frame)
    {
        int lvl = 2;
        scene->transform(frame.pose);
        scene->project(cam[lvl]);
        kscene.project(cam[lvl]);
        HGEigenSparse hg = computeHGMap2(scene.get(), &frame, lvl);
        std::map<int, int> ids = hg.getObservedParamIds();
        Eigen::SparseMatrix<float> H = hg.getH(ids);
        Eigen::VectorXf G = hg.getG(ids);

        // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > solver;
        // Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
        Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
        // Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;
        // Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;

        solver.compute(H);
        // solver.analyzePattern(H_lambda);
        // solver.factorize(H_lambda);

        if (solver.info() != Eigen::Success)
        {
            return 0.0;
        }
        // std::cout << solver.lastErrorMessage() << std::endl;
        Eigen::VectorXf inc = solver.solve(G);
        // inc_depth = -acc_H_depth_lambda.llt().solve(acc_J_depth);
        // inc_depth = - acc_H_depth_lambda.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(acc_J_depth);
        // inc_depth = -acc_H_depth_lambda.colPivHouseholderQr().solve(acc_J_depth);

        if (solver.info() != Eigen::Success)
        {
            return 0.0;
        }

        float relative_error = (H * inc - G).norm() / G.size(); // norm() is L2 norm

        return relative_error;
    }

    dataCPU<float> getIdepth(Sophus::SE3f &pose, int lvl)
    {
        idepth_buffer.set(idepth_buffer.nodata, lvl);

        scene->transform(pose);
        scene->project(cam[lvl]);
        kscene.project(cam[lvl]);

        renderer.renderIdepthParallel(scene.get(), cam[lvl], &idepth_buffer, lvl);
        // renderer.renderInterpolate(cam[lvl], &idepth_buffer, lvl);
        return idepth_buffer;
    }

    /*
    dataCPU<float> getImage(frameCPU &frame, Sophus::SE3f &pose, int lvl)
    {
        image_buffer.set(image_buffer.nodata, lvl);
        renderer.renderImage(cam[lvl], frame, pose, image_buffer, lvl);
        return image_buffer;
    }
    */

    void plotDebug(frameCPU &frame)
    {
        idepth_buffer.set(idepth_buffer.nodata, 1);
        image_buffer.set(image_buffer.nodata, 1);
        error_buffer.set(error_buffer.nodata, 1);
        debug.set(debug.nodata, 0);

        scene->transform(frame.pose);
        scene->project(cam[1]);
        kscene.project(cam[1]);
        // renderer.renderIdepth(cam[1], frame.pose, idepth_buffer, 1);
        renderer.renderIdepthParallel(scene.get(), cam[1], &idepth_buffer, 1);
        renderer.renderImageParallel(&kscene, &kframe, scene.get(), cam[1], &image_buffer, 1);

        error_buffer = frame.image.sub(image_buffer, 1);
        // renderer.renderIdepthLineSearch(&kframe, &frame, cam[1], &error_buffer, 1);
        // idepth_buffer = renderer.getzbuffer();

        show(frame.image, "frame image", 1);
        show(kframe.image, "keyframe image", 1);
        show(error_buffer, "lastFrame error", 1);
        show(idepth_buffer, "lastFrame idepth", 1);
        show(image_buffer, "lastFrame scene", 1);

        scene->project(cam[0]);
        kscene.project(cam[0]);
        renderer.renderDebugParallel(scene.get(), &frame, cam[0], &debug, 0);
        show(debug, "frame debug", 0);
    }

    float getViewPercent(frameCPU &frame)
    {
        int lvl = 1;
        scene->transform(frame.pose);
        scene->project(cam[lvl]);

        std::vector<int> shapeIds = scene->getShapesIds();

        int numVisible = 0;
        for(auto shapeId : shapeIds)
        {
            auto shape = scene->getShape(cam[lvl], shapeId);
            auto pix = shape->getCenterPix();
            if(cam[lvl].isPixVisible(pix))
                numVisible++;
        }

        return float(numVisible)/shapeIds.size();
    }

    void changeKeyframe(frameCPU &frame)
    {
        int lvl = 1;

        idepth_buffer.set(idepth_buffer.nodata, lvl);
        ivar_buffer.set(ivar_buffer.nodata, lvl);

        scene->transform(frame.pose);
        scene->project(cam[lvl]);
        renderer.renderIdepthParallel(scene.get(), cam[lvl], &idepth_buffer, lvl);
        renderer.renderWeightParallel(scene.get(), cam[lvl], &ivar_buffer, lvl);
        //renderer.renderRandom(cam[lvl], &idepth, lvl);
        renderer.renderInterpolate(cam[lvl], &idepth_buffer, lvl);
        renderer.renderInterpolate(cam[lvl], &ivar_buffer, lvl);

        initKeyframe(frame, idepth_buffer, ivar_buffer, lvl);

        //kscene.transform(frame.pose);
        //kscene.project(cam[lvl]);
        //kscene.complete(frame, cam[lvl], idepth_buffer, lvl);
        //scene = kscene.clone();

        /*
        //method 2
        //build frame mesh
        //remove ocluded
        //devide big triangles
        //complete with random points
        MeshCPU frameMesh = buildFrameMesh(frame, lvl);
        keyframeMesh = frameMesh.getCopy();
        keyframe = frame;
        */
    }

    //ScenePatches kscene;
    // SceneSurfels kscene;
    SceneMesh kscene;
    frameCPU kframe;

    camera cam[MAX_LEVELS];

private:
    Error computeError(SceneBase *scene, frameCPU *frame, int lvl);
    // Error computeError(dataCPU<float> &kfIdepth, frameCPU &kframe, frameCPU &frame, int lvl);

    HGEigenDense computeHGPose(SceneBase *scene, frameCPU *frame, int lvl);
    // HGPose computeHGPose(dataCPU<float> &kfIdepth, frameCPU &kframe, frameCPU &frame, int lvl);

    HGMapped computeHGMap(SceneBase *scene, frameCPU *frame, int lvl);
    HGEigenSparse computeHGMap2(SceneBase *scene, frameCPU *frame, int lvl);

    HGMapped computeHGPoseMap(SceneBase *scene, frameCPU *frame, int frameIndex, int numFrames, int lvl);
    HGEigenSparse computeHGPoseMap2(SceneBase *scene, frameCPU *frame, int frameIndex, int numFrames, int lvl);

    std::unique_ptr<SceneBase> scene;
    // params
    bool multiThreading;
    float meshRegularization;
    float meshInitial;

    dataCPU<float> image_buffer;
    dataCPU<float> idepth_buffer;
    dataCPU<float> ivar_buffer;
    dataCPU<float> error_buffer;

    dataCPU<vec6<float>> jpose_buffer;

    //dataCPU<vec1<float>> jmap_buffer;
    //dataCPU<vec1<int>> pId_buffer;

    dataCPU<vec3<float>> jmap_buffer;
    dataCPU<vec3<int>> pId_buffer;

    // debug
    dataCPU<float> debug;
    dataCPU<float> idepthVar;

    renderCPU renderer;
    reduceCPU reducer;
};
