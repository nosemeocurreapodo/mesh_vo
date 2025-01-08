#pragma once

#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>

#include "sophus/se3.hpp"

#include "cpu/frameCPU.h"
//#include "cpu/ScenePatches.h"
#include "cpu/SceneMesh.h"
#include "cpu/Shapes.h"
//#include "cpu/SceneSurfels.h"
//#include "cpu/SceneMeshSmooth.h"

#include "optimizers/sceneOptimizerCPU.h"

template <typename sceneType>
class visualOdometry
{
public:
    visualOdometry(camera &_cam);

    void init(dataCPU<float> &image, Sophus::SE3f pose = Sophus::SE3f());
    void init(frameCPU &frame);
    void init(frameCPU &frame, dataCPU<float> &idepth);
    void init(frameCPU &frame, std::vector<vec2<float>> &pixels, std::vector<float> &idepths);

    void locAndMap(dataCPU<float> &image);
    void lightaffine(dataCPU<float> &image, Sophus::SE3f globalPose);
    void localization(dataCPU<float> &image);
    void mapping(dataCPU<float> &image, Sophus::SE3f globalPose, vec2<float> affine);

    Sophus::SE3f lastPose;
    vec2<float> lastAffine;

private:

    dataCPU<float> getIdepth(Sophus::SE3f pose, int lvl)
    {
        dataCPU<float> buffer(cam[lvl].width, cam[lvl].height, -1);

        renderer.renderIdepthParallel(kscene, pose, cam[lvl], buffer);
        // renderer.renderInterpolate(cam[lvl], &idepth_buffer, lvl);
        return buffer;
    }

    float meanViewAngle(Sophus::SE3f pose1, Sophus::SE3f pose2)
    {
        int lvl = 1;

        kscene.transform(cam[lvl], Sophus::SE3f());

        sceneType scene1 = kscene;
        scene1.transform(cam[lvl], pose1);

        sceneType scene2 = kscene;
        scene2.transform(cam[lvl], pose2);

        Sophus::SE3f frame1PoseInv = pose1.inverse();
        Sophus::SE3f frame2PoseInv = pose2.inverse();

        Eigen::Vector3f frame1Tra = frame1PoseInv.translation();
        Eigen::Vector3f frame2Tra = frame2PoseInv.translation();

        vec3<float> frame1Translation(frame1Tra(0), frame1Tra(1), frame1Tra(2));
        vec3<float> frame2Translation(frame2Tra(0), frame2Tra(1), frame2Tra(2));

        std::vector<int> sIds = kscene.getShapesIds();

        float accAngle = 0;
        int count = 0;
        for (auto sId : sIds)
        {
            auto shape = kscene.getShape(sId);
            vec2<float> centerPix = shape.getCenterPix();
            float centerDepth = shape.getDepth(centerPix);

            auto shape1 = scene1.getShape(sId);
            auto shape2 = scene2.getShape(sId);

            vec2<float> pix1 = shape1.getCenterPix();
            vec2<float> pix2 = shape2.getCenterPix();

            if (!cam[lvl].isPixVisible(pix1) || !cam[lvl].isPixVisible(pix2))
                continue;

            vec3<float> centerRay = cam[lvl].pixToRay(centerPix);
            vec3<float> centerPoint = centerRay * centerDepth;

            vec3<float> diff1 = frame1Translation - centerPoint;
            vec3<float> diff2 = frame2Translation - centerPoint;
            vec3<float> diff1Normalized = diff1 / diff1.norm();
            vec3<float> diff2Normalized = diff2 / diff2.norm();

            float cos_angle = diff1Normalized.dot(diff2Normalized);
            float angle = std::acos(cos_angle);

            accAngle += std::fabs(angle);
            count += 1;
        }

        return accAngle / count;
    }

    float getViewPercent(frameCPU &frame)
    {
        int lvl = 1;
        sceneType scene = kscene;
        scene.transform(cam[lvl], frame.getPose());
        std::vector<int> shapeIds = scene.getShapesIds();

        int numVisible = 0;
        for (auto shapeId : shapeIds)
        {
            auto shape = scene.getShape(shapeId);
            vec2<float> pix = shape.getCenterPix();
            float depth = shape.getDepth(pix);
            if (depth <= 0.0)
                continue;
            if (cam[lvl].isPixVisible(pix))
                numVisible++;
        }

        return float(numVisible) / shapeIds.size();
    }

    float checkInfo(frameCPU &frame)
    {
        /*
        int lvl = 2;
        DenseLinearProblem hg = computeHGMap2(&frame, lvl);
        std::map<int, int> ids = hg.getObservedParamIds();
        Eigen::SparseMatrix<float> H = hg.getHSparse(ids);
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
        */
        return 1.0;
    }

    void plotDebug(frameCPU &frame, std::vector<frameCPU> frames = std::vector<frameCPU>())
    {
        int lvl = 1;

        dataCPU<float> idepth_buffer(cam[lvl].width, cam[lvl].height, -1);
        dataCPU<float> image_buffer(cam[lvl].width, cam[lvl].height, -1);
        dataCPU<float> error_buffer(cam[lvl].width, cam[lvl].height, -1);
        // ivar_buffer.set(ivar_buffer.nodata, 1);
        // debug.set(debug.nodata, 0);

        // renderer.renderIdepth(cam[1], frame.pose, idepth_buffer, 1);
        renderer.renderIdepthParallel(kscene, frame.getPose(), cam[lvl], idepth_buffer);
        //renderer.renderWeightParallel(kscene, frame.getPose(), cam, ivar_buffer);
        renderer.renderResidualParallel(kscene, kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getPose(), cam[lvl], error_buffer);

        show(frame.getRawImage(lvl), "frame image", false);
        show(kframe.getRawImage(lvl), "keyframe image", false);
        show(error_buffer, "frame error", false);
        show(idepth_buffer, "frame idepth", true);
        // show(ivar_buffer, "ivar", true, false, 1);

        // show(frame.getdIdpixImage(), "frame dx image", false, false, 0, 1);
        // show(jpose_buffer, "jpose image", false, false, 0, 1);
        // show(jmap_buffer, "jmap image", false, false, 0, 1);

        if (frames.size() > 0)
        {
            dataCPU<float> frames_buffer(cam[lvl].width * frames.size(), cam[lvl].height, -1);
            dataCPU<float> residual_buffer(cam[lvl].width * frames.size(), cam[lvl].height, -1);
            for (int i = 0; i < (int)frames.size(); i++)
            {
                error_buffer.set(error_buffer.nodata);
                renderer.renderResidualParallel(kscene, kframe.getRawImage(lvl), frames[i].getRawImage(lvl), frames[i].getPose(), cam[lvl], error_buffer);

                for (int y = 0; y < cam[1].height; y++)
                {
                    for (int x = 0; x < cam[1].width; x++)
                    {
                        float pix_val = frames[i].getRawImage(lvl).get(y, x);
                        // float res_val = frames[i].getResidualImage().get(y, x, 1);
                        float res_val = error_buffer.get(y, x);

                        frames_buffer.set(pix_val, y, x + i * cam[lvl].width);
                        residual_buffer.set(res_val, y, x + i * cam[lvl].width);
                    }
                }
            }

            show(frames_buffer, "frames", false);
            show(residual_buffer, "residuals", false);
        }

        // renderer.renderDebugParallel(&kscene, &kimage, Sophus::SE3f(), cam[0], &debug, 0);
        // show(debug, "frame debug", false, false, 0);

        // idepth_buffer.set(idepth_buffer.nodata, 1);
        //  renderer.renderIdepth(cam[1], frame.pose, idepth_buffer, 1);
        // renderer.renderIdepthParallel(&kscene, Sophus::SE3f(), cam[1], &idepth_buffer, 1);
        // show(idepth_buffer, "keyframe idepth", true, false, 1);
    }

    //void checkFrameAndAddToList(frameCPU &frame)
    //{
    //  dataCPU<float> kIdepth = meshOptimizer.getIdepth(meshOptimizer.kframe.getPose(), 1);
    //}

    int lastId;
    cameraMipMap cam;
    Sophus::SE3f lastMovement;
    std::vector<frameCPU> lastFrames;
    std::vector<frameCPU> keyFrames;

    SceneMesh kscene;
    frameCPU kframe;

    sceneOptimizerCPU<SceneMesh, vec3<float>, vec3<int>> sceneOptimizer;
    renderCPU<SceneMesh> renderer;
};
