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

    void init(dataCPU<float> &image, Sophus::SE3f pose);
    void init(dataCPU<float> &image, dataCPU<float> &idepth, Sophus::SE3f pose);

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

        renderer.renderIdepthParallel(scene, pose, cam[lvl], buffer);
        // renderer.renderInterpolate(cam[lvl], &idepth_buffer, lvl);
        return buffer;
    }

    float meanViewAngle(Sophus::SE3f pose1, Sophus::SE3f pose2)
    {
        int lvl = 1;

        sceneType scene1 = scene;
        scene1.transform(cam[lvl], pose1);

        sceneType scene2 = scene;
        scene2.transform(cam[lvl], pose2);

        Sophus::SE3f relativePose = pose1*pose2.inverse();

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

    float getViewPercent(frameCPU &frame)
    {
        int lvl = 1;
        sceneType scene1 = scene;
        scene1.transform(cam[lvl], frame.getPose());
        std::vector<int> shapeIds = scene1.getShapesIds();

        int numVisible = 0;
        for (auto shapeId : shapeIds)
        {
            auto shape = scene1.getShape(shapeId);
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

    //void checkFrameAndAddToList(frameCPU &frame)
    //{
    //  dataCPU<float> kIdepth = meshOptimizer.getIdepth(meshOptimizer.kframe.getPose(), 1);
    //}

    int lastId;
    cameraMipMap cam;
    Sophus::SE3f lastMovement;
    std::vector<frameCPU> lastFrames;
    std::vector<frameCPU> keyFrames;

    SceneMesh scene;
    frameCPU kframe;

    sceneOptimizerCPU<SceneMesh, vec3<float>, vec3<int>> sceneOptimizer;
    renderCPU<sceneType> renderer;
};
