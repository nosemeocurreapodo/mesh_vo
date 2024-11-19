#pragma once

#include <Eigen/Core>
#include <Eigen/CholmodSupport>
// #include <Eigen/SPQRSupport>
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
//#include "cpu/SceneSurfels.h"
//#include "cpu/SceneMeshSmooth.h"
#include "cpu/OpenCVDebug.h"
#include "params.h"

template <typename sceneType, typename shapeType, typename jmapType, typename idsType>
class meshOptimizerCPU
{
public:

    meshOptimizerCPU(camera &cam);
    void initKeyframe(frameCPU &frame, int lvl);
    void initKeyframe(frameCPU &frame, dataCPU<float> &idepth, dataCPU<float> &ivar, int lvl);
    void initKeyframe(frameCPU &frame, std::vector<vec2<float>> &texcoords, std::vector<float> &idepths, int lvl);
    void normalizeDepth();

    void optLightAffine(frameCPU &frame);
    void optPose(frameCPU &frame);
    void optMap(std::vector<frameCPU> &frames, dataCPU<float> &mask);
    void optPoseMap(std::vector<frameCPU> &frame);

    void setMeshRegu(float mr)
    {
        meshRegularization = mr;
    }

    float meanViewAngle(frameCPU *frame1, frameCPU *frame2)
    {
        int lvl = 1;

        kscene.project(cam[lvl]);

        sceneType scene1 = kscene.clone();
        scene1.transform(frame1->getPose());
        scene1.project(cam[lvl]);

        sceneType scene2 = kscene.clone();
        scene2.transform(frame2->getPose());
        scene2.project(cam[lvl]);

        Sophus::SE3f frame1PoseInv = frame1->getPose().inverse();
        Sophus::SE3f frame2PoseInv = frame2->getPose().inverse();

        Eigen::Vector3f frame1Tra = frame1PoseInv.translation();
        Eigen::Vector3f frame2Tra = frame2PoseInv.translation();

        vec3<float> frame1Translation(frame1Tra(0), frame1Tra(1), frame1Tra(2));
        vec3<float> frame2Translation(frame2Tra(0), frame2Tra(1), frame2Tra(2));

        std::vector<int> sIds = kscene.getShapesIds();

        float accAngle = 0;
        int count = 0;
        for (auto sId : sIds)
        {
            shapeType shape = kscene.getShape(sId);
            vec2<float> centerPix = shape.getCenterPix();
            float centerDepth = shape.getDepth(centerPix);

            shapeType shape1 = scene1.getShape(sId);
            shapeType shape2 = scene2.getShape(sId);

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
        sceneType scene = kscene.clone();
        scene.transform(frame.getPose());
        scene.project(cam[lvl]);
        std::vector<int> shapeIds = scene.getShapesIds();

        int numVisible = 0;
        for (auto shapeId : shapeIds)
        {
            shapeType shape = scene.getShape(shapeId);
            vec2<float> pix = shape.getCenterPix();
            float depth = shape.getDepth(pix);
            if(depth <= 0.0)
                continue;
            if (cam[lvl].isPixVisible(pix))
                numVisible++;
        }

        return float(numVisible) / shapeIds.size();
    }

    float checkInfo(frameCPU &frame)
    {
        int lvl = 2;
        dataCPU<float> mask(cam[0].width, cam[0].height, -1);
        HGEigenSparse hg = computeHGMap2(&frame, &mask, lvl);
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

    dataCPU<float> getIdepth(Sophus::SE3f pose, int lvl)
    {
        idepth_buffer.set(idepth_buffer.nodata, lvl);

        renderer.renderIdepthParallel(&kscene, pose, cam[lvl], &idepth_buffer, lvl);
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

    void plotDebug(frameCPU &frame, std::vector<frameCPU> frames = std::vector<frameCPU>())
    {
        idepth_buffer.set(idepth_buffer.nodata, 1);
        image_buffer.set(image_buffer.nodata, 1);
        error_buffer.set(error_buffer.nodata, 1);
        ivar_buffer.set(ivar_buffer.nodata, 1);
        debug.set(debug.nodata, 0);

        // renderer.renderIdepth(cam[1], frame.pose, idepth_buffer, 1);
        renderer.renderIdepthParallel(&kscene, frame.getPose(), cam[1], &idepth_buffer, 1);
        renderer.renderWeightParallel(&kscene, frame.getPose(), cam[1], &ivar_buffer, 1);
        renderer.renderResidualParallel(&kscene, &kimage, &frame, cam[1], &error_buffer, 1);

        show(frame.getRawImage(), "frame image", false, false, 1);
        show(kimage, "keyframe image", false, false, 1);
        show(error_buffer, "frame error", false, true, 1);
        show(idepth_buffer, "frame idepth", true, true, 1);
        show(ivar_buffer, "ivar", true, false, 1);

        show(frame.getdIdpixImage(), "frame dx image", false, true, 1);
        show(jpose_buffer, "jpose image", false, true, 1);
        show(jmap_buffer, "jmap image", false, true, 1);

        if (frames.size() > 0)
        {
            dataCPU<float> frames_buffer(cam[0].width * frames.size(), cam[0].height, -1);
            dataCPU<float> residual_buffer(cam[0].width * frames.size(), cam[0].height, -1);
            for (int i = 0; i < frames.size(); i++)
            {
                for (int y = 0; y < cam[1].height; y++)
                {
                    for (int x = 0; x < cam[1].width; x++)
                    {
                        float pix_val = frames[i].getRawImage().get(y, x, 1);
                        float res_val = frames[i].getResidualImage().get(y, x, 1);
                        frames_buffer.set(pix_val, y, x + i * cam[1].width, 1);
                        residual_buffer.set(res_val, y, x + i * cam[1].width, 1);
                    }
                }
            }

            show(frames_buffer, "frames", false, false, 1);
            show(residual_buffer, "residuals", false, false, 1);
        }

        renderer.renderDebugParallel(&kscene, &kimage, Sophus::SE3f(), cam[0], &debug, 0);
        show(debug, "frame debug", false, false, 0);

        idepth_buffer.set(idepth_buffer.nodata, 1);
        // renderer.renderIdepth(cam[1], frame.pose, idepth_buffer, 1);
        renderer.renderIdepthParallel(&kscene, Sophus::SE3f(), cam[1], &idepth_buffer, 1);
        show(idepth_buffer, "keyframe idepth", true, false, 1);
    }

    void changeKeyframe(frameCPU &frame)
    {
        int lvl = 1;

        idepth_buffer.set(idepth_buffer.nodata, lvl);
        ivar_buffer.set(ivar_buffer.nodata, lvl);

        renderer.renderIdepthParallel(&kscene, frame.getPose(), cam[lvl], &idepth_buffer, lvl);
        renderer.renderWeightParallel(&kscene, frame.getPose(), cam[lvl], &ivar_buffer, lvl);
        // renderer.renderRandom(cam[lvl], &idepth, lvl);
        renderer.renderInterpolate(cam[lvl], &idepth_buffer, lvl);
        renderer.renderInterpolate(cam[lvl], &ivar_buffer, lvl);

        initKeyframe(frame, idepth_buffer, ivar_buffer, lvl);

        // kscene.transform(frame.pose);
        // kscene.project(cam[lvl]);
        // kscene.complete(frame, cam[lvl], idepth_buffer, lvl);
        // scene = kscene.clone();

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

    sceneType kscene;
    //frameCPU kframe;
    dataCPU<float> kimage;
    Sophus::SE3f kpose;
    vec2<float> kDepthAffine;

    camera cam[MAX_LEVELS];

private:
    Error computeError(frameCPU *frame, int lvl, bool useWeights = false);
    // Error computeError(dataCPU<float> &kfIdepth, frameCPU &kframe, frameCPU &frame, int lvl);

    HGEigenDense<2> computeHGLightAffine(frameCPU *frame, int lvl, bool useWeights = false);

    HGEigenDense<8> computeHGPose(frameCPU *frame, int lvl, bool useWeights = false);
    // HGPose computeHGPose(dataCPU<float> &kfIdepth, frameCPU &kframe, frameCPU &frame, int lvl);

    HGMapped computeHGMap(frameCPU *frame, int lvl);
    HGEigenSparse computeHGMap2(frameCPU *frame, dataCPU<float> *mask, int lvl);

    HGMapped computeHGPoseMap(frameCPU *frame, int frameIndex, int numFrames, int lvl);
    HGEigenSparse computeHGPoseMap2(frameCPU *frame, int frameIndex, int numFrames, int lvl);

    // params
    bool multiThreading;
    float meshRegularization;
    float meshInitial;

    dataCPU<float> image_buffer;
    dataCPU<float> idepth_buffer;
    dataCPU<float> ivar_buffer;
    dataCPU<float> error_buffer;

    dataCPU<vec2<float>> jlightaffine_buffer;
    dataCPU<vec8<float>> jpose_buffer;

    dataCPU<jmapType> jmap_buffer;
    dataCPU<idsType> pId_buffer;

    // debug
    dataCPU<float> debug;
    dataCPU<float> idepthVar;

    renderCPU<sceneType, shapeType, jmapType, idsType> renderer;
    reduceCPU<jmapType, idsType> reducer;
};
