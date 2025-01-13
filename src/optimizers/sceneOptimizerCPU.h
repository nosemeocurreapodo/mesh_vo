#pragma once

#include <Eigen/Core>
// #include <Eigen/CholmodSupport>
//  #include <Eigen/SPQRSupport>
// #include <thread>

#include "common/camera.h"
#include "common/types.h"
#include "common/DenseLinearProblem.h"
// #include "common/SparseLinearProblem.h"
//  #include "common/HGMapped.h"
#include "common/Error.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/reduceCPU.h"
// #include "cpu/SceneBase.h"
// #include "cpu/ScenePatches.h"
#include "cpu/SceneMesh.h"
// #include "cpu/SceneSurfels.h"
// #include "cpu/SceneMeshSmooth.h"
#include "cpu/OpenCVDebug.h"
#include "params.h"

template <typename sceneType, typename jmapType, typename idsType>
class sceneOptimizerCPU
{
public:
    sceneOptimizerCPU(camera &_cam);

    void normalizeDepth();

    void optLightAffine(frameCPU &frame, frameCPU &kframe, sceneType &scene);
    void optPose(frameCPU &frame, frameCPU &kframe, sceneType &scene);
    void optMap(std::vector<frameCPU> &frames, frameCPU &kframe, sceneType &scene);
    void optPoseMap(std::vector<frameCPU> &frames, frameCPU &kframe, sceneType &scene);

    void setMeshRegu(float mr)
    {
        sceneRegularization = mr;
    }

    /*
    dataCPU<float> getImage(frameCPU &frame, Sophus::SE3f &pose, int lvl)
    {
        image_buffer.set(image_buffer.nodata, lvl);
        renderer.renderImage(cam[lvl], frame, pose, image_buffer, lvl);
        return image_buffer;
    }
    */

private:
    void plotDebug(sceneType &scene, frameCPU &kframe, std::vector<frameCPU> &frames)
    {
        int lvl = 1;

        dataCPU<float> error_buffer(cam[lvl].width, cam[lvl].height, -1);
        dataCPU<float> idepth_buffer(cam[lvl].width, cam[lvl].height, -1);
        renderer.renderIdepthParallel(scene, kframe.getPose(), cam[lvl], idepth_buffer);

        show(kframe.getRawImage(lvl), "keyframe image", false);
        show(idepth_buffer, "kframe idepth", true);

        if (frames.size() > 0)
        {
            dataCPU<float> frames_buffer(cam[lvl].width * frames.size(), cam[lvl].height, -1);
            dataCPU<float> residual_buffer(cam[lvl].width * frames.size(), cam[lvl].height, -1);
            for (int i = 0; i < (int)frames.size(); i++)
            {
                error_buffer.set(error_buffer.nodata);
                renderer.renderResidualParallel(scene, kframe.getRawImage(lvl), kframe.getAffine(), kframe.getPose(), frames[i].getRawImage(lvl), frames[i].getAffine(), frames[i].getPose(), cam[lvl], error_buffer);

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
    }

    Error computeError(frameCPU &frame, frameCPU &kframe, sceneType &scene, int lvl)
    {
        error_buffer.set(error_buffer.nodata, lvl);
        renderer.renderResidualParallel(scene, kframe.getRawImage(lvl), kframe.getAffine(), kframe.getPose(), frame.getRawImage(lvl), frame.getAffine(), frame.getPose(), cam[lvl], error_buffer.get(lvl));
        Error e = reducer.reduceErrorParallel(error_buffer.get(lvl));
        return e;
    }

    DenseLinearProblem computeHGLightAffine(frameCPU &frame, frameCPU &kframe, sceneType &scene, int lvl)
    {
        jlightaffine_buffer.set(jlightaffine_buffer.nodata, lvl);
        error_buffer.set(error_buffer.nodata, lvl);

        renderer.renderJLightAffineParallel(scene, kframe.getRawImage(lvl), kframe.getAffine(), kframe.getPose(), frame.getRawImage(lvl), frame.getAffine(), frame.getPose(), cam[lvl], jlightaffine_buffer.get(lvl), error_buffer.get(lvl));
        DenseLinearProblem hg = reducer.reduceHGLightAffineParallel(jlightaffine_buffer.get(lvl), error_buffer.get(lvl));

        return hg;
    }

    DenseLinearProblem computeHGPose(frameCPU &frame, frameCPU &kframe, sceneType &scene, int lvl)
    {
        jpose_buffer.set(jpose_buffer.nodata, lvl);
        error_buffer.set(error_buffer.nodata, lvl);

        renderer.renderJPoseParallel(scene, kframe.getRawImage(lvl), kframe.getAffine(), kframe.getPose(), frame.getRawImage(lvl), frame.getAffine(), frame.getdIdpixImage(lvl), frame.getPose(), cam[lvl], jpose_buffer.get(lvl), error_buffer.get(lvl));
        DenseLinearProblem hg = reducer.reduceHGPoseParallel(jpose_buffer.get(lvl), error_buffer.get(lvl));

        return hg;
    }

    /*
    HGMapped computeHGMap(frameCPU *frame, int lvl)
    {
        error_buffer.set(error_buffer.nodata, lvl);
        jmap_buffer.set(jmap_buffer.nodata, lvl);
        pId_buffer.set(pId_buffer.nodata, lvl);

        // renderer.renderJMap(scene, cam[lvl], kframe, frame, jmap_buffer, error_buffer, pId_buffer, lvl);
        renderer.renderJMapParallel(&kscene, &kimage, frame, cam[lvl], &jmap_buffer, &error_buffer, &pId_buffer, lvl);
        // HGMapped hg = reducer.reduceHGMap(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);
        HGMapped hg = reducer.reduceHGMapParallel(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);

        return hg;
    }
    */

    DenseLinearProblem computeHGMap2(frameCPU &frame, frameCPU &kframe, sceneType &scene, int lvl)
    {
        error_buffer.set(error_buffer.nodata, lvl);
        jmap_buffer.set(jmap_buffer.nodata, lvl);
        pId_buffer.set(pId_buffer.nodata, lvl);

        int numMapParams = scene.getParamIds().size();

        renderer.renderJMapParallel(scene, kframe.getRawImage(lvl), kframe.getAffine(), kframe.getPose(), frame.getRawImage(lvl), frame.getAffine(), frame.getdIdpixImage(lvl), frame.getPose(), cam[lvl], jmap_buffer.get(lvl), error_buffer.get(lvl), pId_buffer.get(lvl));
        DenseLinearProblem hg = reducer.reduceHGMap2(numMapParams, jmap_buffer.get(lvl), error_buffer.get(lvl), pId_buffer.get(lvl));

        return hg;
    }
    /*
    HGMapped computeHGPoseMap(frameCPU *frame, int frameIndex, int numFrames, int lvl)
    {
        jpose_buffer.set(jpose_buffer.nodata, lvl);
        jmap_buffer.set(jmap_buffer.nodata, lvl);
        error_buffer.set(error_buffer.nodata, lvl);
        pId_buffer.set(pId_buffer.nodata, lvl);

        renderer.renderJPoseMapParallel(&kscene, &kimage, frame, cam[lvl], &jpose_buffer, &jmap_buffer, &error_buffer, &pId_buffer, lvl);
        HGMapped hg = reducer.reduceHGPoseMapParallel(cam[lvl], frameIndex, numFrames, kscene.getNumParams(), jpose_buffer, jmap_buffer, error_buffer, pId_buffer, lvl);

        return hg;
    }
    */

    DenseLinearProblem computeHGPoseMap2(frameCPU &frame, frameCPU &kframe, sceneType &scene, int frameIndex, int numFrames, int lvl)
    {
        jpose_buffer.set(jpose_buffer.nodata, lvl);
        jmap_buffer.set(jmap_buffer.nodata, lvl);
        error_buffer.set(error_buffer.nodata, lvl);
        pId_buffer.set(pId_buffer.nodata, lvl);

        int numMapParams = scene.getParamIds().size();

        renderer.renderJPoseMapParallel(scene, kframe.getRawImage(lvl), kframe.getAffine(), kframe.getPose(), frame.getRawImage(lvl), frame.getAffine(), frame.getdIdpixImage(lvl), frame.getPose(), cam[lvl], jpose_buffer.get(lvl), jmap_buffer.get(lvl), error_buffer.get(lvl), pId_buffer.get(lvl));
        // HGEigenSparse hg = reducer.reduceHGPoseMap2(cam[lvl], frameIndex, numFrames, scene->getNumParams(), jpose_buffer, jmap_buffer, error_buffer, pId_buffer, lvl);
        DenseLinearProblem hg = reducer.reduceHGPoseMapParallel2(frameIndex, numFrames, numMapParams, jpose_buffer.get(lvl), jmap_buffer.get(lvl), error_buffer.get(lvl), pId_buffer.get(lvl));

        return hg;
    }

    // params
    float sceneRegularization;
    float sceneInitial;
    float poseInitial;

    dataMipMapCPU<float> image_buffer;
    dataMipMapCPU<float> idepth_buffer;
    dataMipMapCPU<float> error_buffer;

    dataMipMapCPU<vec2<float>> jlightaffine_buffer;
    dataMipMapCPU<vec8<float>> jpose_buffer;

    dataMipMapCPU<jmapType> jmap_buffer;
    dataMipMapCPU<idsType> pId_buffer;

    // debug
    dataMipMapCPU<float> debug_buffer;

    renderCPU<sceneType> renderer;
    reduceCPU reducer;

    cameraMipMap cam;
};
