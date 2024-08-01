#pragma once

#include <Eigen/Core>
#include <thread>

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGMapped.h"
#include "common/Error.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/reduceCPU.h"
#include "cpu/SceneBase.h"
//#include "cpu/ScenePatches.h"
#include "cpu/SceneMesh.h"
//#include "cpu/SceneSurfels.h"
//#include "cpu/SceneMeshSmooth.h"
#include "cpu/OpenCVDebug.h"
#include "params.h"

class meshOptimizerCPU
{
public:
    meshOptimizerCPU(camera &cam);

    void initKeyframe(frameCPU &frame, dataCPU<float> &idepth, dataCPU<float> &idepthVar, int lvl);

    void optPose(frameCPU &frame);
    void optMap(std::vector<frameCPU> &frames);
    void optPoseMap(std::vector<frameCPU> &frame);

    /*
    dataCPU<float> getIdepth(Sophus::SE3f &pose, int lvl)
    {
        idepth_buffer.set(idepth_buffer.nodata, lvl);
        renderer.renderIdepthParallel(cam[lvl], pose, idepth_buffer, lvl);
        return idepth_buffer;
    }
    */

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
        debug.set(debug.nodata, 0);

        scene->transform(frame.pose);
        //renderer.renderIdepth(cam[1], frame.pose, idepth_buffer, 1);
        renderer.renderIdepthParallel(scene.get(), cam[1], &idepth_buffer, 1);
        renderer.renderImageParallel(&kscene, &kframe, scene.get(), cam[1], &image_buffer, 1);

        renderer.renderDebugParallel(scene.get(), &frame, cam[0], &debug, 0);

        error_buffer = frame.image.sub(image_buffer, 1);

        show(frame.image, "frame image", 1);
        show(kframe.image, "keyframe image", 1);
        show(error_buffer, "lastFrame error", 1);
        show(idepth_buffer, "lastFrame idepth", 1);
        show(image_buffer, "lastFrame scene", 1);

        show(debug, "frame debug", 0);
    }

    void changeKeyframe(frameCPU &frame)
    {
        int lvl = 1;

        // method 1
        // compute idepth, complete nodata with random
        // init mesh with it
        dataCPU<float> idepth(cam[0].width, cam[0].height, -1);
        idepth.setRandom(lvl);

        scene->transform(frame.pose);
        renderer.renderIdepthParallel(scene.get(), cam[lvl], &idepth, lvl);

        dataCPU<float> invVar(cam[0].width, cam[0].height, -1);
        invVar.set(1.0 / INITIAL_VAR, lvl);

        initKeyframe(frame, idepth, invVar, lvl);

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

    SceneMesh kscene;
    frameCPU kframe;

    camera cam[MAX_LEVELS];

private:

    std::unique_ptr<SceneBase> scene;

    Error computeError(SceneBase *scene, frameCPU *frame, int lvl);
    //Error computeError(dataCPU<float> &kfIdepth, frameCPU &kframe, frameCPU &frame, int lvl);

    HGPose computeHGPose(SceneBase *scene, frameCPU *frame, int lvl);
    //HGPose computeHGPose(dataCPU<float> &kfIdepth, frameCPU &kframe, frameCPU &frame, int lvl);

    HGMapped computeHGMap(SceneBase *scene, frameCPU *frame, int lvl);
    HGMapped computeHGPoseMap(SceneBase *scene, frameCPU *frame, int lvl);

    // params
    bool multiThreading;
    float meshRegularization;
    float meshInitial;

    dataCPU<float> image_buffer;
    dataCPU<float> idepth_buffer;
    dataCPU<float> error_buffer;
    dataCPU<std::array<float, 6>> jpose_buffer;
    dataCPU<std::array<float, MESH_DOF>> jmap_buffer;
    dataCPU<std::array<int, MESH_DOF>> pId_buffer;

    // debug
    dataCPU<float> debug;
    dataCPU<float> idepthVar;

    renderCPU renderer;
    reduceCPU reducer;

};
