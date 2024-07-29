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

    void optPose(frameCPU &keyframe, frameCPU &frame);
    void optMap(frameCPU &keyframe, std::vector<frameCPU> &frames);
    void optPoseMap(frameCPU &keyframe, std::vector<frameCPU> &frame);

    dataCPU<float> getIdepth(Sophus::SE3f &pose, int lvl)
    {
        idepth_buffer.set(idepth_buffer.nodata, lvl);
        renderer.setScene(sceneOptimized);
        renderer.renderIdepth(cam[lvl], pose, idepth_buffer, lvl);
        return idepth_buffer;
    }

    dataCPU<float> getImage(frameCPU &frame, Sophus::SE3f &pose, int lvl)
    {
        image_buffer.set(image_buffer.nodata, lvl);
        renderer.setScene(sceneOptimized);
        renderer.renderImage(cam[lvl], frame, pose, image_buffer, lvl);
        return image_buffer;
    }

    void plotDebug(frameCPU &keyframe, frameCPU &frame)
    {
        idepth_buffer.set(idepth_buffer.nodata, 1);
        image_buffer.set(image_buffer.nodata, 1);

        renderer.setScene(sceneOptimized);

        renderer.renderIdepthParallel(cam[1], frame.pose, idepth_buffer, 1);
        renderer.renderImageParallel(cam[1], keyframe, frame.pose, image_buffer, 1);

        debug.set(debug.nodata, 0);
        renderer.renderDebugParallel(cam[0], frame, debug, 0);

        error_buffer = frame.image.sub(image_buffer, 1);

        show(frame.image, "frame image", 1);
        show(keyframe.image, "keyframe image", 1);
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

        renderer.renderIdepth(cam[lvl], frame.pose, idepth, lvl);

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

    SceneMesh sceneOptimized;
    camera cam[MAX_LEVELS];

private:
    Error computeError(frameCPU &kframe, frameCPU &frame, int lvl);
    //Error computeError(dataCPU<float> &kfIdepth, frameCPU &kframe, frameCPU &frame, int lvl);

    HGPose computeHGPose(frameCPU &kframe, frameCPU &frame, int lvl);
    //HGPose computeHGPose(dataCPU<float> &kfIdepth, frameCPU &kframe, frameCPU &frame, int lvl);

    HGMapped computeHGMap(frameCPU &kframe, frameCPU &frame, int lvl);
    HGMapped computeHGPoseMap(frameCPU &kframe, frameCPU &frame, int lvl);

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
