#pragma once

#include <Eigen/Core>

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGPoseMapMesh.h"
#include "common/Error.h"
#include "common/common.h"
#include "common/Mesh.h"
#include "cpu/frameCPU.h"
#include "cpu/IndexThreadReduce.h"
#include "params.h"

class rayDepthMeshSceneCPU
{
public:
    rayDepthMeshSceneCPU(float fx, float fy, float cx, float cy, int width, int height);

    void init(frameCPU &frame, dataCPU<float> &idepth);

    dataCPU<float> computeFrameIdepth(frameCPU &frame, int lvl);
    dataCPU<float> computeErrorImage(frameCPU &frame, int lvl);
    dataCPU<float> computeSceneImage(frameCPU &frame, int lvl);
    dataCPU<float> computeDebug(frameCPU &frame, int lvl);

    void optPose(frameCPU &frame);
    void optMap(std::vector<frameCPU> &frame);
    void optPoseMap(std::vector<frameCPU> &frame);

    camera getCam()
    {
        return cam;
    }

private:

    Mesh sceneMesh;
    Mesh observedMesh;

    frameCPU keyframe;
    camera cam;

    dataCPU<float> z_buffer;

    Error computeError(frameCPU &frame, int lvl);
    HGPose computeHGPose(frameCPU &frame, int lvl);
    HGPoseMapMesh computeHGMap(frameCPU &frame, int lvl);

    void computeHGPoseMap(frameCPU &frame, HGPoseMapMesh &hg, int frame_index, int lvl);

    void errorPerIndex(frameCPU &frame, int lvl, int tmin, int tmax, Error *e, int tid);
    void HGPosePerIndex(frameCPU &frame, int lvl, int tmin, int tmax, HGPose *hg, int tid);
    void HGMapPerIndex(frameCPU &frame, int lvl, int tmin, int tmax, HGPoseMapMesh *hg, int tid);

    Error errorRegu();
    HGPoseMapMesh HGRegu();

    IndexThreadReduce<Error> errorTreadReduce;
    IndexThreadReduce<HGPose> hgPoseTreadReduce;
    IndexThreadReduce<HGPoseMapMesh> hgPoseMapTreadReduce;

    // params
    bool multiThreading;
    float meshRegularization;

};
