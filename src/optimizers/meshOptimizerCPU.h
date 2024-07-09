#pragma once

#include <Eigen/Core>

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGMapped.h"
#include "common/Error.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/MeshCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/IndexThreadReduce.h"
#include "cpu/OpenCVDebug.h"
#include "params.h"

enum OptimizationMethod
{
    depth,
    idepth,
    log_depth,
    log_idepth
};

class meshOptimizerCPU
{
public:
    meshOptimizerCPU(camera &cam);

    void initKeyframe(frameCPU &frame, dataCPU<float> &idepth, dataCPU<float> &idepthVar, int lvl);
    MeshCPU buildFrameMesh(frameCPU &frame, int lvl);

    void optPose(frameCPU &frame);
    void optMap(std::vector<frameCPU> &frame);
    void optPoseMap(std::vector<frameCPU> &frame);

    dataCPU<float> getIdepth(Sophus::SE3f &pose)
    {
        idepth.set(idepth.nodata, 1);
        renderer.renderIdepth(keyframeMesh, cam[1], pose, idepth, 1);
        return idepth;
    }

    dataCPU<float> getImage(Sophus::SE3f &pose)
    {
        sceneImage.set(sceneImage.nodata, 1);
        renderer.renderImage(keyframeMesh, cam[1], keyframe.image, pose, sceneImage, 1);
        return sceneImage;
    }

    void plotDebug(frameCPU &frame)
    {
        idepth.set(idepth.nodata, 1);
        error.set(error.nodata, 1);
        debug.set(debug.nodata, 0);
        sceneImage.set(sceneImage.nodata, 1);
        renderer.renderIdepth(keyframeMesh, cam[1], frame.pose, idepth, 1);
        renderer.renderError(keyframeMesh, cam[1], keyframe, frame, error, 1);
        renderer.renderDebug(keyframeMesh, cam[0], frame.pose, debug, 0);
        // renderer.renderInvVar(meshOptimizer.keyframeMesh, lastFrame, idepthVar, 1);
        renderer.renderImage(keyframeMesh, cam[1], keyframe.image, frame.pose, sceneImage, 1);

        show(debug, "frame debug", 0);
        show(frame.image, "frame image", 1);
        show(keyframe.image, "keyframe image", 1);
        show(error, "lastFrame error", 1);
        show(idepth, "lastFrame idepth", 1);
        // show(idepthVar, "lastFrame invVar", 1);
        show(sceneImage, "lastFrame scene", 1);
    }

    void changeKeyframe(frameCPU &frame)
    {
        int lvl = 1;
        /*
        // the keyframemesh is relative to the keyframe pose
        // so to transform the pose coordinate system
        // just have to multiply with the pose increment from the keyframe
        keyframeMesh.transform(frame.pose * keyframe.pose.inverse());
        keyframe = frame;


        dataCPU<float> idepth(cam[0].width, cam[0].height, -1);
        idepth.setRandom(lvl);

        renderIdepth(keyframe.pose, idepth, lvl);


        dataCPU<float> invVar(cam[0].width, cam[0].height, -1);
        invVar.set(1.0/INITIAL_VAR, lvl);

        renderInvVar(keyframe.pose, invVar, lvl);

        float idepthNoData = idepth.getPercentNoData(lvl);

        initKeyframe(frame, idepth, invVar, lvl);
        */

        MeshCPU frameMesh = buildFrameMesh(frame, lvl);
        keyframeMesh = frameMesh.getCopy();
        keyframe = frame;
    }

    std::vector<std::array<unsigned int, 2>> getPixelEdges(MeshCPU &frameMesh, Eigen::Vector2f &pix, int lvl)
    {
        // adding a new vertice should be done with respect to a particular image
        // meaning, a particular projection
        // so we use the texcoor of the vert
        // if the vertice is inside the current mesh
        // use the delaunay triangulation
        // if it is not inside the current mesh
        // connect to edge, and update the edge

        std::vector<std::array<unsigned int, 2>> edge_vector;

        std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> edgeFront = frameMesh.computeEdgeFront();
        // std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> edgeFront = frameMesh.getSortedEdgeFront(pix);

        for (auto edge : edgeFront)
        {
            std::array<unsigned int, 2> ed = edge.first;
            unsigned int t_id = edge.second;

            Triangle2D tri2D = frameMesh.getTexCoordTriangle(t_id);
            /*
            if(tri2D.getArea() < MIN_TRIANGLE_AREA)
                continue;
            if(!cam.isPixVisible(frameMesh.getTexCoord(ed[0]), lvl))
                continue;
            if(!cam.isPixVisible(frameMesh.getTexCoord(ed[1]), lvl))
                continue;
            */

            Eigen::Vector2f edgeMean = (frameMesh.getTexCoord(ed[0]) + frameMesh.getTexCoord(ed[1])) / 2.0;
            Eigen::Vector2f dir = (pix - edgeMean).normalized();
            Eigen::Vector2f testpix = edgeMean + 2.0 * dir;
            tri2D.computeTinv();
            tri2D.computeBarycentric(testpix);
            if (tri2D.isBarycentricOk())
                continue;

            edge_vector.push_back(ed);
        }

        return edge_vector;
    }

    frameCPU keyframe;
    MeshCPU keyframeMesh;
    MatrixMapped invVar;
    camera cam[MAX_LEVELS];

private:
    Error computeError(frameCPU &frame, int lvl);
    HGMapped computeHGPose(frameCPU &frame, int lvl);
    HGMapped computeHGMap(frameCPU &frame, int lvl);
    HGMapped computeHGPoseMap(frameCPU &frame, int frame_index, int lvl);

    void errorPerIndex(frameCPU &frame, MeshCPU &frameMesh, std::vector<unsigned int> trisIds, int lvl, int tmin, int tmax, Error *e, int tid);
    void HGPosePerIndex(frameCPU &frame, MeshCPU &frameMesh, std::vector<unsigned int> trisIds, int lvl, int tmin, int tmax, HGMapped *hg, int tid);
    void HGMapPerIndex(frameCPU &frame, MeshCPU &frameMesh, std::vector<unsigned int> trisIds, int lvl, int tmin, int tmax, HGMapped *hg, int tid);
    void HGPoseMapPerIndex(frameCPU &frame, MeshCPU &frameMesh, std::vector<unsigned int> trisIds, int lvl, int tmin, int tmax, HGMapped *hg, int tid);

    Error errorRegu();
    HGMapped HGRegu();

    Error errorInitial(MeshCPU &initialMesh, MatrixMapped &initialInvDepthMap);
    HGMapped HGInitial(MeshCPU &initialMesh, MatrixMapped &initialInvDepthMap);

    renderCPU renderer;

    IndexThreadReduce<Error> errorTreadReduce;
    IndexThreadReduce<HGMapped> hgMappedTreadReduce;

    // params
    bool multiThreading;
    float meshRegularization;
    float meshInitial;

    OptimizationMethod optMethod;

    // debug
    dataCPU<float> idepth;
    dataCPU<float> error;
    dataCPU<float> sceneImage;
    dataCPU<float> debug;
    dataCPU<float> idepthVar;
};
