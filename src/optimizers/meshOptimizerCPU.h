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
#include "cpu/IndexThreadReduce.h"
#include "params.h"

class meshOptimizerCPU
{
public:
    meshOptimizerCPU(camera &cam);

    void initKeyframe(frameCPU &frame, dataCPU<float> &idepth, dataCPU<float> &idepthVar);

    void renderIdepth(Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderImage(Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderDebug(Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderInvVar(Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderError(frameCPU &frame, dataCPU<float> &buffer, int lvl);

    void optPose(frameCPU &frame);
    void optMap(std::vector<frameCPU> &frame);
    void optPoseMap(std::vector<frameCPU> &frame);

    void changeKeyframe(frameCPU &frame)
    {
        int lvl = 1;
        // the keyframemesh is relative to the keyframe pose
        // so to transform the pose coordinate system
        // just have to multiply with the pose increment from the keyframe
        keyframeMesh.transform(frame.pose * keyframe.pose.inverse());
        keyframe = frame;

        dataCPU<float> image(cam[0].width, cam[0].height, -1);
        renderImage(keyframe.pose, image, lvl);

        //float imageNoData = image.getPercentNoData(lvl);

        keyframeMesh.extrapolateMesh(cam[lvl], image, lvl);

        dataCPU<float> idepth(cam[0].width, cam[0].height, -1);
        renderIdepth(keyframe.pose, idepth, lvl);
        
        dataCPU<float> invVar(cam[0].width, cam[0].height, -1);
        renderInvVar(keyframe.pose, invVar, lvl);

        //float idepthNoData = idepth.getPercentNoData(lvl);

        initKeyframe(frame, idepth, invVar);
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
    MatrixMapped invIdepthVar;

private:
    MeshCPU keyframeMesh;

    camera cam[MAX_LEVELS];

    dataCPU<float> z_buffer;

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

    IndexThreadReduce<Error> errorTreadReduce;
    IndexThreadReduce<HGMapped> hgMappedTreadReduce;

    // params
    bool multiThreading;
    float meshRegularization;
    float meshInitial;
};
