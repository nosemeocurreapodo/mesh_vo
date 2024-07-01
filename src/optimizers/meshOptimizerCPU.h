#pragma once

#include <Eigen/Core>

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGPoseMapMesh.h"
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
    meshOptimizerCPU(float fx, float fy, float cx, float cy, int width, int height);

    void init(frameCPU &frame, dataCPU<float> &idepth);

    void renderIdepth(Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderImage(Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderDebug(Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderError(frameCPU &frame, dataCPU<float> &buffer, int lvl);

    void optPose(frameCPU &frame);
    void optMap(std::vector<frameCPU> &frame);
    // void optPoseMap(std::vector<frameCPU> &frame);

    void completeMesh(frameCPU &frame);

    Eigen::Vector3f triangulatePixel(MeshCPU &frameMesh, Eigen::Vector2f &pix, int lvl)
    {
        std::vector<unsigned int> trisIds = frameMesh.getSortedTriangles(pix);
        Eigen::Vector3f pos;
        for (auto triId : trisIds)
        {
            Triangle3D tri = frameMesh.getTriangle3D(triId);
            //if (tri.isBackFace())
            //    continue;
            Eigen::Vector3f ray = cam.toRay(pix, lvl);
            float depth = tri.getDepth(ray);
            pos = ray*depth;
            break;
        }
        return pos;
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

            Triangle2D tri2D = frameMesh.getTriangle2D(t_id);
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

private:
    MeshCPU globalMesh;
    MeshCPU keyframeMesh;

    frameCPU keyframe;
    camera cam;

    dataCPU<float> z_buffer;

    Error computeError(frameCPU &frame, int lvl);
    HGPose computeHGPose(frameCPU &frame, int lvl);
    HGPoseMapMesh computeHGMap(frameCPU &frame, int lvl);

    // void computeHGPoseMap(frameCPU &frame, HGPoseMapMesh &hg, int frame_index, int lvl);

    void errorPerIndex(frameCPU &frame, MeshCPU &frameMesh, std::vector<unsigned int> trisIds, int lvl, int tmin, int tmax, Error *e, int tid);
    void HGPosePerIndex(frameCPU &frame, MeshCPU &frameMesh, std::vector<unsigned int> trisIds, int lvl, int tmin, int tmax, HGPose *hg, int tid);
    void HGMapPerIndex(frameCPU &frame, MeshCPU &frameMesh, std::vector<unsigned int> trisIds, int lvl, int tmin, int tmax, HGPoseMapMesh *hg, int tid);

    Error errorRegu();
    HGPoseMapMesh HGRegu();

    IndexThreadReduce<Error> errorTreadReduce;
    IndexThreadReduce<HGPose> hgPoseTreadReduce;
    IndexThreadReduce<HGPoseMapMesh> hgPoseMapTreadReduce;

    // params
    bool multiThreading;
    float meshRegularization;
};
