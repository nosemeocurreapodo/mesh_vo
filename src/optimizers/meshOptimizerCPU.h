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

    std::pair<Eigen::Vector3f, std::array<unsigned int, 2>> triangulatePixel(MeshCPU &frameMesh, Eigen::Vector2f &pix, int lvl)
    {
        // adding a new vertice should be done with respect to a particular image
        // meaning, a particular projection
        // so we use the texcoor of the vert
        // if the vertice is inside the current mesh
        // use the delaunay triangulation
        // if it is not inside the current mesh
        // connect to edge, and update the edge

        std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> edgeFront = frameMesh.computeEdgeFront();

        std::array<unsigned int, 2> closestEdge;
        unsigned int closestTri;
        float closestDistance = std::numeric_limits<float>::max();
        for (int i = 0; i < edgeFront.size(); i++)
        {
            std::array<unsigned int, 2> edge = edgeFront[i].first;
            unsigned int t_id = edgeFront[i].second;

            Eigen::Vector2f med = (frameMesh.getTexCoord(edge[0]) + frameMesh.getTexCoord(edge[1])) / 2.0;
            float distance = (pix - med).norm();
            if (distance < closestDistance)
            {
                closestEdge = edge;
                closestTri = t_id;
                closestDistance = distance;
            }
        }

        Triangle3D closest_tri_t = frameMesh.getTriangle3D(closestTri);
        Eigen::Vector3f ray = cam.toRay(pix, lvl);
        float depth = closest_tri_t.getDepth(ray);

        Eigen::Vector3f pos = ray * depth;

        std::pair<Eigen::Vector3f, std::array<unsigned int, 2>> pos_and_edge;

        pos_and_edge.first = pos;
        pos_and_edge.second = closestEdge;

        return pos_and_edge;
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
