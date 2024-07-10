#pragma once

#include <Eigen/Core>

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGMapped.h"
#include "common/Error.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/MeshTexCoordsCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/OpenCVDebug.h"
#include "params.h"

class meshOptimizerCPU
{
public:
    meshOptimizerCPU(camera &cam);

    void initKeyframe(frameCPU &frame, dataCPU<float> &idepth, dataCPU<float> &idepthVar, int lvl);
    MeshTexCoordsCPU buildFrameMesh(frameCPU &frame, int lvl);

    void optPose(frameCPU &frame);
    void optMapDepth(std::vector<frameCPU> &frame);
    void optMapNormalDepth(std::vector<frameCPU> &frame);
    void optPoseMap(std::vector<frameCPU> &frame);

    dataCPU<float> getIdepth(Sophus::SE3f &pose, int lvl)
    {
        idepth_buffer.set(idepth_buffer.nodata, lvl);
        renderer.renderIdepth(keyframeMesh, cam[lvl], pose, idepth_buffer, lvl);
        return idepth_buffer;
    }

    dataCPU<float> getImage(Sophus::SE3f &pose, int lvl)
    {
        image_buffer.set(image_buffer.nodata, lvl);
        renderer.renderImage(keyframeMesh, cam[lvl], keyframe.image, pose, image_buffer, lvl);
        return image_buffer;
    }

    void plotDebug(frameCPU &frame)
    {
        idepth_buffer.set(idepth_buffer.nodata, 1);
        image_buffer.set(image_buffer.nodata, 1);

        renderer.renderIdepth(keyframeMesh, cam[1], frame.pose, idepth_buffer, 1);
        renderer.renderImage(keyframeMesh, cam[1], keyframe.image, frame.pose, image_buffer, 1);

        debug.set(debug.nodata, 0);
        renderer.renderDebug(keyframeMesh, cam[0], frame.pose, debug, 0);

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

        renderer.renderIdepth(keyframeMesh, cam[lvl], frame.pose, idepth, lvl);

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

    std::vector<std::array<unsigned int, 2>> getPixelEdges(MeshTexCoordsCPU &frameMesh, Eigen::Vector2f &pix, int lvl)
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
    MeshTexCoordsCPU keyframeMesh;
    MatrixMapped invVar;
    camera cam[MAX_LEVELS];

private:
    Error computeError(frameCPU &frame, int lvl);
    HGMapped computeHGPose(frameCPU &frame, int lvl);
    HGMapped computeHGMapDepth(frameCPU &frame, int lvl);
    HGMapped computeHGMapNormalDepth(frameCPU &frame, int lvl);
    HGMapped computeHGPoseMap(frameCPU &frame, int frame_index, int lvl);

    Error errorRegu();
    HGMapped HGRegu();

    Error errorInitial(MeshTexCoordsCPU &initialMesh, MatrixMapped &initialInvDepthMap);
    HGMapped HGInitial(MeshTexCoordsCPU &initialMesh, MatrixMapped &initialInvDepthMap);

    renderCPU renderer;

    // params
    bool multiThreading;
    float meshRegularization;
    float meshInitial;

    MapJacobianMethod jacMethod;

    dataCPU<float> image_buffer;
    dataCPU<float> idepth_buffer;
    dataCPU<float> error_buffer;
    dataCPU<Eigen::Vector3f> j1_buffer;
    dataCPU<Eigen::Vector3f> j2_buffer;
    dataCPU<Eigen::Vector3f> j3_buffer;
    dataCPU<Eigen::Vector3i> id_buffer;

    // debug
    dataCPU<float> debug;
    dataCPU<float> idepthVar;
};
