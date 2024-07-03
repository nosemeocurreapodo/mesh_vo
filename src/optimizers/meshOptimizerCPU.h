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

    void init(frameCPU &frame, dataCPU<float> &idepth);

    void renderIdepth(Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderImage(Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderDebug(Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderError(frameCPU &frame, dataCPU<float> &buffer, int lvl);

    void optPose(frameCPU &frame);
    void optMap(std::vector<frameCPU> &frame);
    void optPoseMap(std::vector<frameCPU> &frame);

    void changeKeyframe(frameCPU &frame)
    {
        // the keyframemesh is relative to the keyframe pose
        // so to transform the pose coordinate system
        // just have to multiply with the pose increment from the keyframe
        keyframeMesh.transform(frame.pose * keyframe.pose.inverse());
        completeMesh2(keyframeMesh);
        frame.copyTo(keyframe);
    }

    void completeMesh(frameCPU &frame);

    void completeMesh2(MeshCPU &mesh)
    {
        int lvl = 1;

        dataCPU<float> image(cam[0].width, cam[0].height, -1);
        Sophus::SE3f pose;
        mesh.removeOcluded(cam[lvl]);
        renderImage(pose, image, lvl);

        float percentNoData = image.getPercentNoData(lvl);

        for (int y = 0; y < cam[lvl].height; y += float(cam[lvl].height - 1) / (MESH_HEIGHT - 1.0))
        {
            for (int x = 0; x < cam[lvl].width; x += float(cam[lvl].width - 1) / (MESH_WIDTH - 1.0))
            {
                if (image.get(y, x, lvl) != image.nodata)
                    continue;

                Eigen::Vector2f pix(x, y);
                Eigen::Vector3f new_vertice = triangulatePixel(mesh, pix, lvl);

                mesh.addVertice(new_vertice);
            }
        }

        mesh.buildTriangles(cam[lvl]);
    }

    Eigen::Vector3f triangulatePixel(MeshCPU &frameMesh, Eigen::Vector2f &pix, int lvl)
    {
        std::vector<unsigned int> trisIds = frameMesh.getSortedTriangles(pix);
        Eigen::Vector3f pos;
        for (auto triId : trisIds)
        {
            Triangle3D tri = frameMesh.getTriangle3D(triId);
            if (tri.getArea() < 0.0)
                continue;
            // if (tri.isBackFace())
            //     continue;
            Eigen::Vector3f ray = cam[lvl].pixToRay(pix);
            float depth = tri.getDepth(ray);
            pos = ray * depth;
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
    void HGPoseMapPerIndex(frameCPU &frame, int frame_index, MeshCPU &frameMesh, std::vector<unsigned int> trisIds, int lvl, int tmin, int tmax, HGMapped *hg, int tid);

    Error errorRegu();
    HGMapped HGRegu();

    IndexThreadReduce<Error> errorTreadReduce;
    IndexThreadReduce<HGMapped> hgMappedTreadReduce;

    // params
    bool multiThreading;
    float meshRegularization;
};
