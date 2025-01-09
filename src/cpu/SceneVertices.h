#pragma once
#include <memory>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/common.h"
#include "cpu/GeometryVertices.h"
#include "cpu/Shapes.h"
#include "cpu/frameCPU.h"

class ScceneVertices // : public SceneBase
{
public:
    SceneVertices()
    {
    }

    void init(std::vector<vec3<float>> &vertices, Sophus::SE3f pose, camera cam)
    {
        m_globalPose = pose;
        m_geometry.init(cam, vertices);
    }

    void init(std::vector<vec2<float>> &texcoords, std::vector<float> idepths, Sophus::SE3f pose, camera cam)
    {
        m_globalPose = pose;
        m_geometry.init(cam, texcoords, idepths);
    }

    void init(dataCPU<float> &idepth, Sophus::SE3f pose, camera cam)
    {
        m_globalPose = pose;

        std::vector<vec2<float>> texcoords;
        std::vector<float> idepths;

        int i = 0;
        for (float y = 0.0; y < MESH_HEIGHT; y++)
        {
            for (float x = 0.0; x < MESH_WIDTH; x++)
            {
                if (i >= MAX_VERTEX_SIZE)
                    return;

                vec2<float> pix;
                pix(0) = (cam.width - 1) * x / (MESH_WIDTH - 1);
                pix(1) = (cam.height - 1) * y / (MESH_HEIGHT - 1);

                float idph = idepth.get(pix(1), pix(0));

                assert(idph != idepth.nodata);
                assert(idph > 0.0);

                texcoords.push_back(pix);
                idepths.push_back(idph);

                i++;
            }
        }
    }

    void transform(camera cam, Sophus::SE3f globalPose)
    {
        Sophus::SE3f relativePose = globalPose*m_globalPose.inverse();
        m_globalPose = globalPose;
        m_geometry.transform(relativePose);
        m_geometry.project(cam);
    }

    Sophus::SE3f getGlobalPose()
    {
        return m_globalPose;
    }

private:
    GeometryVertices m_geometry;
    Sophus::SE3f m_globalPose;
};
