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

    void init(std::vector<vec3<float>> &vertices, Sophus::SE3f pose, dataCPU<float> &texture, camera cam)
    {
        m_globalPose = pose;
        m_texture = texture;
        m_geometry.init(cam, vertices);
    }

    void init(std::vector<vec2<float>> &texcoords, std::vector<float> idepths, Sophus::SE3f pose, dataCPU<float> &texture, camera cam)
    {
        m_globalPose = pose;
        m_texture = texture;
        m_geometry.init(cam, texcoords, idepths);
    }

    void init(dataCPU<float> &idepth, Sophus::SE3f pose, dataCPU<float> &texture, camera cam, int lvl)
    {
        m_globalPose = pose;
        m_texture = texture;

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

                float idph = idepth.get(pix(1), pix(0), lvl);

                assert(idph != idepth.nodata);
                assert(idph > 0.0);

                texcoords.push_back(pix);
                idepths.push_back(idph);

                i++;
            }
        }
    }

    /*
    void complete(frameCPU &frame, camera &cam, dataCPU<float> &idepth, int lvl)
    {
        float max = 1.0;
        float min = 0.0;

        for (float y = 0.0; y < MESH_HEIGHT; y++)
        {
            for (float x = 0.0; x < MESH_WIDTH; x++)
            {
                vec2<float> pix;
                pix(0) = (cam.width - 1) * x / (MESH_WIDTH - 1);
                pix(1) = (cam.height - 1) * y / (MESH_HEIGHT - 1);

                int size = ((cam.width - 1) / (MESH_WIDTH - 1)) / 2;

                bool isNoData = true;
                for (int y_ = pix(1) - size; y_ <= pix(1) + size; y_++)
                {
                    for (int x_ = pix(0) - size; x_ <= pix(0) + size; x_++)
                    {
                        if (!cam.isPixVisible(x_, y_))
                            continue;

                        vec3<float> ray = cam.pixToRay(x_, y_);
                        float idph = idepth.get(y_, x_, lvl);
                        if (idph != idepth.nodata)
                            isNoData = false;
                    }
                }

                if (!isNoData)
                    continue;

                vec3<float> ray = cam.pixToRay(pix(0), pix(1));
                float idph = (max - min) * float(rand() % 1000) / 1000.0 + min;

                vec3<float> vertice = ray / idph;

                // vertices[id] = vertice;
                // rays[id] = ray;
                // pixels[id] = pix;
                vertices.push_back(vertice);
                rays.push_back(ray);
                pixels.push_back(pix);
                weights.push_back(initialIvar());

                for (int y_ = pix(1) - size; y_ <= pix(1) + size; y_++)
                {
                    for (int x_ = pix(0) - size; x_ <= pix(0) + size; x_++)
                    {
                        if (!cam.isPixVisible(x_, y_))
                            continue;
                        idepth.set(1.0, y_, x_, lvl);
                    }
                }
            }
        }
    }
    */

    void transform(camera cam, Sophus::SE3f relativePose)
    {
        m_geometry.transform(relativePose);
        m_geometry.project(cam);
    }

    Sophus::SE3f getGlobalPose()
    {
        return m_globalPose;
    }

private:
    GeometryVertices m_geometry;
    dataCPU<float> m_texture;
    Sophus::SE3f m_globalPose;
};
