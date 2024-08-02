#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/common.h"
#include "cpu/SceneVerticesBase.h"

class SceneVerticesNormalsBase : public SceneVerticesBase
{
public:

    SceneVerticesNormalsBase() : SceneVerticesBase()
    {
    }

    SceneVerticesNormalsBase(const SceneVerticesNormalsBase &other) : SceneVerticesBase(other)
    {
        normals = other.normals;
    }

    /*
    ScenePointsNormals &operator=(const ScenePointsNormals &other)
    {
        if (this != &other)
        {
            SceneVerticesBase::operator=(other);
            normals = other.normals;
        }
        return *this;
    }
    */

    void clear() override
    {
        SceneVerticesBase::clear();
        normals.clear();
    }

    void init(frameCPU &frame, camera &cam, dataCPU<float> &idepth, int lvl)
    {
        clear();
        setPose(frame.pose);

        for (float y = 0.0; y < MESH_HEIGHT; y++)
        {
            for (float x = 0.0; x < MESH_WIDTH; x++)
            {
                Eigen::Vector2f pix;
                pix[0] = (cam.width - 1) * x / (MESH_WIDTH - 1);
                pix[1] = (cam.height - 1) * y / (MESH_HEIGHT - 1);
                Eigen::Vector3f ray = cam.pixToRay(pix);
                float idph = idepth.get(pix[1], pix[0], lvl);
                if (idph == idepth.nodata)
                    continue;

                if (idph <= 0.0)
                    continue;

                Eigen::Vector3f vertice = ray / idph;

                unsigned int id = addVertice(vertice);
                setNormal(ray, id);
            }
        }
    }

    unsigned int addVertice(Eigen::Vector3f &vert)
    {
        unsigned int id = SceneVerticesBase::addVertice(vert);
        normals[id] = Eigen::Vector3f(0.0, 0.0, 1.0);
        return id;
    }

    unsigned int addVertice(Eigen::Vector3f &vert, Eigen::Vector3f &normal)
    {
        unsigned int id = SceneVerticesBase::addVertice(vert);
        normals[id] = normal;
        return id;
    }

    void removeVertice(unsigned int id)
    {
        SceneVerticesBase::removeVertice(id);
        normals.erase(id);
    }

    Eigen::Vector3f getNormal(unsigned int id)
    {
        if (!normals.count(id))
            throw std::out_of_range("getNormal invalid id");
        return normals[id];
    }

    void setNormal(Eigen::Vector3f &vert, unsigned int id)
    {
        if (!normals.count(id))
            throw std::out_of_range("setVertice invalid id");
        normals[id] = vert;
    }

    void transform(Sophus::SE3f newGlobalPose)
    {
        Sophus::SE3f relativePose = newGlobalPose * getPose().inverse();
        for (auto it = normals.begin(); it != normals.end(); ++it)
        {
            Eigen::Vector3f normal = it->second;
            normal = relativePose * normal;
            it->second = normal;
        }
        SceneVerticesBase::transform(newGlobalPose);
    }

private:
    std::unordered_map<unsigned int, Eigen::Vector3f> normals;
};
