#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/common.h"
#include "cpu/PointSet.h"

class PointSetRayIdepth : public PointSet
{
public:
    Eigen::Vector3f getVertice(unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("getVertice invalid id");
        // always return in cartesian
        return rayIdepthToCartesian(vertices[id]);
    }

    unsigned int addVertice(Eigen::Vector3f &vert)
    {
        // the input vertice is always in cartesian
        last_vertice_id++;
        if (vertices.count(last_vertice_id))
            throw std::out_of_range("addVertice id already exist");
        vertices[last_vertice_id] = cartesianToRayIdepth(vert);
        return last_vertice_id;
    }

    void setVertice(Eigen::Vector3f &vert, unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("setVertice invalid id");
        vertices[id] = cartesianToRayIdepth(vert);
    }

    void setVerticeDepth(float depth, unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("setVerticeDepth invalid id");
        vertices[id](2) = 1.0 / depth;
    }

    float getVerticeDepth(unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("getVerticeDepth invalid id");
        return 1.0 / vertices[id](2);
    }

    void transform(Sophus::SE3f newGlobalPose)
    {
        Sophus::SE3f relativePose = newGlobalPose * globalPose.inverse();
        for (auto it = vertices.begin(); it != vertices.end(); ++it)
        {
            Eigen::Vector3f pos = it->second;
            pos = rayIdepthToCartesian(pos);
            pos = relativePose * pos;
            pos = cartesianToRayIdepth(pos);
            it->second = pos;
        }
        globalPose = newGlobalPose;
    }

    void computeTexCoords(camera &cam)
    {
        texcoords.clear();
        std::vector<unsigned int> ids = getVerticesIds();
        for (auto id : ids)
        {
            Eigen::Vector3f ray;

            ray = getVertice(id);
            ray(2) = 1.0;

            Eigen::Vector2f pix = cam.rayToPix(ray);

            texcoords[id] = pix;
        }
    }

private:
    void toCartesian()
    {

        for (auto it = vertices.begin(); it != vertices.end(); ++it)
        {
            it->second = rayIdepthToCartesian(it->second);
        }
    }
};
