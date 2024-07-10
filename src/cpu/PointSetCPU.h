#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/common.h"

enum VerticeRepresentation
{
    cartesian,
    rayIdepth
};

class PointSetCPU
{
public:
    PointSetCPU()
    {
        representation = cartesian;
        last_vertice_id = 0;
    };

    PointSetCPU(const PointSetCPU &other)
    {
        vertices = other.vertices;
        globalPose = other.globalPose;
        representation = other.representation;
        last_vertice_id = other.last_vertice_id;
    }

    PointSetCPU &operator=(const PointSetCPU &other)
    {
        if (this != &other)
        {
            vertices = other.vertices;
            globalPose = other.globalPose;
            representation = other.representation;
            last_vertice_id = other.last_vertice_id;
        }
        return *this;
    }

    void clear()
    {
        vertices.clear();
        globalPose = Sophus::SE3f();
        last_vertice_id = 0;
    }

    Eigen::Vector3f getVertice(unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("getVertice invalid id");
        // always return in cartesian
        if (representation == rayIdepth)
            return rayIdepthToCartesian(vertices[id]);
        return vertices[id];
    }

    unsigned int addVertice(Eigen::Vector3f &vert)
    {
        // the input vertice is always in cartesian
        last_vertice_id++;
        if (vertices.count(last_vertice_id))
            throw std::out_of_range("addVertice id already exist");
        if (representation == rayIdepth)
            vertices[last_vertice_id] = cartesianToRayIdepth(vert);
        if (representation == cartesian)
            vertices[last_vertice_id] = vert;
        return last_vertice_id;
    }

    void removeVertice(unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("removeVertice id invalid");
        vertices.erase(id);
    }

    void setVertice(Eigen::Vector3f &vert, unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("setVertice invalid id");
        if (representation == rayIdepth)
            vertices[id] = cartesianToRayIdepth(vert);
        if (representation == cartesian)
            vertices[id] = vert;
    }

    void setPose(Sophus::SE3f &pose)
    {
        globalPose = pose;
    }

    void setVerticeDepth(float depth, unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("setVerticeDepth invalid id");
        if (representation == rayIdepth)
            vertices[id](2) = 1.0 / depth;
        if (representation == cartesian)
        {
            vertices[id] = depth * vertices[id] / vertices[id](2);
        }
    }

    float getVerticeDepth(unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("getVerticeDepth invalid id");
        if (representation == rayIdepth)
            return 1.0 / vertices[id](2);
        if (representation == cartesian)
            return vertices[id](2);
    }

    std::vector<unsigned int> getVerticesIds()
    {
        std::vector<unsigned int> keys;
        for (auto it = vertices.begin(); it != vertices.end(); ++it)
        {
            keys.push_back(it->first);
        }
        return keys;
    }

    void transform(Sophus::SE3f newGlobalPose)
    {
        Sophus::SE3f relativePose = newGlobalPose * globalPose.inverse();
        for (auto it = vertices.begin(); it != vertices.end(); ++it)
        {
            Eigen::Vector3f pos = it->second;
            if (representation == rayIdepth)
                pos = rayIdepthToCartesian(pos);
            pos = relativePose * pos;
            if (representation == rayIdepth)
                pos = cartesianToRayIdepth(pos);
            it->second = pos;
        }
        globalPose = newGlobalPose;
    }

    VerticeRepresentation representation;

private:
    void toRayIdepth()
    {
        if (representation == cartesian)
        {
            for (auto it = vertices.begin(); it != vertices.end(); ++it)
            {
                it->second = cartesianToRayIdepth(it->second);
            }
        }

        representation = rayIdepth;
    }

    void toCartesian()
    {
        if (representation == rayIdepth)
        {
            for (auto it = vertices.begin(); it != vertices.end(); ++it)
            {
                it->second = rayIdepthToCartesian(it->second);
            }
        }
        representation = cartesian;
    }

    std::map<unsigned int, Eigen::Vector3f> vertices;
    Sophus::SE3f globalPose;
    int last_vertice_id;
};
