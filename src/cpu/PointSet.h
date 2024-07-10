#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/common.h"
#include "common/camera.h"

class PointSet
{
public:
    PointSet()
    {
        last_vertice_id = 0;
    };

    PointSet(const PointSet &other)
    {
        vertices = other.vertices;
        texcoords = other.texcoords;
        globalPose = other.globalPose;
        last_vertice_id = other.last_vertice_id;
    }

    PointSet &operator=(const PointSet &other)
    {
        if (this != &other)
        {
            vertices = other.vertices;
            texcoords = other.texcoords;
            globalPose = other.globalPose;
            last_vertice_id = other.last_vertice_id;
        }
        return *this;
    }

    void clear()
    {
        vertices.clear();
        texcoords.clear();
        globalPose = Sophus::SE3f();
        last_vertice_id = 0;
    }

    Eigen::Vector3f getVertice(unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("getVertice invalid id");
        return vertices[id];
    }

    unsigned int addVertice(Eigen::Vector3f &vert)
    {
        // the input vertice is always in cartesian
        last_vertice_id++;
        if (vertices.count(last_vertice_id))
            throw std::out_of_range("addVertice id already exist");
        vertices[last_vertice_id] = vert;
        return last_vertice_id;
    }

    void removeVertice(unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("removeVertice id invalid");
        vertices.erase(id);
        texcoords.erase(id);
    }

    void setVertice(Eigen::Vector3f &vert, unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("setVertice invalid id");
        vertices[id] = vert;
    }

    Eigen::Vector2f getTexCoord(unsigned int id)
    {
        if (!texcoords.count(id))
            throw std::out_of_range("getTexCoord id invalid");
        return texcoords[id];
    }

    void setPose(Sophus::SE3f &pose)
    {
        globalPose = pose;
    }

    Sophus::SE3f getPose()
    {
        return globalPose;
    }

    void setVerticeDepth(float depth, unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("setVerticeDepth invalid id");
        vertices[id] = depth * vertices[id] / vertices[id](2);
    }

    float getVerticeDepth(unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("getVerticeDepth invalid id");
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

    void transform(Sophus::SE3f newGlobalPose, camera cam)
    {
        Sophus::SE3f relativePose = newGlobalPose * globalPose.inverse();
        for (auto it = vertices.begin(); it != vertices.end(); ++it)
        {
            Eigen::Vector3f pos = it->second;
            pos = relativePose * pos;
            Eigen::Vector3f ray = pos / pos(2);
            Eigen::Vector2f pix = cam.rayToPix(ray);
            texcoords[it->first] = pix;
            it->second = pos;
        }
        globalPose = newGlobalPose;
    }

private:
    std::map<unsigned int, Eigen::Vector3f> vertices;
    std::map<unsigned int, Eigen::Vector2f> texcoords;

    Sophus::SE3f globalPose;
    int last_vertice_id;
};
