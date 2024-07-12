#pragma once
#include <memory>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/common.h"
#include "common/camera.h"
#include "cpu/Polygon.h"
#include "cpu/frameCPU.h"

class PointSet
{
public:
    PointSet()
    {
        last_vertice_id = 0;
        jacMethod = MapJacobianMethod::depthJacobian;
    };

    PointSet(const PointSet &other)
    {
        vertices = other.vertices;
        globalPose = other.globalPose;
        last_vertice_id = other.last_vertice_id;
        jacMethod = other.jacMethod;
    }
    /*
    PointSet &operator=(const PointSet &other)
    {
        if (this != &other)
        {
            vertices = other.vertices;
            globalPose = other.globalPose;
            last_vertice_id = other.last_vertice_id;
        }
        return *this;
    }
    */

    virtual std::unique_ptr<PointSet> clone() const = 0;
    virtual void init(frameCPU &frame, camera &cam, dataCPU<float> &idepth, int lvl) = 0;
    virtual std::vector<unsigned int> getPolygonsIds() const = 0;
    virtual std::unique_ptr<Polygon> getPolygon(unsigned int polId) = 0;
    // virtual std::vector<unsigned int> getPolygonVerticesIds(unsigned int id) = 0;
    virtual std::vector<unsigned int> getPolygonParamsIds(unsigned int polId) = 0;
    virtual void setParam(float param, unsigned int paramId) = 0;
    virtual float getParam(unsigned int paramId) = 0;

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
    }

    void setVertice(Eigen::Vector3f &vert, unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("setVertice invalid id");
        vertices[id] = vert;
    }

    void setPose(Sophus::SE3f &pose)
    {
        globalPose = pose;
    }

    Sophus::SE3f getPose() const
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

    std::vector<unsigned int> getVerticesIds() const
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
            pos = relativePose * pos;
            it->second = pos;
        }
        globalPose = newGlobalPose;
    }

    MapJacobianMethod getJacMethod()
    {
        return jacMethod;
    }

    void setJackMethod(MapJacobianMethod method)
    {
        jacMethod = method;
    }

private:
    std::map<unsigned int, Eigen::Vector3f> vertices;

    Sophus::SE3f globalPose;
    int last_vertice_id;

    MapJacobianMethod jacMethod;
};
