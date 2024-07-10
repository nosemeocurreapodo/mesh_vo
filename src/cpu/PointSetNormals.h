#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/common.h"
#include "cpu/PointSet.h"

class PointSetNormals : public PointSet
{
public:

    PointSetNormals(const PointSetNormals &other) : PointSet(other)
    {
        normals = other.normals;
    }

    PointSetNormals &operator=(const PointSetNormals &other)
    {
        if (this != &other)
        {
            PointSet::operator=(other);
            normals = other.normals;
        }
        return *this;
    }

    void clear()
    {
        PointSet::clear();
        normals.clear();
    }

    Eigen::Vector3f getNormal(unsigned int id)
    {
        if (!normals.count(id))
            throw std::out_of_range("getNormal invalid id");
        return normals[id];
    }

    unsigned int addVertice(Eigen::Vector3f &vert)
    {
        unsigned int id = PointSet::addVertice(vert);
        normals[id] = Eigen::Vector3f(0.0, 0.0, 1.0);
        return id;
    }

    unsigned int addVertice(Eigen::Vector3f &vert, Eigen::Vector3f &normal)
    {
        unsigned int id = PointSet::addVertice(vert);
        normals[id] = normal;
        return id;
    }

    void removeVertice(unsigned int id)
    {
        PointSet::removeVertice(id);
        normals.erase(id);
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
        PointSet::transform(newGlobalPose);
    }

private:
    std::map<unsigned int, Eigen::Vector3f> normals;
};
