#pragma once

#include "MeshTexCoordsCPU.h"

class MeshNormalsCPU : public MeshTexCoordsCPU
{
public:
    MeshNormalsCPU() : MeshTexCoordsCPU()
    {
    }

    MeshNormalsCPU(const MeshNormalsCPU &other) : MeshTexCoordsCPU(other)
    {
        normals = other.normals;
    }

    MeshNormalsCPU &operator=(const MeshNormalsCPU &other)
    {
        if (this != &other)
        {
            MeshTexCoordsCPU::operator=(other); // Call base class assignment operator
            normals = other.normals;   // Copy texcoord member
        }
        return *this;
    }

    void clear()
    {
        MeshTexCoordsCPU::clear();
        normals.clear();
    }

    unsigned int addVertice(Eigen::Vector3f &vert)
    {
        unsigned int id = MeshTexCoordsCPU::addVertice(vert);
        normals[id] = Eigen::Vector3f(0.0, 0.0, 1.0);
    }

    void transform(Sophus::SE3f newGlobalPose)
    {
        MeshTexCoordsCPU::transform(newGlobalPose);

        Sophus::SE3f relativePose = newGlobalPose * globalPose.inverse();
        for (auto it = normals.begin(); it != normals.end(); ++it)
        {
            Eigen::Vector3f normal = it->second;
            normal = relativePose * normal;
            it->second = normal;
        }
    }

private:
    std::map<unsigned int, Eigen::Vector3f> normals;
};
