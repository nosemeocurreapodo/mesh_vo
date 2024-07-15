#pragma once
#include <memory>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/common.h"
#include "common/camera.h"
#include "cpu/SceneBase.h"
#include "cpu/Shapes.h"
#include "cpu/frameCPU.h"

class SceneVerticesBase : public SceneBase
{
public:
    SceneVerticesBase() : SceneBase()
    {
        last_vertice_id = 0;
        dJacMethod = DepthJacobianMethod::depthJacobian;
    };

    SceneVerticesBase(const SceneVerticesBase &other) : SceneBase(other)
    {
        vertices = other.vertices;
        last_vertice_id = other.last_vertice_id;
        dJacMethod = other.dJacMethod;
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

    /*
     std::unique_ptr<SceneBase> clone() const override
     {
         return std::make_unique<ScenePoints>(*this);
     }
     */

    void init(frameCPU &frame, camera &cam, dataCPU<float> &idepth, int lvl) override
    {
        clear();

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
            }
        }
        setPose(frame.pose);
    }

    void clear() override
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

    void transform(Sophus::SE3f newGlobalPose) override
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

    DepthJacobianMethod getDepthJacMethod()
    {
        return dJacMethod;
    }

    void setDepthJackMethod(DepthJacobianMethod method)
    {
        dJacMethod = method;
    }

    void setDepthParam(float param, unsigned int v_id)
    {
        float new_depth;
        if (getDepthJacMethod() == DepthJacobianMethod::depthJacobian)
            new_depth = param;
        if (getDepthJacMethod() == DepthJacobianMethod::idepthJacobian)
            new_depth = 1.0 / param;
        if (getDepthJacMethod() == DepthJacobianMethod::logDepthJacobian)
            new_depth = std::exp(param);
        if (getDepthJacMethod() == DepthJacobianMethod::logIdepthJacobian)
            new_depth = 1.0 / std::exp(param);

        // set the param (the depth in this case)
        if (new_depth > 0.01 && new_depth < 100.0)
            setVerticeDepth(new_depth, v_id);
    }

    float getDepthParam(unsigned int v_id)
    {
        float param;
        if (getDepthJacMethod() == DepthJacobianMethod::depthJacobian)
            param = getVerticeDepth(v_id);
        if (getDepthJacMethod() == DepthJacobianMethod::idepthJacobian)
            param = 1.0 / getVerticeDepth(v_id);
        if (getDepthJacMethod() == DepthJacobianMethod::logDepthJacobian)
            param = std::log(getVerticeDepth(v_id));
        if (getDepthJacMethod() == DepthJacobianMethod::logIdepthJacobian)
            param = std::log(1.0 / getVerticeDepth(v_id));

        return param;
    }

private:
    std::map<unsigned int, Eigen::Vector3f> vertices;

    Sophus::SE3f globalPose;
    int last_vertice_id;

    DepthJacobianMethod dJacMethod;
};
