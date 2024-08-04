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
        dJacMethod = DepthJacobianMethod::depthJacobian;
    };

    SceneVerticesBase(const SceneVerticesBase &other) : SceneBase(other)
    {
        vertices = other.vertices;
        rays = other.rays;
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

    void clear() override
    {
        vertices.clear();
        rays.clear();
    }

    void init(frameCPU &frame, camera &cam, dataCPU<float> &idepth, int lvl) override
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

                addVertice(vertice);
            }
        }
    }

    inline Eigen::Vector3f &getVertice(unsigned int id)
    {
        if (id >= vertices.size())
            throw std::out_of_range("getVertice invalid id");

        return vertices[id];
    }

    inline Eigen::Vector3f &getRay(unsigned int id)
    {
        if (id >= rays.size())
            throw std::out_of_range("getRay invalid id");
        return rays[id];
    }

    inline float &getDepth(unsigned int id)
    {
        if (id >= vertices.size())
            throw std::out_of_range("getDepth invalid id");

        return vertices[id](2);
    }

    unsigned int addVertice(Eigen::Vector3f &vert)
    {
        int id = vertices.size();
        vertices.push_back(vert);
        rays.push_back(vert / vert(2));
        return id;
    }

    /*
    void removeVertice(unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("removeVertice id invalid");
        vertices.erase(id);
        rays.erase(id);
    }
    */

    void setVertice(Eigen::Vector3f &vert, unsigned int id)
    {
        if (id >= vertices.size())
            throw std::out_of_range("setVertice invalid id");

        vertices[id] = vert;
        rays[id] = vert / vert(2);
    }

    void setVerticeDepth(float depth, unsigned int id)
    {
        if (id >= vertices.size())
            throw std::out_of_range("setVerticeDepth invalid id");

        // vertices[id] = depth * vertices[id] / vertices[id](2);
        vertices[id] = depth * rays[id];
    }

    float getVerticeDepth(unsigned int id)
    {
        if (id >= vertices.size())
            throw std::out_of_range("getVerticeDepth invalid id");

        return vertices[id](2);
    }

    std::vector<unsigned int> getVerticesIds() const
    {
        std::vector<unsigned int> keys;
        for (size_t it = 0; it < vertices.size(); ++it)
        {
            keys.push_back((unsigned int)it);
        }
        return keys;
    }

    void transform(Sophus::SE3f newGlobalPose) override
    {
        Sophus::SE3f relativePose = newGlobalPose * getPose().inverse();
        for (size_t it = 0; it < vertices.size(); ++it)
        {
            vertices[it] = relativePose * vertices[it];
            rays[it] = vertices[it] / vertices[it](2);
        }
        setPose(newGlobalPose);
    }

    inline DepthJacobianMethod getDepthJacMethod()
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
        switch (getDepthJacMethod())
        {
        case DepthJacobianMethod::depthJacobian:
            new_depth = param;
            break;
        case DepthJacobianMethod::idepthJacobian:
            new_depth = 1.0 / param;
            break;
        case DepthJacobianMethod::logDepthJacobian:
            new_depth = std::exp(param);
            break;
        case DepthJacobianMethod::logIdepthJacobian:
            new_depth = 1.0 / std::exp(param);
            break;
        default:
            new_depth = 0.0;
        }

        /*
        if (getDepthJacMethod() == DepthJacobianMethod::depthJacobian)
            new_depth = param;
        if (getDepthJacMethod() == DepthJacobianMethod::idepthJacobian)
            new_depth = 1.0 / param;
        if (getDepthJacMethod() == DepthJacobianMethod::logDepthJacobian)
            new_depth = std::exp(param);
        if (getDepthJacMethod() == DepthJacobianMethod::logIdepthJacobian)
            new_depth = 1.0 / std::exp(param);
        */
        // set the param (the depth in this case)
        if (new_depth > 0.01 && new_depth < 100.0)
            setVerticeDepth(new_depth, v_id);
    }

    float getDepthParam(unsigned int v_id)
    {
        float param;
        switch (getDepthJacMethod())
        {
        case DepthJacobianMethod::depthJacobian:
            param = getVerticeDepth(v_id);
            break;
        case DepthJacobianMethod::idepthJacobian:
            param = 1.0 / getVerticeDepth(v_id);
            break;
        case DepthJacobianMethod::logDepthJacobian:
            param = std::log(getVerticeDepth(v_id));
            break;
        case DepthJacobianMethod::logIdepthJacobian:
            param = -std::log(getVerticeDepth(v_id));
            break;
        default:
            param = 0.0;
        }

        /*
        float param;
        if (getDepthJacMethod() == DepthJacobianMethod::depthJacobian)
            param = getVerticeDepth(v_id);
        if (getDepthJacMethod() == DepthJacobianMethod::idepthJacobian)
            param = 1.0 / getVerticeDepth(v_id);
        if (getDepthJacMethod() == DepthJacobianMethod::logDepthJacobian)
            param = std::log(getVerticeDepth(v_id));
        if (getDepthJacMethod() == DepthJacobianMethod::logIdepthJacobian)
            param = -std::log(getVerticeDepth(v_id));
        */
        return param;
    }

private:
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3f> rays;

    int last_vertice_id;

    DepthJacobianMethod dJacMethod;
};
