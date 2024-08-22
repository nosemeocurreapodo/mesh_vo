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
        pixels = other.pixels;
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
        pixels.clear();
    }

    void init(frameCPU &frame, camera &cam, dataCPU<float> &idepth, int lvl) override
    {
        clear();
        setPose(frame.pose);

        for (float y = 0.0; y < MESH_HEIGHT; y++)
        {
            for (float x = 0.0; x < MESH_WIDTH; x++)
            {
                vec2<float> pix;
                pix(0) = (cam.width - 1) * x / (MESH_WIDTH - 1);
                pix(1) = (cam.height - 1) * y / (MESH_HEIGHT - 1);
                vec3<float> ray = cam.pixToRay(pix);
                float idph = idepth.get(pix(1), pix(0), lvl);
                if (idph == idepth.nodata)
                    continue;

                if (idph <= 0.0)
                    continue;

                vec3<float> vertice = ray / idph;

                // vertices[id] = vertice;
                // rays[id] = ray;
                // pixels[id] = pix;
                vertices.push_back(vertice);
                rays.push_back(ray);
                pixels.push_back(pix);
            }
        }
    }

    void complete(frameCPU &frame, camera &cam, dataCPU<float> &idepth, int lvl) override
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

                int size = ((cam.width - 1) / (MESH_WIDTH - 1));

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

    inline vec3<float> &getVertice(unsigned int id)
    {
#ifdef DEBUG
        if (id >= vertices.size())
            throw std::out_of_range("getVertice invalid id");
#endif
        return vertices[id];
    }

    inline vec3<float> &getRay(unsigned int id)
    {
#ifdef DEBUG
        if (id >= rays.size())
            throw std::out_of_range("getRay invalid id");
#endif
        return rays[id];
    }

    inline vec2<float> &getPix(unsigned int id)
    {
#ifdef DEBUG
        if (id >= pixels.size())
            throw std::out_of_range("getRay invalid id");
#endif
        return pixels[id];
    }

    inline float &getDepth(unsigned int id)
    {
#ifdef DEBUG
        if (id >= vertices.size())
            throw std::out_of_range("getDepth invalid id");
#endif
        return vertices[id](2);
    }

    inline std::vector<vec3<float>> &getVertices()
    {
        return vertices;
    }

    inline std::vector<vec3<float>> &getRays()
    {
        return rays;
    }

    inline std::vector<vec2<float>> &getPixels()
    {
        return pixels;
    }

    /*
    unsigned int addVertice(Eigen::Vector3f &vert)
    {
        int id = vertices.size();
        vertices.push_back(vert);
        rays.push_back(vert / vert(2));
        return id;
    }
    */

    /*
    void removeVertice(unsigned int id)
    {
        if (!vertices.count(id))
            throw std::out_of_range("removeVertice id invalid");
        vertices.erase(id);
        rays.erase(id);
    }
    */

    void setVertice(vec3<float> &vert, unsigned int id)
    {
#ifdef DEBUG
        if (id >= vertices.size())
            throw std::out_of_range("setVertice invalid id");
#endif
        vertices[id] = vert;
        rays[id] = vert / vert(2);
    }

    void setVerticeDepth(float depth, unsigned int id)
    {
#ifdef DEBUG
        if (id >= vertices.size())
            throw std::out_of_range("setVerticeDepth invalid id");
#endif
        // vertices[id] = depth * vertices[id] / vertices[id](2);
        vertices[id] = rays[id] * depth;
    }

    float getVerticeDepth(unsigned int id)
    {
#ifdef DEBUG
        if (id >= vertices.size())
            throw std::out_of_range("getVerticeDepth invalid id");
#endif
        return vertices[id](2);
    }

    std::vector<int> getVerticesIds() const
    {
        std::vector<int> keys;
        for (size_t it = 0; it < vertices.size(); ++it)
        {
            keys.push_back((int)it);
        }
        return keys;
    }

    void transform(Sophus::SE3f newGlobalPose) override
    {
        Sophus::SE3f relativePose = newGlobalPose * getPose().inverse();
        for (size_t it = 0; it < vertices.size(); ++it)
        {
            Eigen::Vector3f vert(vertices[it](0), vertices[it](1), vertices[it](2));
            vert = relativePose * vert;
            vertices[it] = vec3<float>(vert(0), vert(1), vert(2));
            rays[it] = vertices[it] / vertices[it](2);
        }
        setPose(newGlobalPose);
    }

    void project(camera cam) override
    {
        for (size_t it = 0; it < vertices.size(); ++it)
        {
            pixels[it] = cam.rayToPix(rays[it]);
        }
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
        if (new_depth > 0.000000001 && new_depth < 1000000000.0)
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
    std::vector<vec3<float>> vertices;
    std::vector<vec3<float>> rays;
    std::vector<vec2<float>> pixels;

    DepthJacobianMethod dJacMethod;
};
