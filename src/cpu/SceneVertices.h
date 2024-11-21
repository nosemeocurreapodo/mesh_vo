#pragma once
#include <memory>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/common.h"
#include "common/camera.h"
#include "cpu/SceneBase.h"
#include "cpu/Shapes.h"
#include "cpu/frameCPU.h"

struct vertex
{
    vertex()
    {
        used = false;
    }

    vertex(vec3<float> v, vec3<float> r, vec2<float> p, float w)
    {
        ver = v;
        ray = r;
        pix = p;
        weight = w;
        used = true;
    }

    vec3<float> ver;
    vec3<float> ray;
    vec2<float> pix;
    float weight;
    bool used;
}

class SceneVertices // : public SceneBase
{
public:

    SceneVertices() {};// : SceneBase() {
                    //      };

    SceneVertices(const SceneVertices &other)// : SceneBase(other)
    {
        vertices = other.vertices;
    }

    /*
     std::unique_ptr<SceneBase> clone() const override
     {
         return std::make_unique<ScenePoints>(*this);
     }
     */

    /*
    void clear()
    {
        vertices.clear();
        rays.clear();
        pixels.clear();
        weights.clear();
    }
    */

    void init(camera cam, std::vector<vec3<float>> &new_vertices)
    {
        for (int i = 0; i < (int)new_vertices.size(); i++)
        {
            vec3<float> vertice = new_vertices[i];
            vec3<float> ray = vertice / vertice(2);
            float idph = 1.0 / vertice(2);
            vec2<float> pix = cam.rayToPix(ray);
            float iv = 0.1;

            if (idph <= 0.0)
                continue;

            vertices[i] = vertex(vertice, ray, pix, iv);
        }
    }

    void init(camera cam, std::vector<vec2<float>> &texcoords, std::vector<float> &idepths)
    {
        for (int i = 0; i < (int)texcoords.size(); i++)
        {
            vec2<float> pix = texcoords[i];
            float idph = idepths[i];

            vec3<float> ray = cam.pixToRay(pix);
            float iv = 0.1;

            if (idph <= 0.0)
                continue;

            vec3<float> vertice = ray / idph;

            // vertices[id] = vertice;
            // rays[id] = ray;
            // pixels[id] = pix;
            vertices.push_back(vertice);
            rays.push_back(ray);
            pixels.push_back(pix);
            weights.push_back(iv);
        }
    }

    void init(camera cam, dataCPU<float> &idepth, dataCPU<float> &ivar, int lvl)
    {
        clear();

        for (float y = 0.0; y < MESH_HEIGHT; y++)
        {
            for (float x = 0.0; x < MESH_WIDTH; x++)
            {
                vec2<float> pix;
                pix(0) = (cam.width - 1) * x / (MESH_WIDTH - 1);
                pix(1) = (cam.height - 1) * y / (MESH_HEIGHT - 1);
                vec3<float> ray = cam.pixToRay(pix);
                float idph = idepth.get(pix(1), pix(0), lvl);
                float iv = ivar.get(pix(1), pix(0), lvl);
                if (idph == idepth.nodata || iv == ivar.nodata)
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
                weights.push_back(iv);
            }
        }
    }

    void complete(frameCPU &frame, camera &cam, dataCPU<float> &idepth, int lvl)
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

                int size = ((cam.width - 1) / (MESH_WIDTH - 1)) / 2;

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
                weights.push_back(initialIvar());

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

    vec2<float> meanStdDepth()
    {
        float old_m = 0;
        float new_m = 0;
        float old_s = 0;
        float new_s = 0;
        int n = 0;

        std::vector<int> vertsIds = getVerticesIds();

        for (int vertId : vertsIds)
        {
            float depth = vertices[vertId](2);

            n++;

            if (n == 1)
            {
                old_m = depth;
                new_m = depth;
                old_s = 0;
            }
            else
            {
                new_m = old_m + (depth - old_m) / n;
                new_s = old_s + (depth - old_m) * (depth - new_m);
                old_m = new_m;
                old_s = new_s;
            }
        }

        vec2<float> results;
        results(0) = new_m;
        results(1) = new_s / (n - 1);

        return results;
    }

    void scaleDepth(vec2<float> affine)
    {
        std::vector<int> vertsIds = getVerticesIds();

        for (int vertId : vertsIds)
        {
            vec3<float> vertice = vertices[vertId];
            float depth = vertice(2);
            float new_depth = (depth - affine(1)) / affine(0);
            vec3<float> new_vertice = vertice * (new_depth / depth);
            vertices[vertId] = new_vertice;
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

    inline float &getWeight(unsigned int id)
    {
#ifdef DEBUG
        if (id >= weights.size())
            throw std::out_of_range("getDepth invalid id");
#endif
        return weights[id];
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

    unsigned int addVertice(vec3<float> vert)
    {
        int id = vertices.size();
        vertices.push_back(vert);
        rays.push_back(vert / vert(2));
        pixels.push_back(vec2<float>(0.0, 0.0));
        weights.push_back(1.0);
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

    void setWeight(float w, unsigned int id)
    {
#ifdef DEBUG
        if (id >= weights.size())
            throw std::out_of_range("setVerticeDepth invalid id");
#endif
        weights[id] = w;
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

    void transform(Sophus::SE3f relativePose)
    {
        // Sophus::SE3f relativePose = newGlobalPose * getPose().inverse();
        for (size_t it = 0; it < vertices.size(); ++it)
        {
            Eigen::Vector3f vert(vertices[it](0), vertices[it](1), vertices[it](2));
            vert = relativePose * vert;
            vertices[it] = vec3<float>(vert(0), vert(1), vert(2));
            rays[it] = vertices[it] / vertices[it](2);
        }
        // setPose(newGlobalPose);
    }

    void project(camera cam)
    {
        for (size_t it = 0; it < vertices.size(); ++it)
        {
            pixels[it] = cam.rayToPix(rays[it]);
        }
    }

    void setDepthParam(float param, unsigned int v_id)
    {
        float new_depth = fromParamToDepth(param);
        // set the param (the depth in this case)
#ifdef DEBUG
        if (new_depth <= 0.0)
        {
            std::cout << "setDepthParam new_depth " << new_depth << std::endl;
        }
#endif
        setVerticeDepth(new_depth, v_id);
    }

    float getDepthParam(unsigned int v_id)
    {
        return fromDepthToParam(getVerticeDepth(v_id));
    }

    bool isVertInWindow(window &win, int v_id)
    {
        if (win.isPixInWindow(getPix(v_id)))
            return true;
        return false;
    }

private:
    //vec3<float> *vertices;
    //vec3<float> *rays;
    //vec2<float> *pixels;
    //float *weights;

    vertex vertices[MAX_VERTEX];
};
