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
};

template <int size>
class SceneVertices // : public SceneBase
{
public:
    SceneVertices()
    {
        _pose = Sophus::SE3f();
        _cam = camera(0.0, 0.0, 0.0, 0.0, 0, 0);
    };

    SceneVertices(const SceneVertices &other) // : SceneBase(other)
    {
        _vertices = other._vertices;
        _pose = other._pose;
        _cam = other._cam;
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

    void init(camera cam, Sophus::SE3f pose, std::vector<vec3<float>> &vertices)
    {
        _pose = pose;
        _cam = cam;
        for (int i = 0; i < size; i++)
        {
            if (i < vertices.size())
            {
                vec3<float> vertice = vertices[i];
                vec3<float> ray = vertice / vertice(2);
                float idph = 1.0 / vertice(2);
                vec2<float> pix = cam.rayToPix(ray);
                float iv = 0.1;

                if (idph <= 0.0)
                    continue;

                _vertices[i] = vertex(vertice, ray, pix, iv);
            }
            else
            {
                _vertices[i].used = false;
            }
        }
    }

    void init(camera cam, Sophus::SE3f pose, std::vector<vec2<float>> &texcoords, std::vector<float> &idepths)
    {
        _cam = cam;
        _pose = pose;

        for (int i = 0; i < size; i++)
        {
            if (i < texcoords.size())
            {
                vec2<float> pix = texcoords[i];
                float idph = idepths[i];

                vec3<float> ray = cam.pixToRay(pix);
                float iv = 0.1;

                if (idph <= 0.0)
                    continue;

                vec3<float> vertice = ray / idph;

                _vertices[i] = vertex(vertice, ray, pix, iv);
            }
            else
            {
                _vertices[i].used = false;
            }
        }
    }

    void init(camera cam, Sophus::SE3f pose, dataCPU<float> &idepth, dataCPU<float> &ivar, int lvl)
    {
        _cam = cam;
        _pose = pose;
        int i = 0;
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

                _vertices[i] = vertex(vertice, ray, pix, iv);
                i++;
            }
        }
        for (int j = i; j < size; j++)
        {
            _vertices[j].used = false;
        }
    }

    /*
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
    */

    vec2<float> meanStdDepth()
    {
        float old_m = 0;
        float new_m = 0;
        float old_s = 0;
        float new_s = 0;
        int n = 0;

        for (int i = 0; i < size; i++)
        {
            if (!_vertices[i].used)
                continue;

            float depth = _vertices[i].ver(2);

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
        for (int i = 0; i < size; i++)
        {
            if (!_vertices[i].used)
                continue;
            vec3<float> vertice = _vertices[i].ver;
            float depth = vertice(2);
            float new_depth = (depth - affine(1)) / affine(0);
            vec3<float> new_vertice = vertice * (new_depth / depth);
            _vertices[i].ver = new_vertice;
        }
    }

    inline vertex &getVertex(unsigned int id)
    {
#ifdef DEBUG
        if (id >= vertices.size())
            throw std::out_of_range("getVertice invalid id");
#endif
        return _vertices[id];
    }

    /*
    unsigned int addVertice(vec3<float> vert)
    {
        int id = vertices.size();
        vertices.push_back(vert);
        rays.push_back(vert / vert(2));
        pixels.push_back(vec2<float>(0.0, 0.0));
        weights.push_back(1.0);
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

    void setVerticeDepth(float depth, unsigned int id)
    {
#ifdef DEBUG
        if (id >= vertices.size())
            throw std::out_of_range("setVerticeDepth invalid id");
#endif
        // vertices[id] = depth * vertices[id] / vertices[id](2);
        _vertices[id].vert = _vertices[id].ray * depth;
    }

    float getVerticeDepth(unsigned int id)
    {
#ifdef DEBUG
        if (id >= vertices.size())
            throw std::out_of_range("getVerticeDepth invalid id");
#endif
        return _vertices[id].vert(2);
    }

    std::vector<int> getVerticesIds() const
    {
        std::vector<int> keys;
        for (size_t it = 0; it < size; ++it)
        {
            keys.push_back((int)it);
        }
        return keys;
    }

    void transform(camera cam, Sophus::SE3f pose)
    {
        if (!(_pose.translation() == pose.translation()) && !(_pose.unit_quaternion() == pose.unit_quaternion()) )
        {
            Sophus::SE3f relativePose = _pose * pose.inverse();
            _pose = pose;
            for (size_t it = 0; it < size; ++it)
            {
                if (!_vertices[it].used)
                    continue;
                Eigen::Vector3f _vert = relativePose * Eigen::Vector3f(_vertices[it].ver(0), _vertices[it].ver(1), _vertices[it].ver(2));
                vec3<float> vert(_vert(0), _vert(1), _vert(2));
                vec3<float> ray = vert / vert(2);

                _vertices[it].vert = vert;
                _vertices[it].ray = ray;
            }
        }

        if (!(_cam == cam))
        {
            _cam = cam;
            for (size_t it = 0; it < size; ++it)
            {
                if(!_vertices[it].used)
                    continue;
                _vertices[it].pix = _cam.rayToPix(_vertices[it].ray);
            }
        }
    }

    void setDepthParam(float param, unsigned int v_id)
    {
        float new_depth = fromParamToDepth(param);
        setVerticeDepth(new_depth, v_id);
    }

    float getDepthParam(unsigned int v_id)
    {
        return fromDepthToParam(getVerticeDepth(v_id));
    }

    /*
    bool isVertInWindow(window &win, int v_id)
    {
        if (win.isPixInWindow(getPix(v_id)))
            return true;
        return false;
    }
    */

private:
    vertex _vertices[size];
    Sophus::SE3f _pose;
    camera _cam;
};
