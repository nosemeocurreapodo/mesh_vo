#pragma once
#include <memory>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/common.h"
#include "common/camera.h"
#include "cpu/SceneBase.h"
#include "cpu/Shapes.h"
#include "cpu/frameCPU.h"

#define MAX_VERTEX_SIZE 4096 //for a 64*64 mesh

class SceneVertices // : public SceneBase
{
public:

    SceneVertices()
    {
        m_pose = Sophus::SE3f();
        m_cam = camera(0.0, 0.0, 0.0, 0.0, 0, 0);
    }

    /*
    SceneVertices(int size)
    {
        _vertices = new (std::nothrow) vertex[size];
        _pose = Sophus::SE3f();
        _cam = camera(0.0, 0.0, 0.0, 0.0, 0, 0);
        _size = size;
    };
    */

    /*
    SceneVertices(const SceneVertices &other) // : SceneBase(other)
    {
        m_pose = other.m_pose;
        m_cam = other.m_cam;
        m_verticesBufferSize = other.m_verticesBufferSize;

        if(m_vertices != nullptr)
            deleteBuffer();
        createBuffer(m_verticesBufferSize);
        std::memcpy(m_vertices, other.m_vertices, sizeof(vertex) *m_verticesBufferSize);
    }
    */

    /*
    SceneVertices &operator=(const SceneVertices &other)
    {
        if (this != &other)
        {
            m_pose = other.m_pose;
            m_cam = other.m_cam;
            m_verticesBufferSize = other.m_verticesBufferSize;

            //if(m_vertices != nullptr)
            //    deleteBuffer();
            //createBuffer(m_verticesBufferSize);
            std::memcpy(m_vertices, other.m_vertices, sizeof(vertex) *m_verticesBufferSize);
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

    void init(camera cam, Sophus::SE3f pose, std::vector<vec3<float>> &vertices)
    {
        m_pose = pose;
        m_cam = cam;
        //m_verticesBufferSize = vertices.size();
        
        //if(m_vertices != nullptr)
        //    deleteBuffer();
        //createBuffer(m_verticesBufferSize);

        for (int i = 0; i < vertices.size(); i++)
        {
            if(i >= MAX_VERTEX_SIZE)
                break;
            vec3<float> vertice = vertices[i];
            vec3<float> ray = vertice / vertice(2);
            float idph = 1.0 / vertice(2);
            vec2<float> pix = cam.rayToPix(ray);
            float iv = 0.1;

            if (idph <= 0.0)
                continue;

            m_vertices[i] = vertex(vertice, ray, pix, iv);
        }
    }

    void init(camera cam, Sophus::SE3f pose, std::vector<vec2<float>> &texcoords, std::vector<float> &idepths)
    {
        m_cam = cam;
        m_pose = pose;
        //m_verticesBufferSize = texcoords.size();

        //if(m_vertices != nullptr)
        //    deleteBuffer();
        //createBuffer(m_verticesBufferSize);

        for (int i = 0; i < texcoords.size(); i++)
        {
            if(i >= MAX_VERTEX_SIZE)
                break;
            vec2<float> pix = texcoords[i];
            float idph = idepths[i];

            vec3<float> ray = cam.pixToRay(pix);
            float iv = 0.1;

            if (idph <= 0.0)
                continue;

            vec3<float> vertice = ray / idph;

            m_vertices[i] = vertex(vertice, ray, pix, iv);
        }
    }

    void init(camera cam, Sophus::SE3f pose, dataCPU<float> &idepth, dataCPU<float> &ivar, int lvl)
    {
        m_cam = cam;
        m_pose = pose;
        //m_verticesBufferSize = MESH_WIDTH*MESH_HEIGHT;

        //if(m_vertices != nullptr)
        //    deleteBuffer();
        //createBuffer(m_verticesBufferSize);

        int i = 0;
        for (float y = 0.0; y < MESH_HEIGHT; y++)
        {
            for (float x = 0.0; x < MESH_WIDTH; x++)
            {
                if(i >= MAX_VERTEX_SIZE)
                    return;

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

                m_vertices[i] = vertex(vertice, ray, pix, iv);
                i++;
            }
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

    vec2<float> meanStdDepthParam()
    {
        float old_m = 0;
        float new_m = 0;
        float old_s = 0;
        float new_s = 0;
        int n = 0;

        for (int i = 0; i < MAX_VERTEX_SIZE; i++)
        {
            if (!m_vertices[i].used)
                continue;

            float param = getDepthParam(i);

            n++;

            if (n == 1)
            {
                old_m = param;
                new_m = param;
                old_s = 0;
            }
            else
            {
                new_m = old_m + (param - old_m) / n;
                new_s = old_s + (param - old_m) * (param - new_m);
                old_m = new_m;
                old_s = new_s;
            }
        }

        vec2<float> results;
        results(0) = new_m;
        results(1) = new_s / (n - 1);

        return results;
    }

    void scaleDepthParam(vec2<float> affine)
    {
        for (int i = 0; i < MAX_VERTEX_SIZE; i++)
        {
            if (!m_vertices[i].used)
                continue;
            float param = getDepthParam(i);
            float new_param = (param - affine(1)) / affine(0);
            setDepthParam(new_param, i);
        }
    }

    inline vertex &getVertex(unsigned int id)
    {
#ifdef DEBUG
        if (id >= MAX_VERTEX_SIZE)
            throw std::out_of_range("getVertex invalid id");
#endif
        return m_vertices[id];
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

    std::vector<int> getVerticesIds() const
    {
        std::vector<int> keys;
        for (int it = 0; it < MAX_VERTEX_SIZE; ++it)
        {
            if(m_vertices[it].used)
                keys.push_back((int)it);
        }
        return keys;
    }

    void transform(camera cam, Sophus::SE3f pose)
    {
        bool poseUpdated = false;
        if (!(m_pose.translation() == pose.translation()) && !(m_pose.unit_quaternion() == pose.unit_quaternion()) )
        {
            poseUpdated = true;
            Sophus::SE3f relativePose = pose * m_pose.inverse();
            m_pose = pose;
            for (int it = 0; it < MAX_VERTEX_SIZE; ++it)
            {
                if (!m_vertices[it].used)
                    continue;
                Eigen::Vector3f _vert = relativePose * Eigen::Vector3f(m_vertices[it].ver(0), m_vertices[it].ver(1), m_vertices[it].ver(2));
                vec3<float> ver(_vert(0), _vert(1), _vert(2));
                vec3<float> ray = ver / ver(2);

                m_vertices[it].ver = ver;
                m_vertices[it].ray = ray;
            }
        }

        if (!(m_cam == cam) || poseUpdated)
        {
            m_cam = cam;
            for (int it = 0; it < MAX_VERTEX_SIZE; ++it)
            {
                if(!m_vertices[it].used)
                    continue;
                m_vertices[it].pix = m_cam.rayToPix(m_vertices[it].ray);
            }
        }
    }

    std::vector<int> getParamIds()
    {
        return getVerticesIds();
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

    void setParamWeight(float weight, unsigned int v_id)
    {
        m_vertices[v_id].weight = weight;
    }

    float getParamWeight(unsigned int v_id)
    {
        return m_vertices[v_id].weight;
    }

private:

    /*
    void deleteBuffer()
    {
        delete m_vertices;
        m_vertices = nullptr;
    }

    void createBuffer(int size)
    {
        m_vertices = new (std::nothrow) vertex[size];
    }
    */

    void setVerticeDepth(float depth, unsigned int id)
    {
#ifdef DEBUG
        if (id >= MAX_VERTEX_SIZE)
            throw std::out_of_range("setVerticeDepth invalid id");
#endif
        m_vertices[id].ver = m_vertices[id].ray * depth;
    }

    float getVerticeDepth(unsigned int id)
    {
#ifdef DEBUG
        if (id >= MAX_VERTEX_SIZE)
            throw std::out_of_range("getVerticeDepth invalid id");
#endif
        return m_vertices[id].ver(2);
    }

    vertex m_vertices[MAX_VERTEX_SIZE]; 
    Sophus::SE3f m_pose;
    camera m_cam;
    //int m_verticesBufferSize;
};
