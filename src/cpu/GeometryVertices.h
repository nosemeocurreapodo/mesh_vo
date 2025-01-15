#pragma once

#include "params.h"
#include "common/types2.h"
#include "common/camera.h"
#include "common/depthParam.h"
#include "cpu/frameCPU.h"

class GeometryVertices // : public SceneBase
{
public:
    GeometryVertices()
    {
    }

    void init(std::vector<vec3f> &vertices, camera cam)
    {
        assert(vertices.size() <= MAX_VERTEX_SIZE);

        for (size_t i = 0; i < MAX_VERTEX_SIZE; i++)
        {
            m_vertices[i].used = false;
        }

        for (size_t i = 0; i < vertices.size(); i++)
        {
            vec3f vertice = vertices[i];
            vec3f ray = vertice/vertice(2);
            vec2f pix = cam.rayToPix(ray);

            m_vertices[i] = vertex(vertice, ray, pix);
        }
    }

    void init(std::vector<vec2f> &texcoords, std::vector<float> idepths, camera cam)
    {
        assert(texcoords.size() == idepths.size() && texcoords.size() <= MAX_VERTEX_SIZE);

        for (size_t i = 0; i < MAX_VERTEX_SIZE; i++)
        {
            m_vertices[i].used = false;
        }

        for (size_t i = 0; i < texcoords.size(); i++)
        {
            vec2f pix = texcoords[i];
            float idph = idepths[i];

            assert(idph > 0.0);
            assert(pix(0) >= 0 && pix(0) < cam.width && pix(1) >= 0 && pix(1) < cam.height);

            vec3f ray = cam.pixToRay(pix);
            vec3f vertice = ray / idph;

            m_vertices[i] = vertex(vertice, ray, pix);
        }
    }

    vec2f meanStdDepthParam()
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

        vec2f results;
        results(0) = new_m;
        results(1) = new_s / (n - 1);

        return results;
    }

    void scaleDepthParam(vec2f affine)
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

    vertex &getVertex(unsigned int id)
    {
        assert(id >= 0 && id < MAX_VERTEX_SIZE);
        assert(m_vertices[id].used);

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
            if (m_vertices[it].used)
                keys.push_back((int)it);
        }
        return keys;
    }

    void transform(Sophus::SE3f pose)
    {
        for (int it = 0; it < MAX_VERTEX_SIZE; ++it)
        {
            if (!m_vertices[it].used)
                continue;
            vec3f ver = pose * m_vertices[it].ver;
            m_vertices[it].ver = ver;
            m_vertices[it].ray = ver/ver(2);
        }
    }

    void project(camera cam)
    {
        for (int it = 0; it < MAX_VERTEX_SIZE; ++it)
        {
            if (!m_vertices[it].used)
                continue;
            m_vertices[it].pix = cam.rayToPix(m_vertices[it].ray);
        }
    }

    std::vector<int> getParamIds()
    {
        return getVerticesIds();
    }

    void setDepthParam(float param, unsigned int v_id)
    {
        float new_depth = fromParamToDepth(param);
        assert(new_depth > 0.0);
        setVerticeDepth(new_depth, v_id);
    }

    float getDepthParam(unsigned int v_id)
    {
        return fromDepthToParam(getVerticeDepth(v_id));
    }

private:
    void setVerticeDepth(float depth, unsigned int id)
    {
        assert(id >= 0 && id < MAX_VERTEX_SIZE);
        assert(m_vertices[id].used);

        m_vertices[id].ver = m_vertices[id].ray * depth;
    }

    float getVerticeDepth(unsigned int id)
    {
        assert(id >= 0 && id < MAX_VERTEX_SIZE);
        assert(m_vertices[id].used);

        return m_vertices[id].ver(2);
    }

    vertex m_vertices[MAX_VERTEX_SIZE];
};
