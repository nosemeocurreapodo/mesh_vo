#pragma once

#include "params.h"
#include "common/camera.h"
#include "common/depthParam.h"
#include "cpu/frameCPU.h"

class GeometryVertices // : public SceneBase
{
public:
    GeometryVertices()
    {
    }

    void init(std::vector<vec3f> &vertices, std::vector<float> &weights, cameraType cam)
    {
        assert(vertices.size() == weights.size());
        assert(vertices.size() <= mesh_vo::max_vertex_size);

        for (size_t i = 0; i < mesh_vo::max_vertex_size; i++)
        {
            m_vertices[i].used = false;
        }

        for (size_t i = 0; i < vertices.size(); i++)
        {
            vec3f vertice = vertices[i];

            assert(vertice(2) > 0.0);
            assert(!std::isnan(vertice(2)));
            assert(!std::isinf(vertice(2)));

            vec3f ray = vertice / vertice(2);
            vec2f pix = cam.rayToPix(ray);
            float weight = weights[i];

            m_vertices[i] = vertex(vertice, ray, pix, weight);
        }
    }

    void init(std::vector<vec3f> &rays, std::vector<float> depths, std::vector<float> &weights, cameraType cam)
    {
        assert(rays.size() == depths.size());
        assert(rays.size() == weights.size());
        assert(rays.size() <= mesh_vo::max_vertex_size);

        for (size_t i = 0; i < mesh_vo::max_vertex_size; i++)
        {
            m_vertices[i].used = false;
        }

        for (size_t i = 0; i < rays.size(); i++)
        {
            vec3f ray = rays[i];
            float dph = depths[i];
            float weight = weights[i];
            vec2f pix = cam.rayToPix(ray);

            //assert(cam.isPixVisible(pix));
            //assert(dph > 0.0);
            assert(!std::isnan(dph));
            assert(!std::isinf(dph));

            vec3f vertice = ray * dph;

            m_vertices[i] = vertex(vertice, ray, pix, weight);
        }
    }

    vec2f meanStdDepthParams()
    {
        float old_m = 0;
        float new_m = 0;
        float old_s = 0;
        float new_s = 0;
        int n = 0;

        for (int i = 0; i < mesh_vo::max_vertex_size; i++)
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
        results(1) = sqrt(new_s / (n - 1));

        return results;
    }


    vec2f minMaxDepthVertices()
    {
        float min = 0;
        float max = 0;
        int n = 0;

        for (int i = 0; i < mesh_vo::max_vertex_size; i++)
        {
            if (!m_vertices[i].used)
                continue;

            float depth = getVerticeDepth(i);

            n++;

            if (n == 1)
            {
                min = depth;
                max = depth;
            }
            else
            {
                if (depth < min)
                    min = depth;
                if (depth > max)
                    max = depth;
            }
        }

        vec2f results;
        results(0) = min;
        results(1) = max;

        return results;
    }

    vec2f minMaxDepthParams()
    {
        float min = 0;
        float max = 0;
        int n = 0;

        for (int i = 0; i < mesh_vo::max_vertex_size; i++)
        {
            if (!m_vertices[i].used)
                continue;

            float param = getDepthParam(i);

            n++;

            if (n == 1)
            {
                min = param;
                max = param;
            }
            else
            {
                if (param < min)
                    min = param;
                if (param > max)
                    max = param;
            }
        }

        vec2f results;
        results(0) = min;
        results(1) = max;

        return results;
    }

    void scaleDepthParams(float scale)
    {
        for (int i = 0; i < mesh_vo::max_vertex_size; i++)
        {
            if (!m_vertices[i].used)
                continue;
            float param = getDepthParam(i);
            float new_param = param * scale;
            setDepthParam(new_param, i);
        }
    }

    vec2f meanStdDepth()
    {
        float old_m = 0;
        float new_m = 0;
        float old_s = 0;
        float new_s = 0;
        int n = 0;

        for (int i = 0; i < mesh_vo::max_vertex_size; i++)
        {
            if (!m_vertices[i].used)
                continue;

            float param = getVerticeDepth(i);

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
        results(1) = sqrt(new_s / (n - 1));

        return results;
    }

    void scaleVertices(float scale)
    {
        for (int i = 0; i < mesh_vo::max_vertex_size; i++)
        {
            if (!m_vertices[i].used)
                continue;
            m_vertices[i].ver *= scale;
        }
    }

    void scaleWeights(float scale)
    {
        for (int i = 0; i < mesh_vo::max_vertex_size; i++)
        {
            if (!m_vertices[i].used)
                continue;
            //weight = 1/std**2, so to scale it
            //weight = 1/((scale*std)**2)
            //weight = 1/(scane**2*std**2)
            m_vertices[i].weight *= 1.0/(scale*scale);
        }
    }

    vertex &getVertex(int id)
    {
        assert(id >= 0 && id < mesh_vo::max_vertex_size);
        assert(m_vertices[id].used);

        return m_vertices[id];
    }

    void setVertex(int id, vertex vert)
    {
        assert(id >= 0 && id < mesh_vo::max_vertex_size);
        m_vertices[id] = vert;
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
        for (int it = 0; it < mesh_vo::max_vertex_size; ++it)
        {
            if (m_vertices[it].used)
                keys.push_back((int)it);
        }
        return keys;
    }

    void transform(Sophus::SE3f pose)
    {
        for (int it = 0; it < mesh_vo::max_vertex_size; ++it)
        {
            if (!m_vertices[it].used)
                continue;
            vec3f ver = pose * m_vertices[it].ver;
            m_vertices[it].ver = ver;
            m_vertices[it].ray = ver / ver(2);
        }
    }

    void project(cameraType cam)
    {
        for (int it = 0; it < mesh_vo::max_vertex_size; ++it)
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

    void setWeightParam(float weight, unsigned int v_id)
    {
        setVerticeWeight(weight, v_id);
    }

    float getWeightParam(unsigned int v_id)
    {
        return getVerticeWeight(v_id);
    }

private:
    void setVerticeDepth(float depth, unsigned int id)
    {
        assert(id >= 0 && id < mesh_vo::max_vertex_size);
        assert(m_vertices[id].used);

        m_vertices[id].ver = m_vertices[id].ray * depth;
    }

    float getVerticeDepth(unsigned int id)
    {
        assert(id >= 0 && id < mesh_vo::max_vertex_size);
        assert(m_vertices[id].used);

        return m_vertices[id].ver(2);
    }

    void setVerticeWeight(float weight, unsigned int id)
    {
        assert(id >= 0 && id < mesh_vo::max_vertex_size);
        assert(m_vertices[id].used);

        m_vertices[id].weight = weight;
    }

    float getVerticeWeight(unsigned int id)
    {
        assert(id >= 0 && id < mesh_vo::max_vertex_size);
        assert(m_vertices[id].used);

        return m_vertices[id].weight;
    }

    vertex m_vertices[mesh_vo::max_vertex_size];
};
