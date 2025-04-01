#pragma once

#include <pangolin/pangolin.h>
#include "params.h"
#include "common/camera.h"
#include "common/depthParam.h"
#include "cpu/frameCPU.h"

class GeometryVertices // : public SceneBase
{
public:
    GeometryVertices()
    {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
    }

    void init(std::vector<vec3f> &vertices, std::vector<float> &weights, cameraType cam)
    {
        assert(vertices.size() == weights.size());

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size()*3*sizeof(float), scene_vertices.data(), GL_STREAM_DRAW);
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
            // weight = 1/std**2, so to scale it
            // weight = 1/((scale*std)**2)
            // weight = 1/(scane**2*std**2)
            m_vertices[i].weight *= 1.0 / (scale * scale);
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

    unsigned int VAO;
    unsigned int VBO;
};
