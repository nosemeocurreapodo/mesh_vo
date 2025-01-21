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

    void init(camera cam)
    {
        std::vector<vec2f> texcoords;
        std::vector<float> idepths;
        std::vector<float> weights;

        float maxIdepth = 1.0;
        float minIdepth = 0.1;

        for (float y = 0.0; y < MESH_HEIGHT; y++)
        {
            for (float x = 0.0; x < MESH_WIDTH; x++)
            {
                vec2f pix;
                pix(0) = (cam.width - 1) * x / (MESH_WIDTH - 1);
                pix(1) = (cam.height - 1) * y / (MESH_HEIGHT - 1);

                float idph = (maxIdepth - minIdepth) * float(rand() % 1000) / 1000.0 + minIdepth;
                float wght = 1.0 / (INITIAL_PARAM_STD * INITIAL_PARAM_STD);

                texcoords.push_back(pix);
                idepths.push_back(idph);
                weights.push_back(wght);

                assert(idepths.size() <= MAX_VERTEX_SIZE);
            }
        }

        init(texcoords, idepths, weights, cam);
    }

    void init(std::vector<vec3f> &vertices, std::vector<float> &weights, camera cam)
    {
        assert(vertices.size() == weights.size());
        assert(vertices.size() <= MAX_VERTEX_SIZE);

        for (size_t i = 0; i < MAX_VERTEX_SIZE; i++)
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

    void init(std::vector<vec2f> &texcoords, std::vector<float> idepths, std::vector<float> &weights, camera cam)
    {
        assert(texcoords.size() == idepths.size());
        assert(texcoords.size() == weights.size());
        assert(texcoords.size() <= MAX_VERTEX_SIZE);

        for (size_t i = 0; i < MAX_VERTEX_SIZE; i++)
        {
            m_vertices[i].used = false;
        }

        for (size_t i = 0; i < texcoords.size(); i++)
        {
            vec2f pix = texcoords[i];
            float idph = idepths[i];
            float weight = weights[i];

            assert(idph > 0.0);
            assert(!std::isnan(idph));
            assert(!std::isinf(idph));

            assert(pix(0) >= 0 && pix(0) < cam.width && pix(1) >= 0 && pix(1) < cam.height);

            vec3f ray = cam.pixToRay(pix);
            vec3f vertice = ray / idph;

            m_vertices[i] = vertex(vertice, ray, pix, weight);
        }
    }

    void init(dataCPU<float> &idepth, dataCPU<float> &weight, camera cam)
    {
        assert(idepth.width == cam.width && idepth.height == cam.height);
        assert(weight.width == cam.width && weight.height == cam.height);

        std::vector<vec2f> texcoords;
        std::vector<float> idepths;
        std::vector<float> weights;

        vec2f minMax = idepth.getMinMax();
        assert(minMax(0) != idepth.nodata && minMax(1) != idepth.nodata && minMax(0) != minMax(1));

        for (float y = 0.0; y < MESH_HEIGHT; y++)
        {
            for (float x = 0.0; x < MESH_WIDTH; x++)
            {
                vec2f pix;
                pix(0) = (cam.width - 1) * x / (MESH_WIDTH - 1);
                pix(1) = (cam.height - 1) * y / (MESH_HEIGHT - 1);

                float idph = idepth.get(pix(1), pix(0));
                float wght = weight.get(pix(1), pix(0));

                // assert(idph != idepth.nodata);
                if (idph == idepth.nodata)
                {
                    idph = (minMax[1] - minMax[0]) * float(rand() % 1000) / 1000.0 + minMax[0];
                    wght = 1.0 / (INITIAL_PARAM_STD * INITIAL_PARAM_STD);
                }
                if (wght == weight.nodata)
                {
                    wght = 1.0 / (INITIAL_PARAM_STD * INITIAL_PARAM_STD);
                }

                assert(idph > 0.0);
                assert(!std::isnan(idph));
                assert(!std::isinf(idph));

                texcoords.push_back(pix);
                idepths.push_back(idph);
                weights.push_back(wght);

                assert(idepths.size() <= MAX_VERTEX_SIZE);
            }
        }

        init(texcoords, idepths, weights, cam);
    }

    vec2f meanStdDepthParams()
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
        results(1) = sqrt(new_s / (n - 1));

        return results;
    }

    vec2f minMaxDepthParams()
    {
        float min = 0;
        float max = 0;
        int n = 0;

        for (int i = 0; i < MAX_VERTEX_SIZE; i++)
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
        for (int i = 0; i < MAX_VERTEX_SIZE; i++)
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

        for (int i = 0; i < MAX_VERTEX_SIZE; i++)
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
        for (int i = 0; i < MAX_VERTEX_SIZE; i++)
        {
            if (!m_vertices[i].used)
                continue;
            m_vertices[i].ver *= scale;
        }
    }

    void scaleWeights(float scale)
    {
        for (int i = 0; i < MAX_VERTEX_SIZE; i++)
        {
            if (!m_vertices[i].used)
                continue;
            //weight = 1/std**2, so to scale it
            //weight = 1/((scale*std)**2)
            //weight = 1/(scane**2*std**2)
            m_vertices[i].weight *= 1.0/(scale*scale);
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
            m_vertices[it].ray = ver / ver(2);
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

    void setVerticeWeight(float weight, unsigned int id)
    {
        assert(id >= 0 && id < MAX_VERTEX_SIZE);
        assert(m_vertices[id].used);

        m_vertices[id].weight = weight;
    }

    float getVerticeWeight(unsigned int id)
    {
        assert(id >= 0 && id < MAX_VERTEX_SIZE);
        assert(m_vertices[id].used);

        return m_vertices[id].weight;
    }

    vertex m_vertices[MAX_VERTEX_SIZE];
};
