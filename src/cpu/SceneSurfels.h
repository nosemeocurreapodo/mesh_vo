#pragma once
#include <memory>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "cpu/SceneVerticesNormalsBase.h"
#include "cpu/Shapes.h"
#include "common/common.h"
#include "params.h"

class SceneSurfels : public SceneVerticesNormalsBase
{
public:
    SceneSurfels() : SceneVerticesNormalsBase()
    {
        radius = 8;
        setDepthJackMethod(idepthJacobian);
    };

    SceneSurfels(const SceneSurfels &other) : SceneVerticesNormalsBase(other)
    {
        radius = other.radius;
    }
    /*
    SurfelSet &operator=(const SurfelSet &other)
    {
        if (this != &other)
        {
            PointSetNormals::operator=(other);
        }
        return *this;
    }
    */
    std::unique_ptr<SceneBase> clone() const override
    {
        return std::make_unique<SceneSurfels>(*this);
    }

    int getShapesDoF() override
    {
        return 3;
    }

    std::unique_ptr<ShapeBase> getShape(unsigned int id) override
    {
        vec3<float> center = getVertice(id);
        vec3<float> normal = getNormal(id);
        ShapeSurfel pol(center, normal, radius, getDepthJacMethod());
        return std::make_unique<ShapeSurfel>(pol);
    }

    std::vector<unsigned int> getShapesIds() const override
    {
        return getVerticesIds();
    }

    std::vector<unsigned int> getShapeParamsIds(unsigned int polId) override
    {
        // the id of the param (depth, nx, ny) is the id of the vertice * 3 + x
        std::vector<unsigned int> ids;
        ids.push_back(polId * 3);     // for the depth
        ids.push_back(polId * 3 + 1); // for nx
        ids.push_back(polId * 3 + 2); // for ny
        return ids;
    }

    void setParam(float param, unsigned int paramId) override
    {
        unsigned int verticeId = (unsigned int)paramId / 3;
        // if it is multiple of 3, its the depth
        if (paramId % 3 == 0)
        {
            setDepthParam(param, verticeId);
        }
        // otherwise its the normal
        else
        {
            Eigen::Vector3f normal = getNormal(verticeId);
            if (paramId % 3 == 1)
                normal(0) = param;
            else
                normal(1) = param;
            Eigen::Vector3f vertice = getVertice(verticeId);
            //Eigen::Vector3f ray = vertice / vertice(2);
            setNormal(normal, verticeId);
        }
    }

    float getParam(unsigned int paramId) override
    {
        unsigned int verticeId = (unsigned int)paramId / 3;
        // if it is multiple of 3, its the depth
        if (paramId % 3 == 0)
        {
            return getDepthParam(verticeId);
        }
        // otherwise its the normal
        else
        {
            Eigen::Vector3f normal = getNormal(verticeId);
            if (paramId % 3 == 1)
                return normal(0);
            else
                return normal(1);
        }
    }

    Error errorRegu()
    {
        Error error;

        std::vector<unsigned int> polIds = getVerticesIds();

        for (size_t index = 0; index < polIds.size(); index++)
        {
            unsigned int id = polIds[index];

            Eigen::Vector3f vertice = getVertice(id);
            Eigen::Vector3f normal = getNormal(id);
            Eigen::Vector3f diff = normal - vertice / vertice(2);

            error.error += diff(0) * diff(0) + diff(1) * diff(1) + diff(2) * diff(2);
        }
        // divided by the number of triangles
        // we don't want to have less error if we have less triangles
        error.count = polIds.size();
        return error;
    }

    HGMapped HGRegu()
    {
        HGMapped hg;

        std::vector<unsigned int> polIds = getVerticesIds();

        for (size_t i = 0; i < polIds.size(); i++)
        {
            unsigned int p_id = polIds[i];

            Eigen::Vector3f vertice = getVertice(p_id);
            Eigen::Vector3f normal = getNormal(p_id);
            Eigen::Vector3f diff = normal - vertice / vertice(2);

            std::vector<unsigned int> pa_id = getShapeParamsIds(p_id);

            Eigen::Vector3f J[3];
            J[0] = {0.0, 0.0, 0.0};
            J[1] = {1.0, 0.0, 0.0};
            J[2] = {0.0, 1.0, 0.0};

            for (int j = 0; j < 3; j++)
            {
                // if (hg.G(NUM_FRAMES*6 + vertexIndex[j]) == 0)
                //     continue;
                hg.G[pa_id[j]] += diff.dot(J[j]);
                for (int k = 0; k < 3; k++)
                {
                    hg.H[pa_id[j]][pa_id[k]] += J[j].dot(J[k]);
                }
            }
        }

        hg.count = polIds.size();

        return hg;
    }

    Error errorInitial(SceneSurfels &initScene, MatrixMapped &initThetaVar)
    {
        Error error;

        std::vector<unsigned int> vertsIds = getVerticesIds();

        for (size_t index = 0; index < vertsIds.size(); index++)
        {
            unsigned int id = vertsIds[index];

            float initVar = initThetaVar[id][id];

            float theta = getDepthParam(id);
            float initTheta = initScene.getDepthParam(id);

            float diff = theta - initTheta;

            error.error += initVar * diff * diff;
        }
        // divided by the number of triangles
        // we don't want to have less error if we have less triangles
        error.count = vertsIds.size();
        return error;
    }

    HGMapped HGInitial(SceneSurfels &initScene, MatrixMapped &initThetaVar)
    {
        HGMapped hg;

        std::vector<unsigned int> vertsIds = getVerticesIds();

        for (size_t i = 0; i < vertsIds.size(); i++)
        {
            unsigned int v_id = vertsIds[i];

            float initVar = initThetaVar[v_id][v_id];

            float theta = getDepthParam(v_id);
            float initTheta = initScene.getDepthParam(v_id);

            hg.G[v_id] += initVar * (theta - initTheta);
            hg.H[v_id][v_id] += initVar;
        }

        hg.count = vertsIds.size();

        return hg;
    }

private:
    float radius;
};
