#pragma once
#include <memory>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "cpu/SceneVerticesBase.h"
#include "cpu/Shapes.h"
#include "common/common.h"
#include "params.h"

class ScenePatches : public SceneVerticesBase
{
public:
    ScenePatches() : SceneVerticesBase()
    {
        patch_width = 16.0;
        patch_height = 16.0;
        setDepthJackMethod(idepthJacobian);
    };

    ScenePatches(const ScenePatches &other) : SceneVerticesBase(other)
    {
        patch_width = other.patch_width;
        patch_height = other.patch_height;
    }
    /*
    Mesh &operator=(const Mesh &other)
    {
        if (this != &other)
        {
            PointSet::operator=(other);
            triangles = other.triangles;
            last_triangle_id = other.last_triangle_id;
        }
        return *this;
    }
    */

    std::unique_ptr<SceneBase> clone() const override
    {
        return std::make_unique<ScenePatches>(*this);
    }

    int getShapesDoF() override
    {
        return 1;
    }

    std::unique_ptr<ShapeBase> getShape(unsigned int polId) override
    {
        // always return triangle in cartesian
        ShapePatch pol(getVertice(polId), patch_width, patch_height, getDepthJacMethod());
        return std::make_unique<ShapePatch>(pol);
    }

    std::vector<unsigned int> getShapesIds() const override
    {
        return getVerticesIds();
    }

    std::vector<unsigned int> getShapeParamsIds(unsigned int polId) override
    {
        std::vector<unsigned int> paramsIds;
        paramsIds.push_back(polId);
        return paramsIds;
    }

    void setParam(float param, unsigned int paramId) override
    {
        setDepthParam(param, paramId);
    }

    float getParam(unsigned int paramId) override
    {
        return getDepthParam(paramId);
    }

    Error errorRegu()
    {
        Error error;

        std::vector<unsigned int> polIds = getShapesIds();
        
        float meanParam = 0;
        for(auto polId : polIds)
        {
            meanParam += getDepthParam(polId);
        }
        meanParam /= polIds.size();

        for(auto polId : polIds)
        {
            float err = getDepthParam(polId) - meanParam;
            error.error += err*err;
        }

        error.count = polIds.size();
        return error;
    }

    HGMapped HGRegu()
    {
        HGMapped hg;
        std::vector<unsigned int> polIds = getShapesIds();

        float meanParam = 0;
        for(auto polId : polIds)
        {
            meanParam += getDepthParam(polId);
        }
        meanParam /= polIds.size();

        for(auto polId : polIds)
        {
            float error = getDepthParam(polId) - meanParam;
            hg.G[polId] += error;
            hg.H[polId][polId] += 1.0;
        }

        hg.count = polIds.size();

        return hg;
    }

    Error errorInitial(ScenePatches &initScene, MatrixMapped &initThetaVar)
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

    HGMapped HGInitial(ScenePatches &initScene, MatrixMapped &initThetaVar)
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

    int patch_width;
    int patch_height;
};
