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
        patch_width = 8.0;
        patch_height = 8.0;
        setDepthJackMethod(logIdepthJacobian);
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

    int getNumParams() override
    {
        // one depth for each vertice
        return getVerticesIds().size();
    }

    bool isShapeInWindow(window &win, int polId) override
    {
        if (win.isPixInWindow(getPix(polId)))
            return true;
        return false;
    }

    std::unique_ptr<ShapeBase> getShape(int polId) override
    {
        // always return triangle in cartesian
        ShapePatch pol(getRay(polId), getPix(polId), getDepth(polId), patch_width, patch_height, getDepthJacMethod());
        return std::make_unique<ShapePatch>(pol);
    }

    void getShape(ShapeBase *shape, int polId) override
    {
        ShapePatch *_shape = (ShapePatch *)shape;
        _shape->set(getRay(polId), getPix(polId), getDepth(polId), patch_width, patch_height, getDepthJacMethod());
    }

    std::vector<int> getShapesIds() const override
    {
        return getVerticesIds();
    }

    std::vector<int> getShapeParamsIds(int polId) override
    {
        std::vector<int> paramsIds;
        paramsIds.push_back(polId);
        return paramsIds;
    }

    void setParam(float param, int paramId) override
    {
        setDepthParam(param, paramId);
    }

    float getParam(int paramId) override
    {
        return getDepthParam(paramId);
    }

    Error errorRegu()
    {
        Error error;

        std::vector<int> polIds = getShapesIds();

        float meanParam = 0;
        for (auto polId : polIds)
        {
            meanParam += getDepthParam(polId);
        }
        meanParam /= polIds.size();

        for (auto polId : polIds)
        {
            float err = getDepthParam(polId) - meanParam;
            error += err * err;
        }

        return error;
    }

    HGEigenSparse HGRegu(int numFrames = 0)
    {
        std::vector<int> polIds = getShapesIds();

        HGEigenSparse hg(getNumParams() + numFrames * 6);

        float meanParam = 0;
        for (auto polId : polIds)
        {
            meanParam += getDepthParam(polId);
        }
        meanParam /= polIds.size();

        for (auto polId : polIds)
        {
            float error = getDepthParam(polId) - meanParam;

            hg.sparseAdd(1.0, error, polId);
        }

        hg.endSparseAdd();

        return hg;
    }

    Error errorInitial(ScenePatches &initScene, MatrixMapped &initThetaVar)
    {
        Error error;

        /*
        std::vector<int> vertsIds = getShapesIds();

        for (size_t index = 0; index < vertsIds.size(); index++)
        {
            unsigned int id = vertsIds[index];

            float initVar = initThetaVar[id][id];

            float theta = getDepthParam(id);
            float initTheta = initScene.getDepthParam(id);

            float diff = theta - initTheta;

            error += initVar * diff * diff;
        }
        */

        return error;
    }

    HGMapped HGInitial(ScenePatches &initScene, MatrixMapped &initThetaVar)
    {
        HGMapped hg;

        /*
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
        */
        return hg;
    }

private:
    int patch_width;
    int patch_height;
};
