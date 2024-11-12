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
    };

    ScenePatches(const ScenePatches &other) : SceneVerticesBase(other)
    {
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

    std::unique_ptr<ShapeBase> getShape(camera cam, int polId) override
    {
        int patch_width = cam.width / MESH_WIDTH;
        int patch_height = cam.height / MESH_HEIGHT;
        // always return triangle in cartesian
        ShapePatch pol(getRay(polId), getPix(polId), getDepth(polId), getWeight(polId), patch_width, patch_height);
        return std::make_unique<ShapePatch>(pol);
    }

    void getShape(ShapeBase *shape, camera cam, int polId) override
    {
        int patch_width = cam.width / MESH_WIDTH;
        int patch_height = cam.height / MESH_HEIGHT;
        ShapePatch *_shape = (ShapePatch *)shape;
        _shape->set(getRay(polId), getPix(polId), getDepth(polId), getWeight(polId), patch_width, patch_height);
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

    void setParamWeight(float weight, int paramId) override
    {
        setWeight(weight, paramId);
    }

    float getParam(int paramId) override
    {
        return getDepthParam(paramId);
    }

    Error errorRegu(camera cam)
    {
        Error error;

        std::vector<int> polIds = getShapesIds();

        int patch_width = cam.width / MESH_WIDTH;
        int patch_height = cam.height / MESH_HEIGHT;

        for (auto polId : polIds)
        {
            vec2<float> pix = getPix(polId);

            window win(pix(0) - patch_width * 2, pix(0) + patch_width * 2, pix(1) - patch_height * 2, pix(1) + patch_height * 2);

            float meanParam = 0;
            int count = 0;
            for (auto polId2 : polIds)
            {
                //if(polId == polId2)
                //    continue;
                vec2<float> pix2 = getPix(polId2);
                if (win.isPixInWindow(pix2))
                {
                    meanParam += getDepthParam(polId2);
                    count++;
                }
            }
            meanParam /= count;
            float param = getDepthParam(polId);
            float err = param - meanParam;
            error += err * err;
        }

        return error;
    }

    HGEigenSparse HGRegu(camera cam, int numFrames = 0)
    {
        std::vector<int> polIds = getShapesIds();

        HGEigenSparse hg(getNumParams() + numFrames * 8);

        int patch_width = cam.width / MESH_WIDTH;
        int patch_height = cam.height / MESH_HEIGHT;

        for (auto polId : polIds)
        {
            vec2<float> pix = getPix(polId);

            window win(pix(0) - patch_width * 2, pix(0) + patch_width * 2, pix(1) - patch_height * 2, pix(1) + patch_height * 2);

            float meanParam = 0;
            int count;
            std::vector<float> polIds2;
            for (auto polId2 : polIds)
            {
                //if(polId == polId2)
                //    continue;
                vec2<float> pix2 = getPix(polId2);
                if (win.isPixInWindow(pix2))
                {
                    polIds2.push_back(polId2);
                    float d = getDepthParam(polId2);
                    meanParam += d;
                    count++;
                }
            }
            float error = getDepthParam(polId) - meanParam / count;

            hg.sparseAdd(1.0, error, 1.0, polId);
            for (auto polId2 : polIds2)
            {
                hg.sparseAdd(-1.0 / count, error, 1.0, polId2);
            }
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
};
