#pragma once
#include <memory>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "cpu/SceneVertices.h"
#include "cpu/Shapes.h"
#include "common/common.h"
#include "params.h"

class ScenePatches : public SceneVertices
{
public:
    ScenePatches() : SceneVertices()
    {
        patch_width = 10;
        patch_height = 10;
    };

    ScenePatches(const ScenePatches &other) : SceneVertices(other)
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

    ScenePatches clone() const
    {
        return ScenePatches(*this);
    }

    void transform(camera cam, Sophus::SE3f pose)
    {
        SceneVertices::transform(cam, pose);
        patch_width = int(cam.fx*0.1);
        patch_height = int(cam.fy*0.1);
    }

    int getNumParams()
    {
        // one depth for each vertice
        return getVerticesIds().size();
    }

    /*
    int getShapesDoF() override
    {
        return 1;
    }

    bool isShapeInWindow(window &win, int polId) override
    {
        if (win.isPixInWindow(getPix(polId)))
            return true;
        return false;
    }
    */

    ShapePatch getShape(int polId)
    {
        // always return triangle in cartesian
        return ShapePatch(getVertex(polId), polId, patch_width, patch_height);
    }

    /*
    void getShape(ShapeBase *shape, int polId) override
    {
        ShapePatch *_shape = (ShapePatch *)shape;
        _shape->set(getRay(polId), getPix(polId), getDepth(polId), getWeight(polId), polId, patch_width, patch_height);
    }
    */
   
    std::vector<int> getShapesIds() const
    {
        return getVerticesIds();
    }

    /*
    std::vector<int> getShapeParamsIds(int polId) override
    {
        std::vector<int> paramsIds;
        paramsIds.push_back(polId);
        return paramsIds;
    }
    */

    void setParam(float param, int paramId)
    {
        setDepthParam(param, paramId);
    }

    void setWeight(float weight, int paramId)
    {
        setParamWeight(weight, paramId);
    }

    float getParam(int paramId)
    {
        return getDepthParam(paramId);
    }

    float getWeight(int paramId)
    {
        return getParamWeight(paramId);
    }

    Error errorRegu(camera cam)
    {
        Error error;

        std::vector<int> polIds = getShapesIds();

        int patch_width = cam.width / MESH_WIDTH;
        int patch_height = cam.height / MESH_HEIGHT;

        for (auto polId : polIds)
        {
            vec2<float> pix = getVertex(polId).pix;

            window win(pix(0) - patch_width * 2, pix(0) + patch_width * 2, pix(1) - patch_height * 2, pix(1) + patch_height * 2);

            float meanParam = 0;
            int count = 0;
            for (auto polId2 : polIds)
            {
                //if(polId == polId2)
                //    continue;
                vec2<float> pix2 = getVertex(polId2).pix;
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
            vec2<float> pix = getVertex(polId).pix;

            window win(pix(0) - patch_width * 2, pix(0) + patch_width * 2, pix(1) - patch_height * 2, pix(1) + patch_height * 2);

            float meanParam = 0;
            int count;
            std::vector<float> polIds2;
            for (auto polId2 : polIds)
            {
                //if(polId == polId2)
                //    continue;
                vec2<float> pix2 = getVertex(polId2).pix;
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

    int patch_width;
    int patch_height;
};
