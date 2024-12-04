#pragma once
#include <memory>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/common.h"
#include "common/camera.h"
#include "cpu/Shapes.h"
#include "cpu/frameCPU.h"
//#include "common/SparseLinearProblem.h"

template <typename shapeType>
class SceneBase
{
public:
    SceneBase() {};

    SceneBase(const SceneBase &other)
    {
    }

    // virtual std::unique_ptr<SceneBase> clone() const = 0;
    virtual void clear() = 0;

    virtual void transform(Sophus::SE3f relativePose) = 0;
    virtual void project(camera cam) = 0;

    virtual std::vector<int> getShapesIds() const = 0;
    virtual shapeType getShape(int polId) = 0;
    // virtual void getShape(ShapeBase* shape, int polId) = 0;
    virtual int getNumParams() = 0;

    virtual void setParam(float param, int paramId) = 0;
    virtual float getParam(int paramId) = 0;
    virtual void setParamWeight(float weight, int paramId) = 0;

    virtual Error errorRegu(camera cam) = 0;
    //virtual SparseLinearProblem HGRegu(camera cam, int numFrames) = 0;

private:
};
