#pragma once
#include <memory>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/common.h"
#include "common/camera.h"
#include "cpu/Shapes.h"
#include "cpu/frameCPU.h"
#include "common/HGEigenSparse.h"

class SceneBase
{
public:
    SceneBase(){};

    SceneBase(const SceneBase &other)
    {
        globalPose = other.globalPose;
    }

    void setPose(Sophus::SE3f newGlobalPose)
    {
        globalPose = newGlobalPose;
    }

    Sophus::SE3f getPose()
    {
        return globalPose;
    }

    virtual std::unique_ptr<SceneBase> clone() const = 0;
    virtual void clear() = 0;

    virtual void init(frameCPU &frame, camera &cam, dataCPU<float> &idepth, dataCPU<float> &ivar, int lvl) = 0;
    virtual void init(frameCPU &frame, camera &cam, std::vector<vec2<float>> &pixels, std::vector<float> &depths, int lvl) = 0;

    virtual void complete(frameCPU &frame, camera &cam, dataCPU<float> &idepth, int lvl) = 0;
    virtual void transform(Sophus::SE3f newGlobalPose) = 0;
    virtual void project(camera cam) = 0;

    virtual bool isShapeInWindow(window &win, int polId) = 0;

    virtual std::vector<int> getShapesIds() const = 0;
    virtual std::unique_ptr<ShapeBase> getShape(camera cam, int polId) = 0;
    virtual void getShape(ShapeBase* shape, camera cam, int polId) = 0;
    virtual int getNumParams() = 0;
    virtual int getShapesDoF() = 0;

    virtual std::vector<int> getShapeParamsIds(int polId) = 0;
    virtual void setParam(float param, int paramId) = 0;
    virtual float getParam(int paramId) = 0;
    virtual void setParamWeight(float weight, int paramId) = 0;

    virtual Error errorRegu(camera cam) = 0;
    virtual HGEigenSparse HGRegu(camera cam, int numFrames) = 0;
    // virtual Error errorInitial(SceneBase &initScene, MatrixMapped &initThetaVar) = 0;
    // virtual HGMapped HGInitial(SceneBase &initMesh, MatrixMapped &initThetaVar) = 0;

private:
    Sophus::SE3f globalPose;
};
