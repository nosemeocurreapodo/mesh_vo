#pragma once

#include <Eigen/Core>

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGMapped.h"
#include "common/Error.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/PointSet.h"
#include "cpu/frameCPU.h"
#include "cpu/IndexThreadReduce.h"
#include "params.h"

class renderCPU
{
public:
    renderCPU(unsigned int width, unsigned int height)
        : z_buffer(width, height, -1)
    {
    }

    void renderIdepth(PointSet &rendObj, camera &cam, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderImage(PointSet &rendObj, camera &cam, dataCPU<float> &image, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderDebug(PointSet &rendObj, camera &cam, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);

    void renderJPose(PointSet &rendObj, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<Eigen::Vector3f> &jtra_buffer, dataCPU<Eigen::Vector3f> &jrot_buffer, dataCPU<float> &e_buffer, int lvl);
    void renderJPose(dataCPU<float> &frame1Idepth, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<Eigen::Vector3f> &jtra_buffer, dataCPU<Eigen::Vector3f> &jrot_buffer, dataCPU<float> &e_buffer, int lvl);
    void renderJMap(PointSet &rendObj, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<Eigen::Vector3f> &j_buffer, dataCPU<float> &e_buffer, dataCPU<Eigen::Vector3i> &id_buffer, MapJacobianMethod mapJacMethod, int lvl);

    void renderJPoseMap(PointSet &rendObj, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<Eigen::Vector3f> &j1_buffer, dataCPU<Eigen::Vector3f> &j2_buffer, dataCPU<Eigen::Vector3f> &j3_buffer, dataCPU<float> &e_buffer, dataCPU<Eigen::Vector3i> &id_buffer, MapJacobianMethod mapJacMethod, int lvl);

private:
    dataCPU<float> z_buffer;

    //IndexThreadReduce<Error> errorTreadReduce;
    //IndexThreadReduce<HGMapped> hgMappedTreadReduce;
};
