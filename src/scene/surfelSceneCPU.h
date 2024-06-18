#pragma once

#include <Eigen/Core>

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/Error.h"
#include "cpu/frameCPU.h"
#include "cpu/IndexThreadReduce.h"
#include "params.h"

class surfelSceneCPU
{
public:
    surfelSceneCPU(float fx, float fy, float cx, float cy, int width, int height);

    void init(frameCPU &frame, dataCPU<float> &idepth);

    dataCPU<float> computeFrameIdepth(frameCPU &frame, int lvl);
    dataCPU<float> computeErrorImage(frameCPU &frame, int lvl);

    void optPose(frameCPU &frame);
    void optMap(std::vector<frameCPU> &frame);
    void optPoseMap(std::vector<frameCPU> &frame);

    camera getCam()
    {
        return cam;
    }

private:
    // scene
    std::vector<float> scene_vertices;

    frameCPU keyframe;
    camera cam;

    dataCPU<float> z_buffer;

    void setFromIdepth(dataCPU<float> id);

    float computeError(frameCPU &frame, int lvl);
    HGPose computeHGPose(frameCPU &frame, int lvl);
    void computeHGMap(frameCPU &frame, HGPoseMap &hg, int lvl);
    void computeHGPoseMap(frameCPU &frame, HGPoseMap &hg, int frame_index, int lvl);

    void errorPerIndex(frameCPU &frame, int lvl, int tmin, int tmax, Error *e, int tid);
    void HGPosePerIndex(frameCPU &frame, int lvl, int tmin, int tmax, HGPose *hg, int tid);

    float errorRegu();
    void HGRegu(HGPoseMap &hgmap);

    IndexThreadReduce<Error> errorTreadReduce;
    IndexThreadReduce<HGPose> hgPoseTreadReduce;

    bool multiThreading;

    class HGPoseMap
    {
    public:
        HGPoseMap(int frames, int n_surfels)
        {
            num_frames = frames;
            num_surfels = n_surfels;

            H = Eigen::SparseMatrix<float>(num_frames * 6 + num_surfels * 6, num_frames * 6 + num_surfels * 6);
            G = Eigen::VectorXf::Zero(num_frames * 6 + num_surfels * 6);
            count = Eigen::VectorXf::Zero(num_frames * 6 + num_surfels * 6);
        }

        Eigen::SparseMatrix<float> H;
        Eigen::VectorXf G;
        Eigen::VectorXf count;

        int num_frames;
        int num_surfels;

    private:
    };
};
