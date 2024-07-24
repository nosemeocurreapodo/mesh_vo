#pragma once

#include <Eigen/Core>
#include <thread>

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGMapped.h"
#include "common/Error.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/SceneBase.h"
#include "cpu/frameCPU.h"
#include "params.h"

class reduceCPU
{
public:
    reduceCPU()
        : pool()
    {
    }

    Error reduceError(camera cam, dataCPU<float> &image1, dataCPU<float> &image2, int lvl)
    {
        Error err;
        reduceErrorWindow(cam, &image1, &image2, &err, lvl);
        return err;
    }

    Error reduceErrorParallel(camera cam, dataCPU<float> &image1, dataCPU<float> &image2, int lvl)
    {
        int divi = 16;

        Error partialerr[divi];

        std::array<int, 2> windowSize;
        windowSize[0] = cam.width;
        windowSize[1] = cam.height / divi;

        for (int ty = 0; ty < divi; ty++)
        {
            for (int tx = 0; tx < 1; tx++)
            {
                int min_x = tx * windowSize[0];
                int max_x = (tx + 1) * windowSize[0];
                int min_y = ty * windowSize[1];
                int max_y = (ty + 1) * windowSize[1];

                camera cam_window = cam;
                cam_window.setWindow(min_x, max_x, min_y, max_y);

                // renderJPoseWindow(&frame2Idepth, cam_window, &frame1, &frame2, &jtra_buffer, &jrot_buffer, &e_buffer, lvl);
                pool.enqueue(std::bind(&reduceCPU::reduceErrorWindow, this, cam_window, &image1, &image2, &partialerr[tx + ty * divi], lvl));
            }
        }

        pool.waitUntilDone();

        Error err;
        for (int i = 0; i < divi; i++)
        {
            err += partialerr[i];
        }

        return err;
    }

    HGPose reduceHGPose(camera cam, dataCPU<std::array<float, 6>> &jpose_buffer, dataCPU<float> &err_buffer, int lvl)
    {
        HGPose hg;
        reduceHGPoseWindow(cam, &jpose_buffer, &err_buffer, &hg, lvl);
        return hg;
    }

    HGPose reduceHGPoseParallel(camera cam, dataCPU<std::array<float, 6>> &jpose_buffer, dataCPU<float> &err_buffer, int lvl)
    {
        int divi = 16;

        HGPose partialhg[divi];

        std::array<int, 2> windowSize;
        windowSize[0] = cam.width;
        windowSize[1] = cam.height / divi;

        for (int ty = 0; ty < divi; ty++)
        {
            for (int tx = 0; tx < 1; tx++)
            {
                int min_x = tx * windowSize[0];
                int max_x = (tx + 1) * windowSize[0];
                int min_y = ty * windowSize[1];
                int max_y = (ty + 1) * windowSize[1];

                camera cam_window = cam;
                cam_window.setWindow(min_x, max_x, min_y, max_y);

                // renderJPoseWindow(&frame2Idepth, cam_window, &frame1, &frame2, &jtra_buffer, &jrot_buffer, &e_buffer, lvl);
                pool.enqueue(std::bind(&reduceCPU::reduceHGPoseWindow, this, cam_window, &jpose_buffer, &err_buffer, &partialhg[tx + ty * divi], lvl));
            }
        }

        pool.waitUntilDone();

        HGPose hg;
        for (int i = 0; i < divi; i++)
        {
            hg += partialhg[i];
        }

        return hg;
    }

    template <int DoF>
    HGMapped reduceHGMap(camera cam, dataCPU<std::array<float, DoF>> &j_buffer, dataCPU<float> &err_buffer, dataCPU<std::array<int, DoF>> &pId_buffer, int lvl)
    {
        HGMapped hg;
        reduceHGMapWindow<DoF>(cam, &j_buffer, &err_buffer, &pId_buffer, &hg, lvl);
        return hg;
    }

    template <int DoF>
    HGMapped reduceHGPoseMap(camera cam, int frameId, dataCPU<std::array<float, 6>> &jpose_buffer, dataCPU<std::array<float, DoF>> &jmap_buffer, dataCPU<float> &err_buffer, dataCPU<std::array<int, DoF>> &pId_buffer, int lvl)
    {
        HGMapped hg;
        reduceHGPoseMapWindow<DoF>(cam, frameId, &jpose_buffer, &jmap_buffer, &err_buffer, &pId_buffer, &hg, lvl);
        return hg;
    }

private:
    void reduceErrorWindow(camera cam, dataCPU<float> *frame1, dataCPU<float> *frame2, Error *err, int lvl)
    {
        for (int y = cam.window_min_y; y < cam.window_max_y; y++)
            for (int x = cam.window_min_x; x < cam.window_max_x; x++)
            {
                float p1 = frame1->get(y, x, lvl);
                float p2 = frame2->get(y, x, lvl);
                if (p1 == frame1->nodata || p2 == frame2->nodata)
                    continue;
                err->error += std::pow(p1 - p2, 2);
                err->count++;
            }
    }

    void reduceHGPoseWindow(camera cam, dataCPU<std::array<float, 6>> *jpose_buffer, dataCPU<float> *err_buffer, HGPose *hg, int lvl)
    {
        for (int y = cam.window_min_y; y < cam.window_max_y; y++)
        {
            for (int x = cam.window_min_x; x < cam.window_max_x; x++)
            {
                std::array<float, 6> J = jpose_buffer->get(y, x, lvl);
                float err = err_buffer->get(y, x, lvl);
                if (J == jpose_buffer->nodata || err == err_buffer->nodata)
                    continue;

                hg->count++;
                for (int i = 0; i < 6; i++)
                {
                    // hg->G.add(J[i] * err, i - 6);
                    hg->G(i) += J[i] * err;
                    // hg->G[i - 6] = J[i] * residual;
                    for (int j = i; j < 6; j++)
                    {
                        float jj = J[i] * J[j];
                        hg->H(i, j) += jj;
                        hg->H(j, i) += jj;
                        // hg->H.add(jj, i - 6, j - 6);
                        // hg->H.add(jj, j - 6, i - 6);
                    }
                }
            }
        }
    }

    template <int DoF>
    void reduceHGMapWindow(camera cam, dataCPU<std::array<float, DoF>> *jmap_buffer, dataCPU<float> *err_buffer, dataCPU<std::array<int, DoF>> *pId_buffer, HGMapped *hg, int lvl)
    {
        for (int y = cam.window_min_y; y < cam.window_max_y; y++)
        {
            for (int x = cam.window_min_x; x < cam.window_max_x; x++)
            {
                std::array<float, DoF> jac = jmap_buffer->get(y, x, lvl);
                float err = err_buffer->get(y, x, lvl);
                std::array<int, DoF> ids = pId_buffer->get(y, x, lvl);

                if (jac == jmap_buffer->nodata || err == err_buffer->nodata || ids == pId_buffer->nodata)
                    continue;

                hg->count += 1;
                for (int i = 0; i < DoF; i++)
                {
                    // if the jacobian is 0
                    // we really cannot say anything about the depth
                    // can make the hessian non-singular
                    if (jac[i] == 0)
                        continue;

                    hg->G.add(jac[i] * err, ids[i]);
                    //(*hg).G[v_ids[i]] += J[i] * error;

                    for (int j = i; j < DoF; j++)
                    {
                        float jj = jac[i] * jac[j];
                        hg->H.add(jj, ids[i], ids[j]);
                        hg->H.add(jj, ids[j], ids[i]);
                        //(*hg).H[v_ids[i]][v_ids[j]] += jj;
                        //(*hg).H[v_ids[j]][v_ids[i]] += jj;
                    }
                }
            }
        }
    }

    template <int DoF>
    void reduceHGPoseMapWindow(camera cam, int frameId, dataCPU<std::array<float, 6>> *jpose_buffer, dataCPU<std::array<float, DoF>> *jmap_buffer, dataCPU<float> *error_buffer, dataCPU<std::array<int, DoF>> *pId_buffer, HGMapped *hg, int lvl)
    {
        for (int y = cam.window_min_y; y < cam.window_max_y; y++)
        {
            for (int x = cam.window_min_x; x < cam.window_max_x; x++)
            {
                std::array<float, 6> J_pose = jpose_buffer->get(y, x, lvl);
                float error = error_buffer->get(y, x, lvl);

                if (J_pose == jpose_buffer->nodata || error == error_buffer->nodata)
                    continue;

                hg->count += 1;

                for (int i = 0; i < 6; i++)
                {
                    hg->G.add(J_pose[i] * error, i - (frameId + 1) * 6);

                    for (int j = i; j < 6; j++)
                    {
                        float jj = J_pose[i] * J_pose[j];
                        hg->H.add(jj, i - (frameId + 1) * 6, j - (frameId + 1) * 6);
                        hg->H.add(jj, j - (frameId + 1) * 6, i - (frameId + 1) * 6);
                    }
                }
            }
        }

        for (int y = cam.window_min_y; y < cam.window_max_y; y++)
        {
            for (int x = cam.window_min_x; x < cam.window_max_x; x++)
            {
                std::array<float, DoF> j_map = jmap_buffer->get(y, x, lvl);
                float error = error_buffer->get(y, x, lvl);
                std::array<int, DoF> ids = pId_buffer->get(y, x, lvl);

                for (int i = 0; i < DoF; i++)
                {
                    // if the jacobian is 0
                    // we really cannot say anything about the depth
                    // can make the hessian non-singular
                    if (j_map[i] == 0)
                        continue;
                    hg->G.add(j_map[i] * error, ids[i]);
                    //(*hg).G[v_ids[i]] += J[i] * error;

                    for (int j = i; j < DoF; j++)
                    {
                        float jj = j_map[i] * j_map[j];
                        hg->H.add(jj, ids[i], ids[j]);
                        hg->H.add(jj, ids[j], ids[i]);
                        //(*hg).H[v_ids[i]][v_ids[j]] += jj;
                        //(*hg).H[v_ids[j]][v_ids[i]] += jj;
                    }
                }
            }
        }
    }

    ThreadPool pool;
};