#pragma once

#include <Eigen/Core>
#include <thread>

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGMapped.h"
#include "common/HGEigen.h"
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
    //    : pool(2)
    {
    }

    Error reduceError(camera cam, dataCPU<float> &image1, dataCPU<float> &image2, int lvl)
    {
        Error err;
        window win = {0, cam.width, 0, cam.height};
        reduceErrorWindow(win, &image1, &image2, &err, lvl);
        return err;
    }

    Error reduceErrorParallel(camera cam, dataCPU<float> &image1, dataCPU<float> &image2, int lvl)
    {
        int divi_y = 16;
        int divi_x = 1;

        Error partialerr[divi_y * divi_x];

        std::array<int, 2> windowSize;
        windowSize[0] = cam.width / divi_x;
        windowSize[1] = cam.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * windowSize[0];
                int max_x = (tx + 1) * windowSize[0];
                int min_y = ty * windowSize[1];
                int max_y = (ty + 1) * windowSize[1];

                window win = {min_x, max_x, min_y, max_y};

                reduceErrorWindow(win, &image1, &image2, &partialerr[tx + ty * divi_x], lvl);
                // pool.enqueue(std::bind(&reduceCPU::reduceErrorWindow, this, window, &image1, &image2, &partialerr[tx + ty * divi_x], lvl));
            }
        }

        // pool.waitUntilDone();

        Error err;
        for (int i = 0; i < divi_y * divi_x; i++)
        {
            err += partialerr[i];
        }

        return err;
    }

    template <typename Type>
    HGPose reduceHGPose(camera cam, dataCPU<Type> &jpose_buffer, dataCPU<float> &err_buffer, int lvl)
    {
        HGPose hg;
        window win = {0, cam.width, 0, cam.height};
        reduceHGPoseWindow(win, &jpose_buffer, &err_buffer, &hg, lvl);
        return hg;
    }

    template <typename Type>
    HGPose reduceHGPoseParallel(camera cam, dataCPU<Type> &jpose_buffer, dataCPU<float> &err_buffer, int lvl)
    {
        int divi_y = 16;
        int divi_x = 1;

        HGPose partialhg[divi_x * divi_y];

        std::array<int, 2> windowSize;
        windowSize[0] = cam.width / divi_x;
        windowSize[1] = cam.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * windowSize[0];
                int max_x = (tx + 1) * windowSize[0];
                int min_y = ty * windowSize[1];
                int max_y = (ty + 1) * windowSize[1];

                window win(min_x, max_x, min_y, max_y);

                reduceHGPoseWindow(win, &jpose_buffer, &err_buffer, &partialhg[tx + ty * divi_x], lvl);
                // pool.enqueue(std::bind(&reduceCPU::reduceHGPoseWindow, this, window, &jpose_buffer, &err_buffer, &partialhg[tx + ty * divi_x], lvl));
            }
        }

        // pool.waitUntilDone();

        HGPose hg;
        for (int i = 0; i < divi_y * divi_x; i++)
        {
            hg += partialhg[i];
        }

        return hg;
    }

    template <typename Type1, typename Type2>
    HGMapped reduceHGMap(camera cam, int shapeDoF, dataCPU<Type1> &j_buffer, dataCPU<float> &err_buffer, dataCPU<Type2> &pId_buffer, int lvl)
    {
        HGMapped hg;
        window win(0, cam.width, 0, cam.height);
        reduceHGMapWindow(win, shapeDoF, &j_buffer, &err_buffer, &pId_buffer, &hg, lvl);
        return hg;
    }

    template <typename Type1, typename Type2>
    HGEigen reduceHGMap2(camera cam, int shapeDoF, dataCPU<Type1> &j_buffer, dataCPU<float> &err_buffer, dataCPU<Type2> &pId_buffer, int lvl)
    {
        HGEigen hg(MESH_WIDTH*MESH_HEIGHT);
        window win(0, cam.width, 0, cam.height);
        reduceHGMapWindow(win, shapeDoF, &j_buffer, &err_buffer, &pId_buffer, &hg, lvl);
        return hg;
    }

    template <typename Type1, typename Type2>
    HGMapped reduceHGMapParallel(camera cam, int shapeDoF, dataCPU<Type1> &j_buffer, dataCPU<float> &err_buffer, dataCPU<Type2> &pId_buffer, int lvl)
    {
        int divi_y = 1;
        int divi_x = 1;

        HGMapped partialhg[divi_x * divi_y];

        std::array<int, 2> windowSize;
        windowSize[0] = cam.width / divi_x;
        windowSize[1] = cam.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * windowSize[0];
                int max_x = (tx + 1) * windowSize[0];
                int min_y = ty * windowSize[1];
                int max_y = (ty + 1) * windowSize[1];

                window win(min_x, max_x, min_y, max_y);

                reduceHGMapWindow(win, shapeDoF, &j_buffer, &err_buffer, &pId_buffer, &partialhg[tx + ty * divi_x], lvl);
                // pool.enqueue(std::bind(&reduceCPU::reduceHGMapWindow<DoF>, this, window, &j_buffer, &err_buffer, &pId_buffer, &partialhg[tx + ty * divi_x], lvl));
            }
        }

        // pool.waitUntilDone();

        HGMapped hg;
        for (int i = 0; i < divi_y * divi_x; i++)
        {
            hg += partialhg[i];
        }

        return hg;
    }

    template <typename Type1, typename Type2, typename Type3>
    HGMapped reduceHGPoseMap(camera cam, int frameId, int shapeDoF, dataCPU<Type1> &jpose_buffer, dataCPU<Type2> &jmap_buffer, dataCPU<float> &err_buffer, dataCPU<Type3> &pId_buffer, int lvl)
    {
        HGMapped hg;
        window win(0, cam.width, 0, cam.height);
        reduceHGPoseMapWindow(win, frameId, &jpose_buffer, &jmap_buffer, &err_buffer, &pId_buffer, &hg, lvl);
        return hg;
    }

    template <typename Type1, typename Type2, typename Type3>
    HGMapped reduceHGPoseMapParallel(camera cam, int frameId, int shapeDoF, dataCPU<Type1> &jpose_buffer, dataCPU<Type2> &jmap_buffer, dataCPU<float> &err_buffer, dataCPU<Type3> &pId_buffer, int lvl)
    {
        int divi_y = 16;
        int divi_x = 1;

        HGMapped partialhg[divi_x * divi_y];

        std::array<int, 2> windowSize;
        windowSize[0] = cam.width / divi_x;
        windowSize[1] = cam.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * windowSize[0];
                int max_x = (tx + 1) * windowSize[0];
                int min_y = ty * windowSize[1];
                int max_y = (ty + 1) * windowSize[1];

                window win(min_x, max_x, min_y, max_y);

                reduceHGPoseMapWindow(win, frameId, shapeDoF, &jpose_buffer, &jmap_buffer, &err_buffer, &pId_buffer, &partialhg[tx + ty * divi_x], lvl);
                // pool.enqueue(std::bind(&reduceCPU::reduceHGPoseMapWindow<DoF>, this, window, frameId, &jpose_buffer, &jmap_buffer, &err_buffer, &pId_buffer, &partialhg[tx + ty * divi_x], lvl));
            }
        }

        // pool.waitUntilDone();

        HGMapped hg;
        for (int i = 0; i < divi_y * divi_x; i++)
        {
            hg += partialhg[i];
        }

        return hg;
    }

private:
    void reduceErrorWindow(window win, dataCPU<float> *frame1, dataCPU<float> *frame2, Error *err, int lvl)
    {
        for (int y = win.min_y; y < win.max_y; y++)
            for (int x = win.min_x; x < win.max_x; x++)
            {
                float p1 = frame1->get(y, x, lvl);
                float p2 = frame2->get(y, x, lvl);
                if (p1 == frame1->nodata || p2 == frame2->nodata)
                    continue;
                err->error += std::pow(p1 - p2, 2);
                err->count++;
            }
    }

    template <typename Type>
    void reduceHGPoseWindow(window win, dataCPU<Type> *jpose_buffer, dataCPU<float> *err_buffer, HGPose *hg, int lvl)
    {
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                Type J = jpose_buffer->get(y, x, lvl);
                float err = err_buffer->get(y, x, lvl);
                if (J == jpose_buffer->nodata || err == err_buffer->nodata)
                    continue;

                hg->count++;
                for (int i = 0; i < 6; i++)
                {
                    // hg->G.add(J[i] * err, i - 6);
                    hg->G(i) += J(i) * err;
                    // hg->G[i - 6] = J[i] * residual;
                    for (int j = i; j < 6; j++)
                    {
                        float jj = J(i) * J(j);
                        hg->H(i, j) += jj;
                        hg->H(j, i) += jj;
                        // hg->H.add(jj, i - 6, j - 6);
                        // hg->H.add(jj, j - 6, i - 6);
                    }
                }
            }
        }
    }

    template <typename Type1, typename Type2>
    void reduceHGMapWindow(window win, int shapeDoF, dataCPU<Type1> *jmap_buffer, dataCPU<float> *err_buffer, dataCPU<Type2> *pId_buffer, HGMapped *hg, int lvl)
    {
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                Type1 jac = jmap_buffer->get(y, x, lvl);
                float err = err_buffer->get(y, x, lvl);
                Type2 ids = pId_buffer->get(y, x, lvl);

                if (jac == jmap_buffer->nodata || err == err_buffer->nodata || ids == pId_buffer->nodata)
                    continue;

                hg->count += 1;
                for (int i = 0; i < shapeDoF; i++)
                {
                    // if the jacobian is 0
                    // we really cannot say anything about the depth
                    // can make the hessian non-singular
                    if (jac(i) == 0)
                        continue;

                    hg->G.add(jac(i) * err, ids(i));
                    //(*hg).G[v_ids[i]] += J[i] * error;

                    for (int j = i; j < shapeDoF; j++)
                    {
                        float jj = jac(i) * jac(j);
                        hg->H.add(jj, ids(i), ids(j));
                        hg->H.add(jj, ids(j), ids(i));
                        //(*hg).H[v_ids[i]][v_ids[j]] += jj;
                        //(*hg).H[v_ids[j]][v_ids[i]] += jj;
                    }
                }
            }
        }
    }

    template <typename Type1, typename Type2>
    void reduceHGMapWindow(window win, int shapeDoF, dataCPU<Type1> *jmap_buffer, dataCPU<float> *err_buffer, dataCPU<Type2> *pId_buffer, HGEigen *hg, int lvl)
    {
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        //tripletList.reserve(paramIds.size() * 3);

        //Eigen::SparseMatrix<float> eigenMatrix(paramIds.size(), paramIds.size());
        //eigenMatrix.setFromTriplets(tripletList.begin(), tripletList.end());

        //return eigenMatrix;

        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                Type1 jac = jmap_buffer->get(y, x, lvl);
                float err = err_buffer->get(y, x, lvl);
                Type2 ids = pId_buffer->get(y, x, lvl);

                if (jac == jmap_buffer->nodata || err == err_buffer->nodata || ids == pId_buffer->nodata)
                    continue;

                hg->count += 1;
                for (int i = 0; i < shapeDoF; i++)
                {
                    // if the jacobian is 0
                    // we really cannot say anything about the depth
                    // can make the hessian non-singular
                    if (jac(i) == 0)
                        continue;

                    hg->G[ids(i)] += jac(i) * err;
                    //(*hg).G[v_ids[i]] += J[i] * error;

                    for (int j = i; j < shapeDoF; j++)
                    {
                        float jj = jac(i) * jac(j);

                        tripletList.push_back(T(ids(i), ids(j), jj));
                        tripletList.push_back(T(ids(j), ids(i), jj));

                        //hg->H.coeffRef(ids[i], ids[j]) += jj;
                        //hg->H.coeffRef(ids[j], ids[i]) += jj;
                    }
                }
            }
        }

        hg->H.setFromTriplets(tripletList.begin(), tripletList.end());
    }

    template <typename Type1, typename Type2, typename Type3>
    void reduceHGPoseMapWindow(window win, int frameId, int shapeDoF, dataCPU<Type1> *jpose_buffer, dataCPU<Type2> *jmap_buffer, dataCPU<float> *error_buffer, dataCPU<Type3> *pId_buffer, HGMapped *hg, int lvl)
    {
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                Type1 J_pose = jpose_buffer->get(y, x, lvl);
                float error = error_buffer->get(y, x, lvl);

                if (J_pose == jpose_buffer->nodata || error == error_buffer->nodata)
                    continue;

                // hg->count += 1;
                for (int i = 0; i < 6; i++)
                {
                    hg->G.add(J_pose(i) * error, i - (frameId + 1) * 6);

                    for (int j = i; j < 6; j++)
                    {
                        float jj = J_pose(i) * J_pose(j);
                        hg->H.add(jj, i - (frameId + 1) * 6, j - (frameId + 1) * 6);
                        hg->H.add(jj, j - (frameId + 1) * 6, i - (frameId + 1) * 6);
                    }
                }
            }
        }

        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                Type2 j_map = jmap_buffer->get(y, x, lvl);
                float error = error_buffer->get(y, x, lvl);
                Type3 ids = pId_buffer->get(y, x, lvl);

                hg->count += 1;
                for (int i = 0; i < shapeDoF; i++)
                {
                    // if the jacobian is 0
                    // we really cannot say anything about the depth
                    // can make the hessian non-singular
                    if (j_map(i) == 0)
                        continue;
                    hg->G.add(j_map(i) * error, ids(i));
                    //(*hg).G[v_ids[i]] += J[i] * error;

                    for (int j = i; j < shapeDoF; j++)
                    {
                        float jj = j_map(i) * j_map(j);
                        hg->H.add(jj, ids(i), ids(j));
                        hg->H.add(jj, ids(j), ids(i));
                        //(*hg).H[v_ids[i]][v_ids[j]] += jj;
                        //(*hg).H[v_ids[j]][v_ids[i]] += jj;
                    }
                }
            }
        }
    }

    // ThreadPool pool;
};