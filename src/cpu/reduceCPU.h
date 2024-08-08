#pragma once

#include <Eigen/Core>
#include <thread>

#include "common/camera.h"
#include "common/HGEigenDense.h"
#include "common/HGEigenSparse.h"
#include "common/HGMapped.h"
#include "common/Error.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/SceneBase.h"
#include "params.h"

class reduceCPU
{
public:
    reduceCPU()
        : pool(4)
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
        int divi_y = pool.getNumThreads();
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
    HGEigenDense reduceHGPose(camera cam, dataCPU<Type> &jpose_buffer, dataCPU<float> &err_buffer, int lvl)
    {
        HGEigenDense hg;
        window win = {0, cam.width, 0, cam.height};
        reduceHGPoseWindow(win, &jpose_buffer, &err_buffer, &hg, lvl);
        return hg;
    }

    template <typename Type>
    HGEigenDense reduceHGPoseParallel(camera cam, dataCPU<Type> &jpose_buffer, dataCPU<float> &err_buffer, int lvl)
    {
        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        HGEigenDense partialhg[divi_x * divi_y];

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

        HGEigenDense hg;
        for (int i = 0; i < divi_y * divi_x; i++)
        {
            hg += partialhg[i];
        }

        return hg;
    }

    template <typename Type1, typename Type2>
    HGMapped reduceHGMap(camera cam, dataCPU<Type1> &j_buffer, dataCPU<float> &err_buffer, dataCPU<Type2> &pId_buffer, int lvl)
    {
        HGMapped hg;
        window win(0, cam.width, 0, cam.height);
        reduceHGMapWindow(win, &j_buffer, &err_buffer, &pId_buffer, &hg, lvl);
        return hg;
    }

    template <typename Type1, typename Type2>
    HGEigenSparse reduceHGMap2(camera cam, int maxNumParams, dataCPU<Type1> &j_buffer, dataCPU<float> &err_buffer, dataCPU<Type2> &pId_buffer, int lvl)
    {
        HGEigenSparse hg(maxNumParams);
        window win(0, cam.width, 0, cam.height);
        reduceHGMapWindow(win, &j_buffer, &err_buffer, &pId_buffer, &hg, lvl);
        return hg;
    }

    template <typename Type1, typename Type2>
    HGMapped reduceHGMapParallel(camera cam, dataCPU<Type1> &j_buffer, dataCPU<float> &err_buffer, dataCPU<Type2> &pId_buffer, int lvl)
    {
        int divi_y = pool.getNumThreads();
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

                reduceHGMapWindow(win, &j_buffer, &err_buffer, &pId_buffer, &partialhg[tx + ty * divi_x], lvl);
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

    template <typename Type1, typename Type2>
    HGMapped reduceHGPoseMap(camera cam, int frameId, dataCPU<vec6<float>> &jpose_buffer, dataCPU<Type1> &jmap_buffer, dataCPU<float> &err_buffer, dataCPU<Type2> &pId_buffer, int lvl)
    {
        HGMapped hg;
        window win(0, cam.width, 0, cam.height);
        reduceHGPoseMapWindow(win, frameId, &jpose_buffer, &jmap_buffer, &err_buffer, &pId_buffer, &hg, lvl);
        return hg;
    }

    template <typename Type1, typename Type2>
    HGEigenSparse reduceHGPoseMap2(camera cam, int frameId, int numFrames, int numMapParams, dataCPU<vec6<float>> &jpose_buffer, dataCPU<Type1> &jmap_buffer, dataCPU<float> &err_buffer, dataCPU<Type2> &pId_buffer, int lvl)
    {
        HGEigenSparse hg(numMapParams + numFrames * 6);
        window win(0, cam.width, 0, cam.height);
        reduceHGPoseMapWindow2(win, frameId, numMapParams, &jpose_buffer, &jmap_buffer, &err_buffer, &pId_buffer, &hg, lvl);
        return hg;
    }

    template <typename Type1, typename Type2>
    HGMapped reduceHGPoseMapParallel(camera cam, int frameId, int numFrames, int numMapParams, dataCPU<vec6<float>> &jpose_buffer, dataCPU<Type1> &jmap_buffer, dataCPU<float> &err_buffer, dataCPU<Type2> &pId_buffer, int lvl)
    {
        int divi_y = pool.getNumThreads();
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

                reduceHGPoseMapWindow(win, frameId, numMapParams, &jpose_buffer, &jmap_buffer, &err_buffer, &pId_buffer, &partialhg[tx + ty * divi_x], lvl);
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

    template <typename Type1, typename Type2>
    HGEigenSparse reduceHGPoseMapParallel2(camera cam, int frameId, int numFrames, int numMapParams, dataCPU<vec6<float>> &jpose_buffer, dataCPU<Type1> &jmap_buffer, dataCPU<float> &err_buffer, dataCPU<Type2> &pId_buffer, int lvl)
    {
        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        // HGEigenSparse partialhg[divi_x * divi_y];

        std::vector<HGEigenSparse> partialhg;

        // calling constructor
        // for each index of array
        for (int i = 0; i < divi_x * divi_y; i++)
        {
            partialhg.push_back(HGEigenSparse(numMapParams + numFrames * 6));
        }

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

                // reduceHGPoseMapWindow(win, frameId, numMapParams, &jpose_buffer, &jmap_buffer, &err_buffer, &pId_buffer, &partialhg[tx + ty * divi_x], lvl);
                pool.enqueue(std::bind(&reduceCPU::reduceHGPoseMapWindow2<Type1, Type2>, this, win, frameId, numMapParams, &jpose_buffer, &jmap_buffer, &err_buffer, &pId_buffer, &partialhg[tx + ty * divi_x], lvl));
            }
        }

        pool.waitUntilDone();

        HGEigenSparse hg(numMapParams + numFrames * 6);
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
                *err += std::pow(p1 - p2, 2);
            }
    }

    template <typename Type>
    void reduceHGPoseWindow(window win, dataCPU<Type> *jpose_buffer, dataCPU<float> *err_buffer, HGEigenDense *hg, int lvl)
    {
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                Type J = jpose_buffer->get(y, x, lvl);
                float err = err_buffer->get(y, x, lvl);
                if (J == jpose_buffer->nodata || err == err_buffer->nodata)
                    continue;

                hg->add(J, err);
            }
        }
    }

    template <typename Type1, typename Type2>
    void reduceHGMapWindow(window win, dataCPU<Type1> *jmap_buffer, dataCPU<float> *err_buffer, dataCPU<Type2> *pId_buffer, HGMapped *hg, int lvl)
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

                hg->add(jac, err, ids);
            }
        }
    }

    template <typename Type1, typename Type2>
    void reduceHGMapWindow(window win, dataCPU<Type1> *jmap_buffer, dataCPU<float> *err_buffer, dataCPU<Type2> *pId_buffer, HGEigenSparse *hg, int lvl)
    {
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        // tripletList.reserve(paramIds.size() * 3);

        // Eigen::SparseMatrix<float> eigenMatrix(paramIds.size(), paramIds.size());
        // eigenMatrix.setFromTriplets(tripletList.begin(), tripletList.end());

        // return eigenMatrix;

        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                Type1 jac = jmap_buffer->get(y, x, lvl);
                float err = err_buffer->get(y, x, lvl);
                Type2 ids = pId_buffer->get(y, x, lvl);

                if (jac == jmap_buffer->nodata || err == err_buffer->nodata || ids == pId_buffer->nodata)
                    continue;

                hg->sparseAdd(jac, err, ids);
            }
        }

        hg->endSparseAdd();
    }

    template <typename Type1, typename Type2>
    void reduceHGPoseMapWindow(window win, int frameId, int numMapParams, dataCPU<vec6<float>> *jpose_buffer, dataCPU<Type1> *jmap_buffer, dataCPU<float> *error_buffer, dataCPU<Type2> *pId_buffer, HGMapped *hg, int lvl)
    {
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                vec6<float> J_pose = jpose_buffer->get(y, x, lvl);
                Type1 J_map = jmap_buffer->get(y, x, lvl);
                float error = error_buffer->get(y, x, lvl);
                Type2 map_ids = pId_buffer->get(y, x, lvl);

                if (J_pose == jpose_buffer->nodata || J_map == jmap_buffer->nodata || error == error_buffer->nodata)
                    continue;

                vec6<int> pose_ids;
                for (int i = 0; i < 6; i++)
                    pose_ids(i) = numMapParams + frameId * 6 + i;

                vecx<float> J(J_pose, J_map);
                vecx<int> idss(pose_ids, map_ids);

                hg->add(J, error, idss);
            }
        }
        /*
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                vec6<float> J_pose = jpose_buffer->get(y, x, lvl);
                // Type2 J_map = jmap_buffer->get(y, x, lvl);
                float error = error_buffer->get(y, x, lvl);
                // Type3 ids = pId_buffer->get(y, x, lvl);

                if (J_pose == jpose_buffer->nodata || error == error_buffer->nodata)
                    continue;

                vec6<unsigned int> ids;
                for (int i = 0; i < 6; i++)
                    ids(i) = numMapParams + frameId * 6 + i;

                hg->add(J_pose, error, ids);
            }
        }

        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                Type1 J_map = jmap_buffer->get(y, x, lvl);
                float error = error_buffer->get(y, x, lvl);
                Type2 ids = pId_buffer->get(y, x, lvl);

                hg->add(J_map, error, ids);
            }
        }
        */
    }

    template <typename Type1, typename Type2>
    void reduceHGPoseMapWindow2(window win, int frameId, int numMapParams, dataCPU<vec6<float>> *jpose_buffer, dataCPU<Type1> *jmap_buffer, dataCPU<float> *error_buffer, dataCPU<Type2> *pId_buffer, HGEigenSparse *hg, int lvl)
    {
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;

        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                vec6<float> J_pose = jpose_buffer->get(y, x, lvl);
                Type1 J_map = jmap_buffer->get(y, x, lvl);
                float error = error_buffer->get(y, x, lvl);
                Type2 map_ids = pId_buffer->get(y, x, lvl);

                if (J_pose == jpose_buffer->nodata || J_map == jmap_buffer->nodata || error == error_buffer->nodata)
                    continue;

                vec6<int> pose_ids;
                for (int i = 0; i < 6; i++)
                    pose_ids(i) = numMapParams + frameId * 6 + i;

                vecx<float> J(J_pose, J_map);
                vecx<int> idss(pose_ids, map_ids);

                hg->sparseAdd(J, error, idss);
            }
        }

        /*
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                Type2 J_map = jmap_buffer->get(y, x, lvl);
                float error = error_buffer->get(y, x, lvl);
                Type3 ids = pId_buffer->get(y, x, lvl);

                hg->sparseAdd(J_map, error, ids);
            }
        }
        */

        hg->endSparseAdd();
    }

    ThreadPool pool;
};