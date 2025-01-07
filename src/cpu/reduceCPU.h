#pragma once

#include <Eigen/Core>
#include <thread>

#include "common/camera.h"
#include "common/DenseLinearProblem.h"
//#include "common/SparseLinearProblem.h"
//#include "common/DenseSparseLinearProblem.h"
#include "common/HGMapped.h"
#include "common/Error.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "params.h"

class reduceCPU
{
public:
    reduceCPU()
    {
    }

    Error reduceError(dataCPU<float> &residual, dataCPU<float> &weights)
    {
        Error err;
        int width = residual.width;
        int height = residual.height;
        window win = {0, width, 0, height};
        reduceErrorWindow(win, residual, weights, err);
        return err;
    }

    Error reduceErrorParallel(dataCPU<float> &residual, dataCPU<float> &weights)
    {
        const int divi_y = ThreadPool<REDUCER_NTHREADS>::getNumThreads();
        const int divi_x = 1;

        Error partialerr[divi_y * divi_x];

        int width = residual.width / divi_x;
        int height = residual.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win = {min_x, max_x, min_y, max_y};

                reduceErrorWindow(win, residual, weights, partialerr[tx + ty * divi_x]);
                //pool.enqueue(std::bind(&reduceCPU::reduceErrorWindow, this, win, &residual, &weights, &partialerr[tx + ty * divi_x], lvl));
            }
        }

        //pool.waitUntilDone();

        for (int i = 1; i < divi_y * divi_x; i++)
        {
            partialerr[0] += partialerr[i];
        }

        return partialerr[0];
    }

    DenseLinearProblem reduceHGLightAffineParallel(dataCPU<vec2<float>> &jlightaffine_buffer, dataCPU<float> &err_buffer, dataCPU<float> &weights_buffer)
    {
        const int divi_y = ThreadPool<REDUCER_NTHREADS>::getNumThreads();
        const int divi_x = 1;

        std::vector<DenseLinearProblem> partialhg;

        // calling constructor
        // for each index of array
        for (int i = 0; i < divi_x * divi_y; i++)
        {
            partialhg.push_back(DenseLinearProblem(2, 0));
        }

        int width = jlightaffine_buffer.width / divi_x;
        int height = jlightaffine_buffer.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x, min_y, max_y);

                reduceHGLightAffineWindow(win, jlightaffine_buffer, err_buffer, weights_buffer, partialhg[tx + ty * divi_x]);
                //pool.enqueue(std::bind(&reduceCPU::reduceHGLightAffineWindow, this, win, &jlightaffine_buffer, &err_buffer, &weights_buffer, &partialhg[tx + ty * divi_x]));
            }
        }

        //pool.waitUntilDone();

        DenseLinearProblem hg(2, 0);
        for (int i = 0; i < divi_y * divi_x; i++)
        {
            hg += partialhg[i];
        }

        return hg;
    }

    DenseLinearProblem reduceHGPose(dataCPU<vec8<float>> &jpose_buffer, dataCPU<float> &err_buffer, dataCPU<float> &weights_buffer)
    {
        DenseLinearProblem hg(8, 0);
        int width = jpose_buffer.width;
        int height = jpose_buffer.height;
        window win = {0, width, 0, height};
        reduceHGPoseWindow(win, jpose_buffer, err_buffer, weights_buffer, hg);
        return hg;
    }

    DenseLinearProblem reduceHGPoseParallel(dataCPU<vec8<float>> &jpose_buffer, dataCPU<float> &err_buffer, dataCPU<float> &weights_buffer)
    {
        const int divi_y = ThreadPool<REDUCER_NTHREADS>::getNumThreads();
        const int divi_x = 1;

        std::vector<DenseLinearProblem> partialhg;

        // calling constructor
        // for each index of array
        for (int i = 0; i < divi_x * divi_y; i++)
        {
            partialhg.push_back(DenseLinearProblem(8, 0));
        }

        int width = jpose_buffer.width / divi_x;
        int height = jpose_buffer.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x, min_y, max_y);

                reduceHGPoseWindow(win, jpose_buffer, err_buffer, weights_buffer, partialhg[tx + ty * divi_x]);
                //pool.enqueue(std::bind(&reduceCPU::reduceHGPoseWindow, this, win, &jpose_buffer, &err_buffer, &weights_buffer, &partialhg[tx + ty * divi_x]), lvl);
            }
        }

        //pool.waitUntilDone();

        for (int i = 1; i < divi_y * divi_x; i++)
        {
            partialhg[0] += partialhg[i];
        }

        return partialhg[0];
    }

    template <typename jmapType, typename idsType>
    HGMapped reduceHGMap(dataCPU<jmapType> &j_buffer, dataCPU<float> &err_buffer, dataCPU<idsType> &pId_buffer)
    {
        HGMapped hg;
        window win(0, j_buffer.width, 0, j_buffer.height);
        reduceHGMapWindow(win, j_buffer, err_buffer, pId_buffer, hg);
        return hg;
    }

    template <typename jmapType, typename idsType>
    DenseLinearProblem reduceHGMap2(int maxNumParams, dataCPU<jmapType> &j_buffer, dataCPU<float> &err_buffer, dataCPU<idsType> &pId_buffer)
    {
        DenseLinearProblem hg(0, maxNumParams);
        window win(0, j_buffer.width, 0, j_buffer.height);
        reduceHGMapWindow(win, j_buffer, err_buffer, pId_buffer, hg);
        return hg;
    }

    template <typename jmapType, typename idsType>
    HGMapped reduceHGMapParallel(dataCPU<jmapType> &j_buffer, dataCPU<float> &err_buffer, dataCPU<idsType> &pId_buffer)
    {
        const int divi_y = ThreadPool<REDUCER_NTHREADS>::getNumThreads();
        const int divi_x = 1;

        HGMapped partialhg[divi_x * divi_y];

        int width = j_buffer.width / divi_x;
        int height = j_buffer.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x, min_y, max_y);

                reduceHGMapWindow(win, j_buffer, err_buffer, pId_buffer, partialhg[tx + ty * divi_x]);
                //pool.enqueue(std::bind(&reduceCPU::reduceHGMapWindow<Type1, Type2>, this, win, &j_buffer, &err_buffer, &pId_buffer, &partialhg[tx + ty * divi_x], lvl));
            }
        }

        //pool.waitUntilDone();

        HGMapped hg;
        for (int i = 0; i < divi_y * divi_x; i++)
        {
            hg += partialhg[i];
        }

        return hg;
    }

    template <typename jmapType, typename idsType>
    DenseLinearProblem reduceHGMapParallel2(int maxNumParams, dataCPU<jmapType> &j_buffer, dataCPU<float> &err_buffer, dataCPU<idsType> &pId_buffer)
    {
        const int divi_y = ThreadPool<REDUCER_NTHREADS>::getNumThreads();
        const int divi_x = 1;

        std::vector<DenseLinearProblem> partialhg;

        // calling constructor
        // for each index of array
        for (int i = 0; i < divi_x * divi_y; i++)
        {
            partialhg.push_back(DenseLinearProblem(0, maxNumParams));
        }

        int width = j_buffer.width / divi_x;
        int height = j_buffer.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x, min_y, max_y);

                reduceHGMapWindow(win, j_buffer, err_buffer, partialhg[tx + ty * divi_x]);
                //pool.enqueue(std::bind(&reduceCPU::reduceHGPoseWindow, this, win, &jpose_buffer, &err_buffer, &weights_buffer, &partialhg[tx + ty * divi_x]), lvl);
            }
        }

        //pool.waitUntilDone();

        DenseLinearProblem hg(maxNumParams);
        for (int i = 0; i < divi_y * divi_x; i++)
        {
            hg += partialhg[i];
        }

        return hg;
    }

    template <typename jmapType, typename idsType>
    HGMapped reduceHGPoseMap(int frameId, dataCPU<vec8<float>> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &err_buffer, dataCPU<idsType> &pId_buffer, int lvl)
    {
        HGMapped hg;
        window win(0, jpose_buffer.width, 0, jpose_buffer.height);
        reduceHGPoseMapWindow(win, frameId, jpose_buffer, jmap_buffer, err_buffer, pId_buffer, hg);
        return hg;
    }

    template <typename jmapType, typename idsType>
    DenseLinearProblem reduceHGPoseMap2(int frameId, int numFrames, int numMapParams, dataCPU<vec8<float>> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &err_buffer, dataCPU<idsType> &pId_buffer)
    {
        DenseLinearProblem hg(numMapParams + numFrames * 6);
        window win(0, jpose_buffer.width, 0, jpose_buffer.height);
        reduceHGPoseMapWindow(win, frameId, numMapParams, jpose_buffer, jmap_buffer, err_buffer, pId_buffer, hg);
        return hg;
    }

    template <typename jmapType, typename idsType>
    HGMapped reduceHGPoseMapParallel(int frameId, int numFrames, int numMapParams, dataCPU<vec8<float>> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &err_buffer, dataCPU<idsType> &pId_buffer)
    {
        const int divi_y = ThreadPool<REDUCER_NTHREADS>::getNumThreads();
        const int divi_x = 1;

        HGMapped partialhg[divi_x * divi_y];

        int width = jpose_buffer.width / divi_x;
        int height = jpose_buffer.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x, min_y, max_y);

                reduceHGPoseMapWindow(win, frameId, numMapParams, jpose_buffer, jmap_buffer, err_buffer, pId_buffer, partialhg[tx + ty * divi_x]);
                //pool.enqueue(std::bind(&reduceCPU::reduceHGPoseMapWindow<Type1, Type2>, this, win, frameId, numMapParams, &jpose_buffer, &jmap_buffer, &err_buffer, &pId_buffer, &partialhg[tx + ty * divi_x], lvl));
            }
        }

        //pool.waitUntilDone();

        HGMapped hg;
        for (int i = 0; i < divi_y * divi_x; i++)
        {
            hg += partialhg[i];
        }

        return hg;
    }

    template <typename jmapType, typename idsType>
    DenseLinearProblem reduceHGPoseMapParallel2(int frameId, int numFrames, int numMapParams, dataCPU<vec8<float>> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &err_buffer, dataCPU<idsType> &pId_buffer)
    {
        const int divi_y = ThreadPool<REDUCER_NTHREADS>::getNumThreads();
        const int divi_x = 1;

        // HGEigenSparse partialhg[divi_x * divi_y];

        std::vector<DenseLinearProblem> partialhg;

        // calling constructor
        // for each index of array
        for (int i = 0; i < divi_x * divi_y; i++)
        {
            partialhg.push_back(DenseLinearProblem(numFrames*8, numMapParams));
        }

        int width = jpose_buffer.width / divi_x;
        int height = jpose_buffer.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x, min_y, max_y);

                reduceHGPoseMapWindow(win, frameId, jpose_buffer, jmap_buffer, err_buffer, pId_buffer, partialhg[tx + ty * divi_x]);
                //pool.enqueue(std::bind(&reduceCPU::reduceHGPoseMapWindow<Type1, Type2>, this, win, frameId, numMapParams, &jpose_buffer, &jmap_buffer, &err_buffer, &pId_buffer, &partialhg[tx + ty * divi_x], lvl));
            }
        }

        //pool.waitUntilDone();

        for (int i = 1; i < divi_y * divi_x; i++)
        {
            partialhg[0] += partialhg[i];
        }

        return partialhg[0];
    }

private:

    void reduceErrorWindow(window win, dataCPU<float> &residual, dataCPU<float> &weights, Error &err)
    {
        for (int y = win.min_y; y < win.max_y; y++)
            for (int x = win.min_x; x < win.max_x; x++)
            {
                float res = residual.get(y, x);
                if (res == residual.nodata)
                    continue;
                float w = weights.get(y, x);
                if (w == weights.nodata)
                    w = 1.0;
                float absresidual = std::fabs(res);
                float hw = 1.0;
                if (absresidual > HUBER_THRESH_PIX)
                    hw = HUBER_THRESH_PIX / absresidual;
                err += w * hw * res * res;
            }
    }

    void reduceHGLightAffineWindow(window win, dataCPU<vec2<float>> &jlightaffine_buffer, dataCPU<float> &res_buffer, dataCPU<float> &weights_buffer, DenseLinearProblem &hg)
    {
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                vec2<float> J = jlightaffine_buffer.get(y, x);
                float res = res_buffer.get(y, x);
                if (J == jlightaffine_buffer.nodata || res == res_buffer.nodata || J == vec2<float>(0.0, 0.0))
                    continue;
                float w = weights_buffer.get(y, x);
                if (w == weights_buffer.nodata)
                    w = 1.0;
                float absres = std::fabs(res);
                float hw = 1.0;
                if (absres > HUBER_THRESH_PIX)
                    hw = HUBER_THRESH_PIX / absres;

                hg.add(J, res, w * hw);
            }
        }
    }

    void reduceHGPoseWindow(window win, dataCPU<vec8<float>> &jpose_buffer, dataCPU<float> &res_buffer, dataCPU<float> &weights_buffer, DenseLinearProblem &hg)
    {
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                vec8<float> J = jpose_buffer.get(y, x);
                float res = res_buffer.get(y, x);
                if (J == jpose_buffer.nodata || res == res_buffer.nodata || J == vec8<float>(0.0))
                    continue;
                float w = weights_buffer.get(y, x);
                if (w == weights_buffer.nodata)
                    w = 1.0;
                float absres = std::fabs(res);
                float hw = 1.0;
                if (absres > HUBER_THRESH_PIX)
                    hw = HUBER_THRESH_PIX / absres;

                hg.add(J, res, w * hw);
            }
        }
    }

    template <typename jmapType, typename idsType>
    void reduceHGMapWindow(window win, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &res_buffer, dataCPU<idsType> &pId_buffer, HGMapped &hg)
    {
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                jmapType jac = jmap_buffer.get(y, x);
                float res = res_buffer.get(y, x);
                idsType ids = pId_buffer.get(y, x);

                if (res == res_buffer.nodata)// jac == jmap_buffer->nodata || ids == pId_buffer->nodata || jac == Type1::zero())
                    continue;

                float absres = std::fabs(res);
                float hw = 1.0;
                if (absres > HUBER_THRESH_PIX)
                    hw = HUBER_THRESH_PIX / absres;

                hg.add(jac, res, hw, ids);
            }
        }
    }

    template <typename jmapType, typename idsType>
    void reduceHGMapWindow(window win, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &res_buffer, dataCPU<idsType> &pId_buffer, DenseLinearProblem &hg)
    {
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                jmapType jac = jmap_buffer.get(y, x);
                float res = res_buffer.get(y, x);
                idsType ids = pId_buffer.get(y, x);

                if (res == res_buffer.nodata)// jac == jmap_buffer->nodata || || ids == pId_buffer->nodata || jac == Type1::zero())
                    continue;

                float absres = std::fabs(res);
                float hw = 1.0;
                if (absres > HUBER_THRESH_PIX)
                    hw = HUBER_THRESH_PIX / absres;

                hg.add(jac, res, hw, ids);
            }
        }
    }

    template <typename jmapType, typename idsType>
    void reduceHGPoseMapWindow(window win, int frameId, int numMapParams, dataCPU<vec8<float>> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &res_buffer, dataCPU<idsType> &pId_buffer, HGMapped &hg)
    {
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                vec8<float> J_pose = jpose_buffer.get(y, x);
                jmapType J_map = jmap_buffer.get(y, x);
                float res = res_buffer.get(y, x);
                idsType map_ids = pId_buffer.get(y, x);

                if (J_pose == jpose_buffer.nodata || res == res_buffer.nodata)// || J_map == jmap_buffer->nodata || J_pose == vec8<float>::zero() || J_map == Type1::zero())
                    continue;

                vec8<int> pose_ids;
                for (int i = 0; i < 8; i++)
                    pose_ids(i) = numMapParams + frameId * 8 + i;

                vecx<8 + jmapType::size(), float> J(0);
                vecx<8 + jmapType::size(), int> ids(0);

                for(int i = 0; i < 8; i++)
                {
                    J(i) = J_pose(i);
                    ids(i) = pose_ids(i);
                }
                for(int i = 0; i < jmapType::size(); i++)
                {
                    J(i + 8) = J_map(i);
                    ids(i + 8) = map_ids(i);
                }

                float absres = std::fabs(res);
                float hw = 1.0;
                if (absres > HUBER_THRESH_PIX)
                    hw = HUBER_THRESH_PIX / absres;

                hg.add(J, res, hw, ids);
            }
        }
    }

    template <typename jmapType, typename idsType>
    void reduceHGPoseMapWindow(window win, int frameId, dataCPU<vec8<float>> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &res_buffer, dataCPU<idsType> &pId_buffer, DenseLinearProblem &hg)
    {
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                vec8<float> J_pose = jpose_buffer.get(y, x);
                jmapType J_map = jmap_buffer.get(y, x);
                float res = res_buffer.get(y, x);
                idsType map_ids = pId_buffer.get(y, x);

                if (res == res_buffer.nodata || J_pose == vec8<float>(0.0))// J_pose == jpose_buffer->nodata || J_map == vecx<float>::zero())
                    continue;

                float absres = std::fabs(res);
                float hw = 1.0;
                if (absres > HUBER_THRESH_PIX)
                    hw = HUBER_THRESH_PIX / absres;

                hg.add(J_pose, J_map, res, hw, frameId, map_ids);

                /*
                vec8<int> pose_ids;
                for (int i = 0; i < 8; i++)
                    pose_ids(i) = numMapParams + frameId * 8 + i;

                vecx<8 + jmapType::size(), float> J(0);
                vecx<8 + jmapType::size(), int> ids(0);

                for(int i = 0; i < 8; i++)
                {
                    J(i) = J_pose(i);
                    ids(i) = pose_ids(i);
                }
                for(int i = 0; i < jmapType::size(); i++)
                {
                    J(i + 8) = J_map(i);
                    ids(i + 8) = map_ids(i);
                }

                hg->add(J, res, hw, ids);
                */
            }
        }
    }

    ThreadPool<REDUCER_NTHREADS> pool;
};