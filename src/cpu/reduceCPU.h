#pragma once

#include <thread>

#include "params.h"
#include "common/window.h"
#include "common/Error.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "common/DenseLinearProblem.h"

class reduceCPU
{
public:
    reduceCPU()
    {
    }

    Error reduceErrorParallel(dataCPU<float> &residual)
    {
        const int divi_y = ThreadPool<mesh_vo::reducer_nthreads>::getNumThreads();
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

                window<int> win(min_x, max_x, min_y, max_y);

                reduceErrorWindow(win, residual, partialerr[tx + ty * divi_x]);
                // pool.enqueue(std::bind(&reduceCPU::reduceErrorWindow, this, win, &residual, &weights, &partialerr[tx + ty * divi_x], lvl));
            }
        }

        // pool.waitUntilDone();

        for (int i = 1; i < divi_y * divi_x; i++)
        {
            partialerr[0] += partialerr[i];
        }

        return partialerr[0];
    }

    DenseLinearProblem reduceHGExpParallel(dataCPU<vec2f> &jexp_buffer, dataCPU<float> &err_buffer)
    {
        const int divi_y = ThreadPool<mesh_vo::reducer_nthreads>::getNumThreads();
        const int divi_x = 1;

        std::vector<DenseLinearProblem> partialhg;

        // calling constructor
        // for each index of array
        for (int i = 0; i < divi_x * divi_y; i++)
        {
            partialhg.push_back(DenseLinearProblem(2));
        }

        int width = jexp_buffer.width / divi_x;
        int height = jexp_buffer.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window<int> win(min_x, max_x, min_y, max_y);

                reduceHGExpWindow(win, jexp_buffer, err_buffer, partialhg[tx + ty * divi_x]);
                // pool.enqueue(std::bind(&reduceCPU::reduceHGLightAffineWindow, this, win, &jlightaffine_buffer, &err_buffer, &weights_buffer, &partialhg[tx + ty * divi_x]));
            }
        }

        // pool.waitUntilDone();

        for (int i = 1; i < divi_y * divi_x; i++)
        {
            partialhg[0] += partialhg[i];
        }

        return partialhg[0];
    }

    DenseLinearProblem reduceHGPoseParallel(dataCPU<vec6f> &jpose_buffer, dataCPU<float> &err_buffer)
    {
        const int divi_y = ThreadPool<mesh_vo::reducer_nthreads>::getNumThreads();
        const int divi_x = 1;

        std::vector<DenseLinearProblem> partialhg;

        // calling constructor
        // for each index of array
        for (int i = 0; i < divi_x * divi_y; i++)
        {
            partialhg.push_back(DenseLinearProblem(6));
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

                window<int> win(min_x, max_x, min_y, max_y);

                reduceHGPoseWindow(win, jpose_buffer, err_buffer, partialhg[tx + ty * divi_x]);
                // pool.enqueue(std::bind(&reduceCPU::reduceHGPoseWindow, this, win, &jpose_buffer, &err_buffer, &weights_buffer, &partialhg[tx + ty * divi_x]), lvl);
            }
        }

        // pool.waitUntilDone();

        for (int i = 1; i < divi_y * divi_x; i++)
        {
            partialhg[0] += partialhg[i];
        }

        return partialhg[0];
    }

    DenseLinearProblem reduceHGMapParallel(int maxNumParams, dataCPU<jmapType> &j_buffer, dataCPU<float> &err_buffer, dataCPU<idsType> &pId_buffer)
    {
        const int divi_y = ThreadPool<mesh_vo::reducer_nthreads>::getNumThreads();
        const int divi_x = 1;

        std::vector<DenseLinearProblem> partialhg;

        // calling constructor
        // for each index of array
        for (int i = 0; i < divi_x * divi_y; i++)
        {
            partialhg.push_back(DenseLinearProblem(maxNumParams));
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

                window<int> win(min_x, max_x, min_y, max_y);

                reduceHGMapWindow(win, j_buffer, err_buffer, pId_buffer, partialhg[tx + ty * divi_x]);
                // pool.enqueue(std::bind(&reduceCPU::reduceHGPoseWindow, this, win, &jpose_buffer, &err_buffer, &weights_buffer, &partialhg[tx + ty * divi_x]), lvl);
            }
        }

        // pool.waitUntilDone();

        DenseLinearProblem hg(maxNumParams);
        for (int i = 0; i < divi_y * divi_x; i++)
        {
            hg += partialhg[i];
        }

        return hg;
    }

    DenseLinearProblem reduceHGPoseMapParallel(int frameId, int numFrames, int numMapParams, dataCPU<vec6f> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &err_buffer, dataCPU<idsType> &pId_buffer)
    {
        const int divi_y = ThreadPool<mesh_vo::reducer_nthreads>::getNumThreads();
        const int divi_x = 1;

        // HGEigenSparse partialhg[divi_x * divi_y];

        std::vector<DenseLinearProblem> partialhg;

        // calling constructor
        // for each index of array
        for (int i = 0; i < divi_x * divi_y; i++)
        {
            partialhg.push_back(DenseLinearProblem(numFrames * 6 + numMapParams));
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

                window<int> win(min_x, max_x, min_y, max_y);

                reduceHGPoseMapWindow(win, frameId, numFrames, jpose_buffer, jmap_buffer, err_buffer, pId_buffer, partialhg[tx + ty * divi_x]);
                // pool.enqueue(std::bind(&reduceCPU::reduceHGPoseMapWindow<Type1, Type2>, this, win, frameId, numMapParams, &jpose_buffer, &jmap_buffer, &err_buffer, &pId_buffer, &partialhg[tx + ty * divi_x], lvl));
            }
        }

        // pool.waitUntilDone();

        for (int i = 1; i < divi_y * divi_x; i++)
        {
            partialhg[0] += partialhg[i];
        }

        return partialhg[0];
    }

    DenseLinearProblem reduceHGIntrinsicPoseMapParallel(int frameId, int numFrames, int numMapParams, dataCPU<vec4f> &jintrinsic_buffer, dataCPU<vec6f> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &err_buffer, dataCPU<idsType> &pId_buffer)
    {
        const int divi_y = ThreadPool<mesh_vo::reducer_nthreads>::getNumThreads();
        const int divi_x = 1;

        // HGEigenSparse partialhg[divi_x * divi_y];

        std::vector<DenseLinearProblem> partialhg;

        // calling constructor
        // for each index of array
        for (int i = 0; i < divi_x * divi_y; i++)
        {
            partialhg.push_back(DenseLinearProblem(4 + numFrames * 6 + numMapParams));
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

                window<int> win(min_x, max_x, min_y, max_y);

                reduceHGIntrinsicPoseMapWindow(win, frameId, numFrames, jintrinsic_buffer, jpose_buffer, jmap_buffer, err_buffer, pId_buffer, partialhg[tx + ty * divi_x]);
                // pool.enqueue(std::bind(&reduceCPU::reduceHGPoseMapWindow<Type1, Type2>, this, win, frameId, numMapParams, &jpose_buffer, &jmap_buffer, &err_buffer, &pId_buffer, &partialhg[tx + ty * divi_x], lvl));
            }
        }

        // pool.waitUntilDone();

        for (int i = 1; i < divi_y * divi_x; i++)
        {
            partialhg[0] += partialhg[i];
        }

        return partialhg[0];
    }

private:
    void reduceErrorWindow(window<int> win, dataCPU<float> &residual, Error &err)
    {
        for (int y = win.min_y; y < win.max_y; y+=1)
            for (int x = win.min_x; x < win.max_x; x+=1)
            {
                float res = residual.getTexel(y, x);
                if (res == residual.nodata)
                    continue;
                float absresidual = std::fabs(res);
                float hw = 1.0;
                if (absresidual > mesh_vo::huber_thresh_pix)
                    hw = mesh_vo::huber_thresh_pix / absresidual;
                err += hw * res * res;
            }
    }

    void reduceHGExpWindow(window<int> win, dataCPU<vec2f> &jexp_buffer, dataCPU<float> &res_buffer, DenseLinearProblem &hg)
    {
        for (int y = win.min_y; y < win.max_y; y+=1)
        {
            for (int x = win.min_x; x < win.max_x; x+=1)
            {
                vec2f J = jexp_buffer.getTexel(y, x);
                float res = res_buffer.getTexel(y, x);
                if (J == jexp_buffer.nodata || res == res_buffer.nodata)
                    continue;
                float absres = std::fabs(res);
                float hw = 1.0;
                if (absres > mesh_vo::huber_thresh_pix)
                    hw = mesh_vo::huber_thresh_pix / absres;

                hg.add(J, res, hw);
            }
        }
    }

    void reduceHGPoseWindow(window<int> win, dataCPU<vec6f> &jpose_buffer, dataCPU<float> &res_buffer, DenseLinearProblem &hg)
    {
        for (int y = win.min_y; y < win.max_y; y+=1)
        {
            for (int x = win.min_x; x < win.max_x; x+=1)
            {
                vec6f J = jpose_buffer.getTexel(y, x);
                float res = res_buffer.getTexel(y, x);
                if (J == jpose_buffer.nodata || res == res_buffer.nodata)
                    continue;
                float absres = std::fabs(res);
                float hw = 1.0;
                if (absres > mesh_vo::huber_thresh_pix)
                    hw = mesh_vo::huber_thresh_pix / absres;

                hg.add(J, res, hw);
            }
        }
    }

    void reduceHGMapWindow(window<int> win, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &res_buffer, dataCPU<idsType> &pId_buffer, DenseLinearProblem &hg)
    {
        for (int y = win.min_y; y < win.max_y; y+=1)
        {
            for (int x = win.min_x; x < win.max_x; x+=1)
            {
                jmapType jac = jmap_buffer.getTexel(y, x);
                float res = res_buffer.getTexel(y, x);
                idsType ids = pId_buffer.getTexel(y, x);

                if (res == res_buffer.nodata || jac == jmap_buffer.nodata || ids == pId_buffer.nodata)
                    continue;

                float absres = std::fabs(res);
                float hw = 1.0;
                if (absres > mesh_vo::huber_thresh_pix)
                    hw = mesh_vo::huber_thresh_pix / absres;

                hg.add(jac, res, hw, ids);
            }
        }
    }

    void reduceHGPoseMapWindow(window<int> win, int frameId, int numFrames, dataCPU<vec6f> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &res_buffer, dataCPU<idsType> &pId_buffer, DenseLinearProblem &hg)
    {
        for (int y = win.min_y; y < win.max_y; y+=1)
        {
            for (int x = win.min_x; x < win.max_x; x+=1)
            {
                vec6f J_pose = jpose_buffer.getTexel(y, x);
                jmapType J_map = jmap_buffer.getTexel(y, x);
                float res = res_buffer.getTexel(y, x);
                idsType map_ids = pId_buffer.getTexel(y, x);

                if (res == res_buffer.nodata || J_pose == jpose_buffer.nodata || J_map == jmap_buffer.nodata || map_ids == pId_buffer.nodata)
                    continue;

                float absres = std::fabs(res);
                float hw = 1.0;
                if (absres > mesh_vo::huber_thresh_pix)
                    hw = mesh_vo::huber_thresh_pix / absres;

                //hg.add(J_pose, J_map, res, hw, frameId, map_ids);

                vecxf J(6 + J_map.rows());
                vecxi ids(6 + J_map.rows());

                for(int i = 0; i < 6; i++)
                {
                    J(i) = J_pose(i);
                    ids(i) = frameId * 6 + i;;
                }
                for(int i = 0; i < J_map.rows(); i++)
                {
                    J(i + 6) = J_map(i);
                    ids(i + 6) = map_ids(i) + numFrames*6;
                }

                hg.add(J, res, hw, ids);
            }
        }
    }

    void reduceHGIntrinsicPoseMapWindow(window<int> win, int frameId, int numFrames, dataCPU<vec4f> &jintrinsic_buffer, dataCPU<vec6f> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &res_buffer, dataCPU<idsType> &pId_buffer, DenseLinearProblem &hg)
    {
        for (int y = win.min_y; y < win.max_y; y+=1)
        {
            for (int x = win.min_x; x < win.max_x; x+=1)
            {
                vecxf J_int = jintrinsic_buffer.getTexel(y, x);
                vec6f J_pose = jpose_buffer.getTexel(y, x);
                jmapType J_map = jmap_buffer.getTexel(y, x);
                float res = res_buffer.getTexel(y, x);
                idsType map_ids = pId_buffer.getTexel(y, x);

                if (res == res_buffer.nodata || J_int == jintrinsic_buffer.nodata || J_pose == jpose_buffer.nodata || J_map == jmap_buffer.nodata || map_ids == pId_buffer.nodata)
                    continue;

                float absres = std::fabs(res);
                float hw = 1.0;
                if (absres > mesh_vo::huber_thresh_pix)
                    hw = mesh_vo::huber_thresh_pix / absres;

                //hg.add(J_pose, J_map, res, hw, frameId, map_ids);

                vecxf J(J_int.rows() + 6 + J_map.rows());
                vecxi ids(J_int.rows() + 6 + J_map.rows());

                for(int i = 0; i < J_int.rows(); i++)
                {
                    J(i) = J_int(i);
                    ids(i) = i;
                }
                for(int i = 0; i < 6; i++)
                {
                    J(i + J_int.rows()) = J_pose(i);
                    ids(i + J_int.rows()) = frameId * 6 + i + J_int.rows();
                }
                for(int i = 0; i < J_map.rows(); i++)
                {
                    J(i + 6 + J_int.rows()) = J_map(i);
                    ids(i + 6 + J_int.rows()) = map_ids(i) + numFrames*6 + J_int.rows();
                }

                hg.add(J, res, hw, ids);
            }
        }
    }

    ThreadPool<mesh_vo::reducer_nthreads> pool;
};