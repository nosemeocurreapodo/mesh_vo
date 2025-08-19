#pragma once

#include <thread>

#include "params.h"

#include "common/DenseLinearProblem.h"

template <typename In1Type, typename In2Type, typename In3Type, typename OutType>
class BaseReducerCPU
{
public:
    BaseReducerCPU() = default;
    virtual ~BaseReducerCPU() = default;

    virtual void reducepartial(int min, int max,
                               BufferCPU<In1Type> &reduce1,
                               BufferCPU<In2Type> &reduce2,
                               BufferCPU<In3Type> &reduce3,
                               OutType &out) = 0;

    OutType reduce(BufferCPU<In1Type> &reduce1, BufferCPU<In2Type> &reduce2, BufferCPU<In3Type> &reduce3)
    {
        static_assert(reduce1.size() == reduce2.size())

            const int divi = 1; // ThreadPool<mesh_vo::reducer_nthreads>::getNumThreads();

        OutType partial[divi];

        int size = reduce1.size();
        int step = reduce1.size() / divi;

        for (int t = 0; t < size; t += step)
        {
            int min = t;
            int max = t + step - 1;

            reducepartial(min, max, reduce1, reduce2, reduce3, partial[t]);
            // pool.enqueue(std::bind(&reduceCPU::reduceErrorWindow, this, win, &residual, &partialerr[tx + ty * divi_x]));
        }
    }

    // pool.waitUntilDone();

    for (int i = 1; i < divi_y * divi_x; i++)
    {
        partial[0] += partial[i];
    }

    return partial[0];
}
}

class ErrorReducerCPU : public BaseReducerCPU<ImageType, Error>
{

public:
    void reducepartial(int min, int max, BufferCPU<ImageType> &image, BufferCPU<ImageType> kimage_projected, BufferCPU<ImageType> &notused, Error &err)
    {
        for (int i = min; i < max; i++)
        {
            ImageType val = image[i];
            ImageType kval = kimage_projected[i];
            // if (val == image.nodata || kval == kimage_projected.nodata)
            //     continue;
            float res = val - kval;
            float absresidual = std::fabs(res);
            float hw = 1.0;
            if (absresidual > mesh_vo::huber_thresh_pix)
                hw = mesh_vo::huber_thresh_pix / absresidual;
            err += hw * res * res;
        }
    }
}

class HGPoseReducerCPU : public BaseReducerCPU<JPoseType, ImageType, ImageType, DenseLinearProblem>
{
public:
    void reducepartial(int min, int max,
                       BufferCPU<jposeType> &jpose_buffer,
                       BufferCPU<imageType> &image,
                       BufferCPU<imageType> &kimage_projected,
                       DenseLinearProblem &hg)
    {
        for (int i = min; i < max; i++)
        {
            JPoseType J = jpose_buffer[i];
            ImageType val = image[i];
            ImageType kval = kimage_projected[i];
            // if (J == jpose_buffer.nodata || val == image.nodata || kval == kimage_projected.nodata)
            //     continue;
            float res = val - kval;
            float absres = std::fabs(res);
            float hw = 1.0;
            if (absres > mesh_vo::huber_thresh_pix)
                hw = mesh_vo::huber_thresh_pix / absres;

            hg.add(J, res, hw);
        }
    }
}

/*
class HGPoseVelReducerCPU : public BaseReducerCPU<JPoseType, ImageType, ImageType, DenseLinearProblem>
{
public:
    void
    reducepartial(int min, int max,
                  BufferCPU<jposeType> &jpose_buffer,
                  BufferCPU<jvelType> &jvel_buffer,
                  BufferCPU<errorType> &res_buffer, DenseLinearProblem &hg)
    {
        for (int y = win.min_y; y < win.max_y; y += 1)
        {
            for (int x = win.min_x; x < win.max_x; x += 1)
            {
                vec6f J_pose = jpose_buffer.getTexel(y, x);
                vec6f J_vel = jvel_buffer.getTexel(y, x);
                float res = res_buffer.getTexel(y, x);
                if (J_pose == jpose_buffer.nodata || J_vel == jvel_buffer.nodata || res == res_buffer.nodata)
                    continue;
                float absres = std::fabs(res);
                float hw = 1.0;
                if (absres > mesh_vo::huber_thresh_pix)
                    hw = mesh_vo::huber_thresh_pix / absres;

                vecxf J(12);

                for (int i = 0; i < 6; i++)
                {
                    J(i) = J_pose(i);
                }
                for (int i = 0; i < 6; i++)
                {
                    J(i + 6) = J_vel(i);
                }

                hg.add(J, res, hw);
            }
        }
    }
}
*/

class HGMapReducerCPU : public BaseReducerCPU<JMapType, ImageType, ImageType, DenseLinearProblem>
{
public:
    void
    reducepartial(int min, int max,
                  BufferCPU<jposeType> &jpose_buffer,
                  BufferCPU<ImageType> &image_buffer,
                  BufferCPU<ImageType> &kimage_projected_buffer,
                  DenseLinearProblem &hg)
    {
        for (int i = min; i < max; i++)
        {
            JMapType jac = jmap_buffer[i];
            ImageType val = image_buffer[i];
            ImageType kval = kimage_projected_buffer[i];
            IdsType ids = pId_buffer.getTexel(y, x);

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

class HGPoseMapReducerCPU : public BaseReducerCPU<JMapType, ImageType, ImageType, DenseLinearProblem>
{
public:
    void reducepartial(int min, int max,
 int frameId, int numFrames, dataCPU<jposeType> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<errorType> &res_buffer, dataCPU<idsType> &pId_buffer, DenseLinearProblem &hg)
{
    for (int y = win.min_y; y < win.max_y; y += 1)
    {
        for (int x = win.min_x; x < win.max_x; x += 1)
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

            // hg.add(J_pose, J_map, res, hw, frameId, map_ids);

            vecxf J(6 + J_map.rows());
            vecxi ids(6 + J_map.rows());

            for (int i = 0; i < 6; i++)
            {
                J(i) = J_pose(i);
                ids(i) = frameId * 6 + i;
                ;
            }
            for (int i = 0; i < J_map.rows(); i++)
            {
                J(i + 6) = J_map(i);
                ids(i + 6) = map_ids(i) + numFrames * 6;
            }

            hg.add(J, res, hw, ids);
        }
    }
}

/*
void reduceHGIntrinsicPoseMapWindow(window<int> win, int frameId, int numFrames, dataCPU<jcamType> &jintrinsic_buffer, dataCPU<jposeType> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<errorType> &res_buffer, dataCPU<idsType> &pId_buffer, DenseLinearProblem &hg)
{
    for (int y = win.min_y; y < win.max_y; y += 1)
    {
        for (int x = win.min_x; x < win.max_x; x += 1)
        {
            jcamType J_int = jintrinsic_buffer.getTexel(y, x);
            jposeType J_pose = jpose_buffer.getTexel(y, x);
            jmapType J_map = jmap_buffer.getTexel(y, x);
            errorType res = res_buffer.getTexel(y, x);
            idsType map_ids = pId_buffer.getTexel(y, x);

            if (res == res_buffer.nodata || J_int == jintrinsic_buffer.nodata || J_pose == jpose_buffer.nodata || J_map == jmap_buffer.nodata || map_ids == pId_buffer.nodata)
                continue;

            float absres = std::fabs(res);
            float hw = 1.0;
            if (absres > mesh_vo::huber_thresh_pix)
                hw = mesh_vo::huber_thresh_pix / absres;

            // hg.add(J_pose, J_map, res, hw, frameId, map_ids);

            vecxf J(J_int.rows() + 6 + J_map.rows());
            vecxi ids(J_int.rows() + 6 + J_map.rows());

            for (int i = 0; i < J_int.rows(); i++)
            {
                J(i) = J_int(i);
                ids(i) = i;
            }
            for (int i = 0; i < 6; i++)
            {
                J(i + J_int.rows()) = J_pose(i);
                ids(i + J_int.rows()) = frameId * 6 + i + J_int.rows();
            }
            for (int i = 0; i < J_map.rows(); i++)
            {
                J(i + 6 + J_int.rows()) = J_map(i);
                ids(i + 6 + J_int.rows()) = map_ids(i) + numFrames * 6 + J_int.rows();
            }

            hg.add(J, res, hw, ids);
        }
    }
}
*/
