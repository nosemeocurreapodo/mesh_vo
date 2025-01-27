#pragma once

#include "params.h"
#include "common/camera.h"
#include "common/types.h"
#include "common/Error.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/reduceCPU.h"
#include "common/DenseLinearProblem.h"
#include "optimizers/baseOptimizerCPU.h"
#include "cpu/OpenCVDebug.h"

class poseMapOptimizerCPU : public baseOptimizerCPU
{
public:
    poseMapOptimizerCPU(camera &_cam)
        : baseOptimizerCPU(_cam),
          jpose_buffer(_cam.width, _cam.height, vec6f::Zero()),
          jmap_buffer(_cam.width, _cam.height, jmapType::Zero()),
          pId_buffer(_cam.width, _cam.height, idsType::Zero())
    {
        reguWeight = 0.0;
        priorWeight = 2.0;
    }

    void optimize(std::vector<frameCPU> &frames, keyFrameCPU &kframe)
    {
        //do a line search of any un-initialized depths
        //renderer.renderIdepthLineSearch(kframe, frames[frames.size()-1], cam, 1);

        std::vector<int> mapParamsIds = kframe.getGeometry().getParamIds();
        int numPoseParams = frames.size() * 6;
        int numMapParams = mapParamsIds.size();
        int numParams = numPoseParams + numMapParams;

        invCovariance = matxf::Identity(numParams, numParams);

        vecxf init_params = vecxf::Zero(numParams);

        for (size_t i = 0; i < frames.size(); i++)
        {
            init_params.segment<6>(i * 6) = frames[i].getLocalPose().log();
            for (int j = 0; j < 6; j++)
            {
                invCovariance(i * 6 + j, i * 6 + j) = 1.0 / (INITIAL_POSE_STD * INITIAL_POSE_STD);
            }
        }

        for (size_t i = 0; i < mapParamsIds.size(); i++)
        {
            init_params(i + numPoseParams) = kframe.getGeometry().getDepthParam(mapParamsIds[i]);
            invCovariance(i + numPoseParams, i + numPoseParams) = kframe.getGeometry().getWeightParam(mapParamsIds[i]);
            //invCovariance(i + numPoseParams, i + numPoseParams) = 1.0 / (INITIAL_PARAM_STD * INITIAL_PARAM_STD);
        }

        // invCovariance.block(0, 0, numPoseParams, numPoseParams) *= 1.0 / (INITIAL_POSE_STD * INITIAL_POSE_STD);
        // invCovariance.block(numPoseParams, numPoseParams, numMapParams, numMapParams) *= 1.0 / (INITIAL_PARAM_STD * INITIAL_PARAM_STD);

        matxf init_invcovariance = invCovariance;
        matxf init_invcovariancesqrt;

        if (priorWeight > 0.0)
            init_invcovariancesqrt = invCovariance.sqrt();

        for (int lvl = 1; lvl >= 1; lvl--)
        {
            Error e;
            for (std::size_t i = 0; i < frames.size(); i++)
            {
                Error ef = computeError(frames[i], kframe, lvl);
                assert(ef.getCount() > 0.5 * cam[lvl].width * cam[lvl].height);
                ef *= 1.0 / ef.getCount();
                e += ef;
            }
            e *= 1.0 / frames.size();

            if (reguWeight > 0.0)
            {
                Error e_regu = kframe.getGeometry().errorRegu();
                assert(e_regu.getCount() > 0);
                e_regu *= reguWeight / e_regu.getCount();
                e += e_regu;
            }

            if (priorWeight > 0.0)
            {
                vecxf params(numParams);
                for (size_t index = 0; index < frames.size(); index++)
                {
                    params.segment(index * 6, 6) = frames[index].getLocalPose().log();
                }

                for (size_t index = 0; index < mapParamsIds.size(); index++)
                {
                    params(index + numPoseParams) = kframe.getGeometry().getDepthParam(mapParamsIds[index]);
                }

                vecxf res = params - init_params;
                vecxf conv_dot_res = init_invcovariance * res;
                float weight = priorWeight / numParams;
                float priorError = weight * (res.dot(conv_dot_res));

                e += priorError;
            }

            float last_error = e.getError();

            std::cout << "optPoseMap initial error " << last_error << " " << lvl << std::endl;
            plotDebug(kframe, frames, "poseMapOptimizerCPU");

            int maxIterations = 1000;
            float lambda = 0.0;
            for (int it = 0; it < maxIterations; it++)
            {
                DenseLinearProblem problem(numParams);
                for (std::size_t i = 0; i < frames.size(); i++)
                {
                    DenseLinearProblem fhg = computeProblem(frames[i], kframe, i, frames.size(), lvl);
                    assert(fhg.getCount() > 0.5 * cam[lvl].width * cam[lvl].height);
                    fhg *= 1.0 / fhg.getCount();
                    problem += fhg;
                }
                problem *= 1.0 / frames.size();

                if (reguWeight > 0.0)
                {
                    DenseLinearProblem hg_regu = kframe.getGeometry().HGRegu(numPoseParams);
                    assert(hg_regu.getCount() > 0);
                    hg_regu *= reguWeight / hg_regu.getCount();
                    problem += hg_regu;
                }

                if (priorWeight > 0.0)
                {
                    vecxf params = vecxf::Zero(numParams);

                    for (size_t index = 0; index < frames.size(); index++)
                    {
                        params.segment(index * 6, 6) = frames[index].getLocalPose().log();
                    }

                    for (size_t i = 0; i < mapParamsIds.size(); i++)
                    {
                        params(i + numPoseParams) = kframe.getGeometry().getDepthParam(mapParamsIds[i]);
                    }
                    // error = (sqrt(H)*diff)**2
                    // jacobian = sqrt(H)*ones

                    vecxf res = init_invcovariancesqrt * (params - init_params);
                    matxf jacobian = init_invcovariancesqrt;
                    // vecx<float> res(_res);
                    // matx<float> jacobian(_jacobian);
                    float weight = priorWeight / numParams;
                    problem.add(jacobian, res, weight);
                }

                // saveH(hg, "H.png");

                std::vector<int> linearProbleParamIds = problem.getParamIds();

                int n_try = 0;
                lambda = 0.0;
                while (true)
                {
                    if (n_try > 0)
                    {
                        if (lambda < MIN_LAMBDA)
                            lambda = MIN_LAMBDA;
                        lambda *= std::pow(2.0, n_try);
                    }
                    n_try++;

                    if (!problem.prepareH(lambda))
                        continue;

                    vecxf inc = problem.solve();

                    vecxf poseInc = inc.segment(0, numPoseParams);
                    vecxf mapInc = inc.segment(numPoseParams, numMapParams);

                    std::vector<Sophus::SE3f> best_poses;
                    for (size_t i = 0; i < frames.size(); i++)
                    {
                        best_poses.push_back(frames[i].getLocalPose());
                        SE3f new_pose = frames[i].getLocalPose() * SE3f::exp(poseInc.segment(i * 6, 6)).inverse();
                        frames[i].setLocalPose(new_pose);
                    }

                    std::vector<float> best_mapParams;
                    for (size_t i = 0; i < mapParamsIds.size(); i++)
                    {
                        best_mapParams.push_back(kframe.getGeometry().getDepthParam(mapParamsIds[i]));
                        kframe.getGeometry().setDepthParam(kframe.getGeometry().getDepthParam(mapParamsIds[i]) - mapInc(i), mapParamsIds[i]);
                        //kframe.getGeometry().setWeightParam(problem.getH()(mapParamsIds[i], mapParamsIds[i]), mapParamsIds[i]);
                        kframe.getGeometry().setWeightParam(1.0/(GOOD_PARAM_STD*GOOD_PARAM_STD), mapParamsIds[i]);
                    }

                    e.setZero();
                    for (std::size_t i = 0; i < frames.size(); i++)
                    {
                        Error fe = computeError(frames[i], kframe, lvl);
                        if (fe.getCount() < 0.5 * cam[lvl].width * cam[lvl].height)
                        {
                            // too few pixels, unreliable, set to large error
                            fe.setZero();
                            fe += last_error;
                        }
                        fe *= 1.0 / fe.getCount();
                        e += fe;
                    }
                    e *= 1.0 / frames.size();

                    if (reguWeight > 0.0)
                    {
                        Error e_regu = kframe.getGeometry().errorRegu();
                        assert(e_regu.getCount() > 0);
                        e_regu *= reguWeight / e_regu.getCount();
                        e += e_regu;
                    }

                    if (priorWeight > 0.0)
                    {
                        vecxf params(numParams);

                        for (size_t index = 0; index < frames.size(); index++)
                        {
                            params.segment(index * 6, 6) = frames[index].getLocalPose().log();
                        }

                        for (size_t index = 0; index < mapParamsIds.size(); index++)
                        {
                            params(index + numPoseParams) = kframe.getGeometry().getDepthParam(mapParamsIds[index]);
                        }

                        vecxf res = params - init_params;
                        vecxf conv_dot_res = init_invcovariance * res;
                        float weight = priorWeight / numParams;
                        float priorError = weight * (res.dot(conv_dot_res));

                        e += priorError;
                    }

                    float error = e.getError();

                    std::cout << "new error " << error << " " << lambda << " " << it << " " << n_try << " lvl: " << lvl << std::endl;
                    plotDebug(kframe, frames, "poseMapOptimizerCPU");

                    if (error < last_error)
                    {
                        // accept update, decrease lambda
                        float p = error / last_error;

                        last_error = error;

                        if (p > 0.999f)
                        {
                            // std::cout << "lvl " << lvl << " converged after " << it << " itarations with lambda " << lambda << std::endl;
                            //  if converged, do next level
                            it = maxIterations;
                            std::cout << "converged p:" << p << std::endl;
                        }

                        // if update accepted, do next iteration
                        break;
                    }
                    else
                    {
                        for (size_t i = 0; i < frames.size(); i++)
                        {
                            frames[i].setLocalPose(best_poses[i]);
                        }

                        for (size_t i = 0; i < mapParamsIds.size(); i++)
                        {
                            kframe.getGeometry().setDepthParam(best_mapParams[i], mapParamsIds[i]);
                        }

                        // reject update, increase lambda, use un-updated data

                        if (poseInc.dot(poseInc) < 1e-16 && mapInc.dot(mapInc) < 1e-16)
                        {
                            // if too small, do next level!
                            it = maxIterations;
                            std::cout << "too small " << poseInc.dot(poseInc) << " " << mapInc.dot(mapInc) << std::endl;
                            break;
                        }
                    }
                }
            }
        }
    }

private:
    DenseLinearProblem computeProblem(frameCPU &frame, keyFrameCPU &kframe, int frameId, int numFrames, int lvl)
    {
        error_buffer.setToNoData(lvl);
        jpose_buffer.setToNoData(lvl);
        jmap_buffer.setToNoData(lvl);
        pId_buffer.setToNoData(lvl);

        int numMapParams = kframe.getGeometry().getParamIds().size();

        renderer.renderJPoseMapParallel(kframe, frame, jpose_buffer, jmap_buffer, error_buffer, pId_buffer, cam, lvl);
        DenseLinearProblem problem = reducer.reduceHGPoseMapParallel(frameId, numFrames, numMapParams, jpose_buffer.get(lvl), jmap_buffer.get(lvl), error_buffer.get(lvl), pId_buffer.get(lvl));

        return problem;
    }

    float reguWeight;
    float priorWeight;

    dataMipMapCPU<vec6f> jpose_buffer;
    dataMipMapCPU<jmapType> jmap_buffer;
    dataMipMapCPU<idsType> pId_buffer;

    matxf invCovariance;
};
