#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
// #include <Eigen/CholmodSupport>
//  #include <Eigen/SPQRSupport>
// #include <thread>

#include "common/camera.h"
#include "common/types.h"
#include "common/DenseLinearProblem.h"
// #include "common/SparseLinearProblem.h"
//  #include "common/HGMapped.h"
#include "common/Error.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/reduceCPU.h"
// #include "cpu/SceneBase.h"
// #include "cpu/ScenePatches.h"
#include "cpu/SceneMesh.h"
// #include "cpu/SceneSurfels.h"
// #include "cpu/SceneMeshSmooth.h"
#include "optimizers/baseOptimizerCPU.h"
#include "cpu/OpenCVDebug.h"
#include "params.h"

template <typename sceneType, typename jmapType, typename idsType>
class mapOptimizerCPU : public baseOptimizerCPU<sceneType>
{
public:
    mapOptimizerCPU(camera &_cam)
        : baseOptimizerCPU<sceneType>(_cam),
          j_buffer(_cam.width, _cam.height, jmapType::zero()),
          pId_buffer(_cam.width, _cam.height, idsType::zero())
    {
        reguWeight = 0.0;
        priorWeight = 1.0;
    }

    void optimize(std::vector<frameCPU> &frames, frameCPU &kframe, sceneType &scene)
    {
        std::vector<int> sceneParamsIds = scene.getParamIds();
        int numParams = sceneParamsIds.size();

        Eigen::VectorXf init_params = Eigen::VectorXf::Zero(numParams);
        for (size_t i = 0; i < sceneParamsIds.size(); i++)
        {
            init_params(i) = scene.getParam(sceneParamsIds[i]);
        }

        if (invCovariance.rows() != numParams || invCovariance.cols() != numParams)
        {
            invCovariance = Eigen::MatrixXf::Identity(numParams, numParams) / (INITIAL_PARAM_STD * INITIAL_PARAM_STD);
        }

        Eigen::MatrixXf init_invcovariance = invCovariance;
        Eigen::MatrixXf init_invcovariancesqrt = invCovariance.sqrt();

        for (int lvl = 1; lvl >= 1; lvl--)
        {
            Error e;
            for (std::size_t i = 0; i < frames.size(); i++)
            {
                Error ef = baseOptimizerCPU<sceneType>::computeError(frames[i], kframe, scene, lvl);
                assert(ef.getCount() > 0.5 * baseOptimizerCPU<sceneType>::cam[lvl].width * baseOptimizerCPU<sceneType>::cam[lvl].height);
                ef *= 1.0 / ef.getCount();
                e += ef;
            }
            e *= 1.0 / frames.size();

            if (reguWeight > 0.0)
            {
                Error e_regu = scene.errorRegu();
                assert(e_regu.getCount() > 0);
                e_regu *= 1.0 / e_regu.getCount();
                e_regu *= reguWeight;
                e += e_regu;
            }

            if (priorWeight > 0.0)
            {
                Eigen::VectorXf params(numParams);

                for (size_t index = 0; index < sceneParamsIds.size(); index++)
                {
                    params(index) = scene.getParam(sceneParamsIds[index]);
                }

                Eigen::VectorXf res = params - init_params;
                Eigen::VectorXf conv_dot_res = init_invcovariance * res;
                float weight = priorWeight / numParams;
                float priorError = weight * (res.dot(conv_dot_res));

                e += priorError;
            }

            float last_error = e.getError();

            std::cout << "optMap initial error " << last_error << " " << lvl << std::endl;
            baseOptimizerCPU<sceneType>::plotDebug(scene, kframe, frames);

            int maxIterations = 1000;
            float lambda = 0.0;
            for (int it = 0; it < maxIterations; it++)
            {
                DenseLinearProblem problem(numParams);
                for (std::size_t i = 0; i < frames.size(); i++)
                {
                    DenseLinearProblem fhg = computeProblem(frames[i], kframe, scene, lvl);
                    assert(fhg.getCount() > 0.5 * baseOptimizerCPU<sceneType>::cam[lvl].width * baseOptimizerCPU<sceneType>::cam[lvl].height);
                    fhg *= 1.0 / fhg.getCount();
                    problem += fhg;
                }
                problem *= 1.0 / frames.size();

                if (reguWeight > 0.0)
                {
                    DenseLinearProblem hg_regu = scene.HGRegu(0);
                    assert(hg_regu.getCount() > 0);
                    hg_regu *= 1.0 / hg_regu.getCount();
                    hg_regu *= reguWeight;
                    problem += hg_regu;
                }

                if (priorWeight > 0.0)
                {
                    Eigen::VectorXf params = Eigen::VectorXf::Zero(numParams);

                    for (size_t i = 0; i < numParams; i++)
                    {
                        params(i) = scene.getParam(sceneParamsIds[i]);
                    }
                    // error = (sqrt(H)*diff)**2
                    // jacobian = sqrt(H)*ones

                    Eigen::VectorXf res = init_invcovariancesqrt * (params - init_params);
                    Eigen::MatrixXf jacobian = init_invcovariancesqrt;
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

                    Eigen::VectorXf inc = problem.solve();

                    std::vector<float> best_params;

                    for (size_t index = 0; index < linearProbleParamIds.size(); index++)
                    {
                        int paramId = linearProbleParamIds[index];

                        best_params.push_back(scene.getParam(paramId));
                        // the derivative is with respecto to the keyframe pose
                        // the update should take this into account
                        scene.setParam(scene.getParam(paramId) - inc(index), paramId);
                    }

                    e.setZero();
                    for (std::size_t i = 0; i < frames.size(); i++)
                    {
                        Error fe = baseOptimizerCPU<sceneType>::computeError(frames[i], kframe, scene, lvl);
                        if (fe.getCount() < 0.5 * baseOptimizerCPU<sceneType>::cam[lvl].width * baseOptimizerCPU<sceneType>::cam[lvl].height)
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
                        Error e_regu = scene.errorRegu();
                        assert(e_regu.getCount() > 0);
                        e_regu *= 1.0 / e_regu.getCount();
                        e_regu *= reguWeight;
                        e += e_regu;
                    }

                    if (priorWeight > 0.0)
                    {
                        Eigen::VectorXf params(numParams);
                        for (size_t index = 0; index < sceneParamsIds.size(); index++)
                        {
                            params(index) = scene.getParam(sceneParamsIds[index]);
                        }

                        Eigen::VectorXf res = params - init_params;
                        Eigen::VectorXf conv_dot_res = init_invcovariance * res;
                        float weight = priorWeight / numParams;
                        float priorError = weight * (res.dot(conv_dot_res));

                        e += priorError;
                    }

                    float error = e.getError();

                    std::cout << "new error " << error << " " << lambda << " " << it << " " << n_try << " lvl: " << lvl << std::endl;
                    baseOptimizerCPU<sceneType>::plotDebug(scene, kframe, frames);

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
                        }

                        // if update accepted, do next iteration
                        break;
                    }
                    else
                    {
                        for (size_t index = 0; index < linearProbleParamIds.size(); index++)
                        {
                            scene.setParam(best_params[index], linearProbleParamIds[index]);
                        }

                        // reject update, increase lambda, use un-updated data

                        if (inc.dot(inc) < 1e-16)
                        {
                            // if too small, do next level!
                            it = maxIterations;
                            break;
                        }
                    }
                }
            }
        }
    }

private:
    DenseLinearProblem computeProblem(frameCPU &frame, frameCPU &kframe, sceneType &scene, int lvl)
    {
        baseOptimizerCPU<sceneType>::error_buffer.setToNoData(lvl);
        j_buffer.setToNoData(lvl);
        pId_buffer.setToNoData(lvl);

        int numMapParams = scene.getParamIds().size();

        baseOptimizerCPU<sceneType>::renderer.renderJMapParallel(scene, kframe.getRawImage(lvl), kframe.getAffine(), kframe.getPose(), frame.getRawImage(lvl), frame.getAffine(), frame.getdIdpixImage(lvl), frame.getPose(), baseOptimizerCPU<sceneType>::cam[lvl], j_buffer.get(lvl), baseOptimizerCPU<sceneType>::error_buffer.get(lvl), pId_buffer.get(lvl));
        DenseLinearProblem problem = baseOptimizerCPU<sceneType>::reducer.reduceHGMapParallel(numMapParams, j_buffer.get(lvl), baseOptimizerCPU<sceneType>::error_buffer.get(lvl), pId_buffer.get(lvl));

        return problem;
    }

    float reguWeight;
    float priorWeight;

    dataMipMapCPU<jmapType> j_buffer;
    dataMipMapCPU<idsType> pId_buffer;

    Eigen::MatrixXf invCovariance;
};
