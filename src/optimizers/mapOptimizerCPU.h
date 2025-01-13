#pragma once

#include <Eigen/Core>
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
          j_buffer(_cam.width, _cam.height, jmapType(0.0)),
          pId_buffer(_cam.width, _cam.height, idsType(-1))
    {
        reguWeight = 10.0;
        priorWeight = 0.0;
    }

    void optimize(std::vector<frameCPU> &frames, frameCPU &kframe, sceneType &scene)
    {
        std::vector<int> sceneParamsIds = scene.getParamIds();
        int numMapParams = sceneParamsIds.size();

        Eigen::Matrix<float, numMapParams, 1>() init_sceneParams;
        for (int i = 0; i < sceneParamsIds.size(); i++)
        {
            float param = scene.getParam(sceneParamsIds[i]);
            init_sceneParams(i) = param;
        }

        if(invCovariance.rows() != numMapParams || invCovariance.cols() != numMapParams)
        {
            invCovariance = Eigen::Matrix<float, numMapParams, numMapParams>();
            for(int i = 0; i < numMapParams; i++)
            {
                invCovariance(i, i) = 0.0001;
            }
        }

        Eigen::Matrix<float, numMapParams, numMapParams> init_invcovariance = invCovariance;

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
                Eigen::Matrix<float, numMapParams, 1> sceneParams;
                for (size_t index = 0; index < sceneParamsIds.size(); index++)
                {
                    sceneParams(i) = scene.getParam(sceneParamId);
                }

                Eigen::Matrix<float, numMapParams, 1> res = sceneParams - init_sceneParams;
                Eigen::Matrix<float, numMapParams, 1> conv_dot_res = init_invcovariance * res;
                float priorError = priorWeight * (res.dot(conv_dot_res)) / numMapParams;

                e += priorError;
            }

            float last_error = e.getError();

            std::cout << "optMap initial error " << last_error << " " << lvl << std::endl;
            baseOptimizerCPU<sceneType>::plotDebug(scene, kframe, frames);

            int maxIterations = 1000;
            float lambda = 0.0;
            for (int it = 0; it < maxIterations; it++)
            {
                DenseLinearProblem hg(numMapParams);
                for (std::size_t i = 0; i < frames.size(); i++)
                {
                    DenseLinearProblem fhg = computeProblem(frames[i], kframe, scene, lvl);
                    assert(fhg.getCount() > 0.5 * baseOptimizerCPU<sceneType>::cam[lvl].width * baseOptimizerCPU<sceneType>::cam[lvl].height);
                    hg += fhg;
                }
                hg *= 1.0 / hg.getCount();

                if (reguWeight > 0.0)
                {
                    DenseLinearProblem hg_regu = scene.HGRegu(0);
                    assert(hg_regu.getCount() > 0);
                    hg_regu *= 1.0 / hg_regu.getCount();
                    hg_regu *= reguWeight;
                    hg += hg_regu;
                }

                if (priorWeight > 0.0)
                {
                    for (size_t index = 0; index < sceneParamsIds.size(); index++)
                    {
                        int sceneParamId = sceneParamsIds[index];
                        int linearProblemParamId = sceneParamId;

                        float res = scene.getParam(sceneParamId) - init_sceneParams[sceneParamId];

                        vec1<float> jacobian = 1.0;
                        vec1<int> ids = linearProblemParamId;
                        hg.add(jacobian, res, 1.0, ids);
                    }
                }

                // saveH(hg, "H.png");

                // std::vector<int> paramIds = hg.removeUnobservedParams();
                std::vector<int> linearProbleParamIds = hg.getParamIds();

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

                    if (!hg.prepareH(lambda))
                        continue;

                    Eigen::VectorXf inc = hg.solve();

                    std::vector<float> best_params;
                    float map_inc_mag = 0.0;
                    int map_inc_mag_count = 0;
                    for (size_t index = 0; index < linearProbleParamIds.size(); index++)
                    {
                        int paramId = linearProbleParamIds[index];

                        float best_param = scene.getParam(paramId);
                        float inc_param = inc(index);
                        float new_param = best_param - inc_param;

                        best_params.push_back(best_param);
                        // the derivative is with respecto to the keyframe pose
                        // the update should take this into account
                        scene.setParam(new_param, paramId);
                        map_inc_mag += inc_param * inc_param;
                        map_inc_mag_count += 1;
                    }
                    map_inc_mag /= map_inc_mag_count;

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
                        e += fe;
                    }
                    e *= 1.0 / e.getCount();

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
                        Error sceneInitialError;

                        for (size_t index = 0; index < sceneParamsIds.size(); index++)
                        {
                            int sceneParamId = sceneParamsIds[index];
                            float res = scene.getParam(sceneParamId) - init_sceneParams[sceneParamId];
                            sceneInitialError += res;
                        }

                        sceneInitialError *= 1.0 / sceneParamsIds.size();
                        sceneInitialError *= priorWeight;
                        e += sceneInitialError;
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

                        if (map_inc_mag < 1e-16)
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
