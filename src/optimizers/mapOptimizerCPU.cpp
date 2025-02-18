#include "optimizers/mapOptimizerCPU.h"

mapOptimizerCPU::mapOptimizerCPU(int width, int height)
    : baseOptimizerCPU(width, height),
      j_buffer(width, height, jmapType::Zero()),
      pId_buffer(width, height, idsType::Zero())
{
}

void mapOptimizerCPU::optimize(std::vector<frameCPU> &frames, keyFrameCPU &kframe, camera &cam)
{
    std::vector<int> sceneParamsIds = kframe.getGeometry().getParamIds();
    int numParams = sceneParamsIds.size();

    invCovariance = matxf::Identity(numParams, numParams) / mesh_vo::mapping_param_initial_var;

    Eigen::VectorXf init_params = Eigen::VectorXf::Zero(numParams);
    for (size_t i = 0; i < sceneParamsIds.size(); i++)
    {
        init_params(i) = kframe.getGeometry().getDepthParam(sceneParamsIds[i]);
        invCovariance(i, i) = kframe.getGeometry().getWeightParam(sceneParamsIds[i]);
        // invCovariance(i, i) = 1.0 / mesh_vo::mapping_param_initial_var;
    }

    matxf init_invcovariance = invCovariance;
    matxf init_invcovariancesqrt;

    if (mesh_vo::mapping_prior_weight > 0)
        init_invcovariancesqrt = invCovariance.sqrt();

    for (int lvl = 1; lvl >= 1; lvl--)
    {
        float last_error = 0;
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            Error ef = computeError(frames[i], kframe, cam, lvl);
            //assert(ef.getCount() > 0.5 * cam[lvl].width * cam[lvl].height);
            last_error += ef.getError() / ef.getCount();
        }
        last_error *= 1.0 / frames.size();

        if (mesh_vo::mapping_regu_weight > 0.0)
        {
            Error e_regu = kframe.getGeometry().errorRegu();
            assert(e_regu.getCount() > 0);
            last_error += mesh_vo::mapping_regu_weight * e_regu.getError() / e_regu.getCount();
        }

        if (mesh_vo::mapping_prior_weight > 0.0)
        {
            vecxf params(numParams);

            for (size_t index = 0; index < sceneParamsIds.size(); index++)
            {
                params(index) = kframe.getGeometry().getDepthParam(sceneParamsIds[index]);
            }

            vecxf res = params - init_params;
            vecxf conv_dot_res = init_invcovariance * res;
            float weight = mesh_vo::mapping_prior_weight / numParams;
            float priorError = weight * (res.dot(conv_dot_res));

            last_error += priorError;
        }

        std::cout << "optMap initial error " << last_error << " " << lvl << std::endl;
        plotDebug(kframe, frames, cam, "mapOptimizerCPU");

        int maxIterations = 1000;
        float lambda = 0.0;
        for (int it = 0; it < maxIterations; it++)
        {
            DenseLinearProblem problem(numParams);
            for (std::size_t i = 0; i < frames.size(); i++)
            {
                DenseLinearProblem fhg = computeProblem(frames[i], kframe, cam, lvl);
                //assert(fhg.getCount() > 0.5 * cam[lvl].width * cam[lvl].height);
                fhg *= 1.0 / fhg.getCount();
                problem += fhg;
            }
            problem *= 1.0 / frames.size();

            if (mesh_vo::mapping_regu_weight > 0.0)
            {
                float weight = mesh_vo::mapping_regu_weight / numParams;
                DenseLinearProblem hg_regu = kframe.getGeometry().HGRegu(0, weight);
                assert(hg_regu.getCount() > 0);
                problem += hg_regu;
            }

            if (mesh_vo::mapping_prior_weight > 0.0)
            {
                vecxf params = vecxf::Zero(numParams);

                for (size_t i = 0; i < numParams; i++)
                {
                    params(i) = kframe.getGeometry().getDepthParam(sceneParamsIds[i]);
                }
                // error = (sqrt(H)*diff)**2
                // jacobian = sqrt(H)*ones

                vecxf res = init_invcovariancesqrt * (params - init_params);
                matxf jacobian = init_invcovariancesqrt;
                // vecx<float> res(_res);
                // matx<float> jacobian(_jacobian);
                float weight = mesh_vo::mapping_prior_weight / numParams;
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
                    if (lambda < mesh_vo::min_lambda)
                        lambda = mesh_vo::min_lambda;
                    lambda *= std::pow(2.0, n_try);
                }
                n_try++;

                if (!problem.prepareH(lambda))
                    continue;

                vecxf inc = problem.solve();

                std::vector<float> best_params;

                for (size_t index = 0; index < linearProbleParamIds.size(); index++)
                {
                    int paramId = linearProbleParamIds[index];

                    best_params.push_back(kframe.getGeometry().getDepthParam(paramId));
                    // the derivative is with respecto to the keyframe pose
                    // the update should take this into account
                    kframe.getGeometry().setDepthParam(kframe.getGeometry().getDepthParam(paramId) - inc(index), paramId);
                    // kframe.getGeometry().setWeightParam(problem.getH()(paramId, paramId), paramId);
                    kframe.getGeometry().setWeightParam(1.0 / mesh_vo::mapping_param_good_var, paramId);
                }

                float error = 0;
                for (std::size_t i = 0; i < frames.size(); i++)
                {
                    Error fe = computeError(frames[i], kframe, cam, lvl);
                    if (fe.getCount() < 0.5 * frames[i].getRawImage(lvl).width * frames[i].getRawImage(lvl).height)
                    {
                        // too few pixels, unreliable, set to large error
                        error += last_error;
                    }
                    else
                    {
                        error += fe.getError() / fe.getCount();
                    }
                }
                error *= 1.0 / frames.size();

                if (mesh_vo::mapping_regu_weight > 0.0)
                {
                    Error e_regu = kframe.getGeometry().errorRegu();
                    assert(e_regu.getCount() > 0);
                    error += mesh_vo::mapping_regu_weight * e_regu.getError() / e_regu.getCount();
                }

                if (mesh_vo::mapping_prior_weight > 0.0)
                {
                    Eigen::VectorXf params(numParams);
                    for (size_t index = 0; index < sceneParamsIds.size(); index++)
                    {
                        params(index) = kframe.getGeometry().getDepthParam(sceneParamsIds[index]);
                    }

                    Eigen::VectorXf res = params - init_params;
                    Eigen::VectorXf conv_dot_res = init_invcovariance * res;
                    float weight = mesh_vo::mapping_prior_weight / numParams;
                    float priorError = weight * (res.dot(conv_dot_res));

                    error += priorError;
                }

                std::cout << "new error " << error << " " << lambda << " " << it << " " << n_try << " lvl: " << lvl << std::endl;
                plotDebug(kframe, frames, cam, "mapOptimizerCPU");

                if (error <= last_error)
                {
                    // accept update, decrease lambda
                    float p = error / last_error;

                    last_error = error;

                    if (p > mesh_vo::mapping_convergence_p)
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
                        kframe.getGeometry().setDepthParam(best_params[index], linearProbleParamIds[index]);
                    }

                    // reject update, increase lambda, use un-updated data

                    //if (inc.dot(inc) < 1e-16)
                    if(false)
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

DenseLinearProblem mapOptimizerCPU::computeProblem(frameCPU &frame, keyFrameCPU &kframe, camera &cam, int lvl)
{
    error_buffer.setToNoData(lvl);
    j_buffer.setToNoData(lvl);
    pId_buffer.setToNoData(lvl);

    int numMapParams = kframe.getGeometry().getParamIds().size();

    renderer.renderJMapParallel(kframe, frame, j_buffer, error_buffer, pId_buffer, cam, lvl);
    DenseLinearProblem problem = reducer.reduceHGMapParallel(numMapParams, j_buffer.get(lvl), error_buffer.get(lvl), pId_buffer.get(lvl));

    return problem;
}
