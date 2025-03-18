#include "optimizers/mapOptimizerCPU.h"

mapOptimizerCPU::mapOptimizerCPU(int width, int height, bool _printLog)
    : baseOptimizerCPU(width, height),
      jmap_buffer(width, height, jmapType::Zero()),
      pId_buffer(width, height, idsType::Zero())
{
    printLog = _printLog;
}

void mapOptimizerCPU::init(std::vector<frameCPU> &frames, keyFrameCPU &kframe, cameraType &cam, int lvl)
{
    std::vector<int> mapParamsIds = kframe.getGeometry().getParamIds();
    int numParams = mapParamsIds.size();

    invCovariance = matxf::Identity(numParams, numParams);

    init_params = vecxf::Zero(numParams);

    for (size_t i = 0; i < mapParamsIds.size(); i++)
    {
        // init_params(i + numPoseParams) = kframe.getGeometry().getDepthParam(mapParamsIds[i]);
        invCovariance(i , i) = kframe.getGeometry().getWeightParam(mapParamsIds[i]);
        // invCovariance(i + numPoseParams, i + numPoseParams) = 1.0 / mesh_vo::mapping_param_initial_var;
    }

    // invCovariance.block(0, 0, numPoseParams, numPoseParams) *= 1.0 / mesh_vo::mapping_pose_var;
    // invCovariance.block(numPoseParams, numPoseParams, numMapParams, numMapParams) *= 1.0 / mesh_vo::initial_param_var;

    init_invcovariance = invCovariance;

    if (mesh_vo::mapping_prior_weight > 0.0)
        init_invcovariancesqrt = invCovariance.sqrt();

    init_error = 0;
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        Error ef = computeError(frames[i], kframe, cam, lvl);
        // assert(ef.getCount() > 0.5 * cam[lvl].width * cam[lvl].height);
        init_error += ef.getError() / ef.getCount();
    }
    init_error *= 1.0 / frames.size();

    if (mesh_vo::mapping_regu_weight > 0.0)
    {
        Error e_regu = kframe.getGeometry().errorRegu();
        assert(e_regu.getCount() > 0);
        init_error += mesh_vo::mapping_regu_weight * e_regu.getError() / e_regu.getCount();
    }

    if (mesh_vo::mapping_prior_weight > 0.0)
    {
        vecxf params(numParams);

        for (size_t index = 0; index < mapParamsIds.size(); index++)
        {
            params(index) = kframe.getGeometry().getDepthParam(mapParamsIds[index]);
        }

        vecxf res = params - init_params;
        vecxf conv_dot_res = init_invcovariance * res;
        float weight = mesh_vo::mapping_prior_weight / numParams;
        float priorError = weight * (res.dot(conv_dot_res));

        init_error += priorError;
    }

    if(printLog)
        std::cout << "mapOptimizer initial error " << init_error << " " << lvl << std::endl;
    
        reachedConvergence = false;
}

void mapOptimizerCPU::step(std::vector<frameCPU> &frames, keyFrameCPU &kframe, cameraType &cam, int lvl)
{
    std::vector<int> mapParamsIds = kframe.getGeometry().getParamIds();
    int numParams = mapParamsIds.size();

    DenseLinearProblem problem(numParams);
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        DenseLinearProblem fhg = computeProblem(frames[i], kframe, cam, lvl);
        // assert(fhg.getCount() > 0.5 * cam[lvl].width * cam[lvl].height);
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

        for (size_t i = 0; i < mapParamsIds.size(); i++)
        {
            params(i) = kframe.getGeometry().getDepthParam(mapParamsIds[i]);
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

    std::vector<int> linearProbleParamIds = problem.getParamIds();

    int n_try = 0;
    float lambda = 0.0;
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

        std::vector<float> best_mapParams;
        for (size_t i = 0; i < mapParamsIds.size(); i++)
        {
            best_mapParams.push_back(kframe.getGeometry().getDepthParam(mapParamsIds[i]));
            kframe.getGeometry().setDepthParam(kframe.getGeometry().getDepthParam(mapParamsIds[i]) - inc(i), mapParamsIds[i]);
            // kframe.getGeometry().setWeightParam(problem.getH()(mapParamsIds[i], mapParamsIds[i]), mapParamsIds[i]);
            kframe.getGeometry().setWeightParam(1.0 / mesh_vo::mapping_param_good_var, mapParamsIds[i]);
        }

        float error = 0;
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            Error fe = computeError(frames[i], kframe, cam, lvl);
            if (fe.getCount() < 0.5 * frames[i].getRawImage(lvl).width * frames[i].getRawImage(lvl).height)
            {
                // too few pixels, unreliable, set to large error
                error += init_error * 2.0;
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
            vecxf params(numParams);

            for (size_t index = 0; index < mapParamsIds.size(); index++)
            {
                params(index) = kframe.getGeometry().getDepthParam(mapParamsIds[index]);
            }

            vecxf res = params - init_params;
            vecxf conv_dot_res = init_invcovariance * res;
            float weight = mesh_vo::mapping_prior_weight / numParams;
            float priorError = weight * (res.dot(conv_dot_res));

            error += priorError;
        }

        if(printLog)
            std::cout << "mapOptimizer new error " << error << " " << lambda << " " << n_try << " lvl: " << lvl << " mesh_regu: " << mesh_vo::mapping_regu_weight << std::endl;

        if (error <= init_error)
        {
            // accept update, decrease lambda
            float p = error / init_error;

            init_error = error;

            if (p >= mesh_vo::mapping_convergence_p)
            {
                // std::cout << "lvl " << lvl << " converged after " << it << " itarations with lambda " << lambda << std::endl;
                //  if converged, do next level
                reachedConvergence = true;
                if(printLog)
                    std::cout << "poseMapOptimizer converged p:" << p << std::endl;
            }

            // if update accepted, do next iteration
            break;
        }
        else
        {
            for (size_t i = 0; i < mapParamsIds.size(); i++)
            {
                kframe.getGeometry().setDepthParam(best_mapParams[i], mapParamsIds[i]);
            }

            // reject update, increase lambda, use un-updated data
            float incMag = inc.dot(inc)/numParams;

            if (incMag <= mesh_vo::mapping_convergence_m_v)
            {
                // if too small, do next level!
                reachedConvergence = true;
                if(printLog)
                    std::cout << "mapOptimizer too small " << incMag << std::endl;
                break;
            }
        }
    }
}

DenseLinearProblem mapOptimizerCPU::computeProblem(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl)
{
    error_buffer.setToNoData(lvl);
    jmap_buffer.setToNoData(lvl);
    pId_buffer.setToNoData(lvl);

    int numMapParams = kframe.getGeometry().getParamIds().size();

    renderer.renderJMapParallel(kframe, frame, jmap_buffer, error_buffer, pId_buffer, cam, lvl);
    DenseLinearProblem problem = reducer.reduceHGMapParallel(numMapParams, jmap_buffer.get(lvl), error_buffer.get(lvl), pId_buffer.get(lvl));

    return problem;
}

std::vector<dataCPU<float>> mapOptimizerCPU::getDebugData(std::vector<frameCPU> &frames, keyFrameCPU &kframe, cameraType &cam, int lvl)
{
    std::vector<dataCPU<float>> toShow;

    toShow.push_back(kframe.getRawImage(lvl).convert<float>());

    depth_buffer.setToNoData(lvl);
    //weight_buffer.setToNoData(lvl);

    renderer.renderDepthParallel(kframe, SE3f(), depth_buffer, cam, lvl);
    //renderer.renderWeightParallel(kframe, SE3f(), weight_buffer, cam, lvl);

    depth_buffer.get(lvl).invert();

    toShow.push_back(depth_buffer.get(lvl));
    //toShow.push_back(weight_buffer.get(lvl));

    for (frameCPU frame : frames)
    {
        error_buffer.setToNoData(lvl);
        // depth_buffer.setToNoData(lvl);
        renderer.renderResidualParallel(kframe, frame, error_buffer, cam, lvl);
        // renderer.renderDepthParallel(kframe, frames[i].getLocalPose(), depth_buffer, cam, lvl);
        toShow.push_back(frame.getRawImage(lvl).convert<float>());
        toShow.push_back(error_buffer.get(lvl).convert<float>());
        // toShow.push_back(depth_buffer.get(lvl));
    }

    return toShow;
}