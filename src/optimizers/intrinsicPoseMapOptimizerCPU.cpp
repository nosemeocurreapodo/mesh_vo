#include "optimizers/intrinsicPoseMapOptimizerCPU.h"

intrinsicPoseMapOptimizerCPU::intrinsicPoseMapOptimizerCPU(camera &_cam)
    : baseOptimizerCPU(_cam),
      jintrinsic_buffer(_cam.width, _cam.height, vec4f::Zero()),
      jpose_buffer(_cam.width, _cam.height, vec6f::Zero()),
      jmap_buffer(_cam.width, _cam.height, jmapType::Zero()),
      pId_buffer(_cam.width, _cam.height, idsType::Zero())
{
}

void intrinsicPoseMapOptimizerCPU::optimize(std::vector<frameCPU> &frames, keyFrameCPU &kframe)
{
    std::vector<int> mapParamsIds = kframe.getGeometry().getParamIds();
    int numIntrinsicParams = 4;
    int numPoseParams = frames.size() * 6;
    int numMapParams = mapParamsIds.size();
    int numParams = numIntrinsicParams + numPoseParams + numMapParams;

    invCovariance = matxf::Identity(numParams, numParams);

    vecxf init_params = vecxf::Zero(numParams);

    for (int lvl = 1; lvl >= 1; lvl--)
    {
        init_params(0) = cam[lvl].fx;
        init_params(1) = cam[lvl].fy;
        init_params(2) = cam[lvl].cx;
        init_params(3) = cam[lvl].cy;
    
        invCovariance(0, 0) = 1.0 / mesh_vo::mapping_intrinsic_initial_var;
        invCovariance(1, 1) = 1.0 / mesh_vo::mapping_intrinsic_initial_var;
        invCovariance(2, 2) = 1.0 / mesh_vo::mapping_intrinsic_initial_var;
        invCovariance(3, 3) = 1.0 / mesh_vo::mapping_intrinsic_initial_var;
    
        for (size_t i = 0; i < frames.size(); i++)
        {
            init_params.segment(i * 6 + numIntrinsicParams, 6) = frames[i].getLocalPose().log();
            for (int j = 0; j < 6; j++)
            {
                invCovariance(i * 6 + j + numIntrinsicParams, i * 6 + j + numIntrinsicParams) = 1.0 / mesh_vo::mapping_pose_initial_var;
            }
        }
    
        for (size_t i = 0; i < mapParamsIds.size(); i++)
        {
            // init_params(i + numPoseParams) = kframe.getGeometry().getDepthParam(mapParamsIds[i]);
            invCovariance(i + numPoseParams + numIntrinsicParams, i + numPoseParams + numIntrinsicParams) = kframe.getGeometry().getWeightParam(mapParamsIds[i]);
            //invCovariance(i + numPoseParams, i + numPoseParams) = 1.0 / mesh_vo::mapping_param_initial_var;
        }
    
        // invCovariance.block(0, 0, numPoseParams, numPoseParams) *= 1.0 / mesh_vo::mapping_pose_var;
        // invCovariance.block(numPoseParams, numPoseParams, numMapParams, numMapParams) *= 1.0 / mesh_vo::initial_param_var;
    
        matxf init_invcovariance = invCovariance;
        matxf init_invcovariancesqrt;
    
        if (mesh_vo::mapping_prior_weight > 0.0)
            init_invcovariancesqrt = invCovariance.sqrt();

        Error e;
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            Error ef = computeError(frames[i], kframe, lvl);
            assert(ef.getCount() > 0.5 * cam[lvl].width * cam[lvl].height);
            ef *= 1.0 / ef.getCount();
            e += ef;
        }
        e *= 1.0 / frames.size();

        if (mesh_vo::mapping_regu_weight > 0.0)
        {
            Error e_regu = kframe.getGeometry().errorRegu();
            assert(e_regu.getCount() > 0);
            e_regu *= mesh_vo::mapping_regu_weight / e_regu.getCount();
            e += e_regu;
        }

        if (mesh_vo::mapping_prior_weight > 0.0)
        {
            vecxf params(numParams);

            params(0) = cam[lvl].fx;
            params(1) = cam[lvl].fy;
            params(2) = cam[lvl].cx;
            params(3) = cam[lvl].cy;

            for (size_t index = 0; index < frames.size(); index++)
            {
                params.segment(index * 6 + numIntrinsicParams, 6) = frames[index].getLocalPose().log();
            }

            for (size_t index = 0; index < mapParamsIds.size(); index++)
            {
                params(index + numPoseParams + numIntrinsicParams) = kframe.getGeometry().getDepthParam(mapParamsIds[index]);
            }

            vecxf res = params - init_params;
            vecxf conv_dot_res = init_invcovariance * res;
            float weight = mesh_vo::mapping_prior_weight / numParams;
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

            if (mesh_vo::mapping_regu_weight > 0.0)
            {
                float weight = mesh_vo::mapping_regu_weight / numMapParams;
                DenseLinearProblem hg_regu = kframe.getGeometry().HGRegu(numPoseParams, weight);
                assert(hg_regu.getCount() > 0);
                problem += hg_regu;
            }

            if (mesh_vo::mapping_prior_weight > 0.0)
            {
                vecxf params = vecxf::Zero(numParams);

                params(0) = cam[lvl].fx;
                params(1) = cam[lvl].fy;
                params(2) = cam[lvl].cx;
                params(3) = cam[lvl].cy;

                for (size_t index = 0; index < frames.size(); index++)
                {
                    params.segment(index * 6 + numIntrinsicParams, 6) = frames[index].getLocalPose().log();
                }

                for (size_t i = 0; i < mapParamsIds.size(); i++)
                {
                    params(i + numPoseParams + numIntrinsicParams) = kframe.getGeometry().getDepthParam(mapParamsIds[i]);
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

                vecxf intrincisInc = inc.segment(0, numIntrinsicParams);
                vecxf poseInc = inc.segment(numIntrinsicParams, numPoseParams);
                vecxf mapInc = inc.segment(numPoseParams, numMapParams);

                camera best_intrinsics = cam[lvl];
                cam[lvl] = new camera(cam[lvl].fx - intrincisInc(0), cam[lvl].fy - intrincisInc(1), cam[lvl].cx - intrincisInc(2), cam[lvl].cy - intrincisInc(3), cam[lvl].width, cam[lvl].height);

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
                    // kframe.getGeometry().setWeightParam(problem.getH()(mapParamsIds[i], mapParamsIds[i]), mapParamsIds[i]);
                    kframe.getGeometry().setWeightParam(1.0 / mesh_vo::mapping_param_good_var, mapParamsIds[i]);
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

                if (mesh_vo::mapping_regu_weight > 0.0)
                {
                    Error e_regu = kframe.getGeometry().errorRegu();
                    assert(e_regu.getCount() > 0);
                    e_regu *= mesh_vo::mapping_regu_weight / e_regu.getCount();
                    e += e_regu;
                }

                if (mesh_vo::mapping_prior_weight > 0.0)
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
                    float weight = mesh_vo::mapping_prior_weight / numParams;
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

                    if (p > mesh_vo::mapping_convergence_p)
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
                    cam[lvl] = best_intrinsics;

                    for (size_t i = 0; i < frames.size(); i++)
                    {
                        frames[i].setLocalPose(best_poses[i]);
                    }

                    for (size_t i = 0; i < mapParamsIds.size(); i++)
                    {
                        kframe.getGeometry().setDepthParam(best_mapParams[i], mapParamsIds[i]);
                    }

                    // reject update, increase lambda, use un-updated data

                    //if (poseInc.dot(poseInc) < 1e-16 && mapInc.dot(mapInc) < 1e-16)
                    if(false)
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

DenseLinearProblem intrinsicPoseMapOptimizerCPU::computeProblem(frameCPU &frame, keyFrameCPU &kframe, int frameId, int numFrames, int lvl)
{
    error_buffer.setToNoData(lvl);
    jpose_buffer.setToNoData(lvl);
    jmap_buffer.setToNoData(lvl);
    pId_buffer.setToNoData(lvl);

    int numMapParams = kframe.getGeometry().getParamIds().size();

    renderer.renderJPoseMapParallel(kframe, frame, jpose_buffer, jmap_buffer, error_buffer, pId_buffer, cam, lvl);
    renderer.renderJIntrinsicParallel(kframe, frame, jintrinsic_buffer, error_buffer, pId_buffer, cam, lvl);
    DenseLinearProblem problem = reducer.reduceHGIntrinsicPoseMapParallel(frameId, numFrames, numMapParams, jpose_buffer.get(lvl), jmap_buffer.get(lvl), error_buffer.get(lvl), pId_buffer.get(lvl));

    return problem;
}
