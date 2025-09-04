#include "optimizers/mapOptimizer.h"

MapOptimizer::MapOptimizer(int w, int h, bool _printLog)
    : BaseOptimizer(w, h),
      jmap_texture_(w, h, Vec3(0.0, 0.0, 0.0)),
      pids_texture_(w, h, Vec3(-1, -1, -1))
{
    printLog = _printLog;
}

void MapOptimizer::init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int lvl)
{
    int numParams = kframe.mesh().vertex_count();

    invCovariance = Matx::Identity(numParams, numParams);
    init_params = Vecx::Zero(numParams);

    auto pos_map = kframe.mesh().MapReadPositions();
    for (size_t i = 0; i < numParams; i++)
    {
        init_params(i) = pos_map[i * 3 + 2];
        invCovariance(i, i) = 1.0 / mesh_vo::mapping_param_initial_var;
    }

    init_invcovariance = invCovariance;

    if (mesh_vo::mapping_prior_weight > 0.0)
        init_invcovariancesqrt = invCovariance.sqrt();

    init_error = 0;
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        Error ef = compute_error_(frames[i], kframe, cam, lvl);
        init_error += ef.getError() / ef.getCount();
    }
    init_error *= 1.0 / frames.size();

    if (mesh_vo::mapping_regu_weight > 0.0)
    {
        init_error += mesh_vo::mapping_regu_weight * error_regu_(kframe.mesh());
    }

    if (mesh_vo::mapping_prior_weight > 0.0)
    {
        Vecx params(numParams);
        auto pos_map = kframe.mesh().MapReadPositions();
        for (size_t i = 0; i < numParams; i++)
        {
            params(i) = pos_map[i * 3 + 2];
        }

        Vecx res = params - init_params;
        Vecx conv_dot_res = init_invcovariance * res;
        float weight = mesh_vo::mapping_prior_weight / numParams;
        float priorError = weight * (res.dot(conv_dot_res));

        init_error += priorError;
    }

    if (printLog)
        std::cout << "mapOptimizer initial error " << init_error << " " << lvl << std::endl;

    reached_convergence_ = false;
}

void MapOptimizer::step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int lvl)
{
    int numParams = kframe.mesh().vertex_count();

    DenseLinearProblem problem(numParams);
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        DenseLinearProblem fhg = compute_problem_(frames[i], kframe, cam, lvl);
        if (fhg.count() > 0)
        {
            fhg.scale(1.0 / fhg.count());
            problem += fhg;
        }
    }

    if (mesh_vo::mapping_regu_weight > 0.0)
    {
        auto pos_map = kframe.mesh().MapReadPositions();
        auto ids_map = kframe.mesh().MapReadIndices();
        for (size_t i = 0; i < ids_map.size(); i += 3)
        {
            Vec3i ids(ids_map[i + 0],
                      ids_map[i + 1],
                      ids_map[i + 2]);
            Vec3 depths(pos_map[ids(0) * 3 + 2],
                        pos_map[ids(1) * 3 + 2],
                        pos_map[ids(2) * 3 + 2]);
            // regu_error += (depth(0) - depth(1)) * (depth(0) - depth(1)) + (depth(1) - depth(2)) * (depth(1) - depth(2));

            float r1 = depths(0) - depths(1);
            Vec3 jac1(1.0, -1.0, 0.0);
            problem.add(jac1, r1, mesh_vo::mapping_regu_weight / numParams, ids);

            float r2 = depths(0) - depths(2);
            Vec3 jac2(1.0, 0.0, -1.0);
            problem.add(jac2, r2, mesh_vo::mapping_regu_weight / numParams, ids);

            float r3 = depths(1) - depths(2);
            Vec3 jac3(0.0, 1.0, -1.0);
            problem.add(jac3, r3, mesh_vo::mapping_regu_weight / numParams, ids);
        }
    }

    if (mesh_vo::mapping_prior_weight > 0.0)
    {
        Vecx params = Vecx::Zero(numParams);
        auto pos_map = kframe.mesh().MapReadPositions();
        for (size_t i = 0; i < numParams; i++)
        {
            params(i) = pos_map[i * 3 + 2];
        }

        Vecx res = init_invcovariancesqrt * (params - init_params);
        Matx jacobian = init_invcovariancesqrt;
        float weight = mesh_vo::mapping_prior_weight / numParams;
        problem.add(jacobian, res, weight);
    }

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

        Vecx inc = problem.solve(lambda);

        std::vector<Vec3> best_map_pos;
        auto pos_map = kframe.mesh().MapWritePositions();
        for (size_t i = 0; i < numParams; i++)
        {
            Vec3 pos(pos_map[i * 3 + 0], pos_map[i * 3 + 1], pos_map[i * 3 + 2]);
            Vec3 pos_up = (pos / pos(2)) * (pos(2) + inc(i));
            best_map_pos.push_back(pos);
            pos_map[i * 3 + 0] = pos_up(0);
            pos_map[i * 3 + 1] = pos_up(1);
            pos_map[i * 3 + 2] = pos_up(2);
        }

        float error = 0;
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            Error fe = compute_error_(frames[i], kframe, cam, lvl);
            if (fe.getCount() < 0.5 * frames[i].image().width(lvl) * frames[i].image().height(lvl))
            {
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
            error += mesh_vo::mapping_regu_weight * error_regu_(kframe.mesh());
        }

        if (mesh_vo::mapping_prior_weight > 0.0)
        {
            Vecx params(numParams);
            auto pos_map_read = kframe.mesh().MapReadPositions();
            for (size_t i = 0; i < numParams; i++)
            {
                params(i) = pos_map_read[i * 3 + 2];
            }

            Vecx res = params - init_params;
            Vecx conv_dot_res = init_invcovariance * res;
            float weight = mesh_vo::mapping_prior_weight / numParams;
            float priorError = weight * (res.dot(conv_dot_res));

            error += priorError;
        }

        if (printLog)
            std::cout << "mapOptimizer new error " << error << " " << lambda << " " << n_try << " lvl: " << lvl << " mesh_regu: " << mesh_vo::mapping_regu_weight << std::endl;

        if (error <= init_error)
        {
            float p = error / init_error;
            init_error = error;

            if (p >= mesh_vo::mapping_convergence_p)
            {
                reached_convergence_ = true;
                if (printLog)
                    std::cout << "poseMapOptimizer converged p:" << p << std::endl;
            }
            break;
        }
        else
        {
            auto pos_map_write = kframe.mesh().MapWritePositions();
            for (size_t i = 0; i < numParams; i++)
            {
                pos_map_write[i * 3 + 0] = best_map_pos[i](0);
                pos_map_write[i * 3 + 1] = best_map_pos[i](1);
                pos_map_write[i * 3 + 2] = best_map_pos[i](2);
            }

            float incMag = inc.dot(inc) / numParams;

            if (incMag <= mesh_vo::mapping_convergence_m_v)
            {
                reached_convergence_ = true;
                if (printLog)
                    std::cout << "mapOptimizer too small " << incMag << std::endl;
                break;
            }
        }
    }
}

DenseLinearProblem MapOptimizer::compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl)
{
    int numMapParams = kframe.mesh().vertex_count();

    jmaprenderer_.Render(kframe.mesh(), frame.local_pose(), cam, lvl, lvl, kframe.frame().image(), frame.image(), frame.didxy(), jmap_texture_, pids_texture_, r_texture_);
    return hgmapreducer_.reduce(lvl, numMapParams, jmap_texture_, pids_texture_, r_texture_);
}
