#include "optimizers/mapOptimizer.h"

MapOptimizer::MapOptimizer(int w, int h, bool _printLog)
    : BaseOptimizer(w, h),
      jmap_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      pids_texture_(w, h, Vec3<float>(-1, -1, -1))
{
    printLog = _printLog;
}

void MapOptimizer::init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int lvl)
{
    int numParams = kframe.mesh().vertex_count();

    init_positions = kframe.mesh().get_positions();
    init_indices = kframe.mesh().get_indices();

    invCovariance = Matxf::Identity(numParams, numParams);
    init_params = Vecxf::Zero(numParams);

    for (size_t i = 0; i < numParams; i++)
    {
        init_params(i) = fromDepthToParam(init_positions[i * 3 + 2]);
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
        // init_error += (mesh_vo::mapping_regu_weight / numParams) * error_regu_(kframe.mesh());

        float regu_error = 0.0f;
        for (size_t i = 0; i < init_indices.size(); i += 3)
        {
            Vec3i id(init_indices[i + 0],
                     init_indices[i + 1],
                     init_indices[i + 2]);
            Vec3f depth(init_positions[id(0) * 3 + 2],
                        init_positions[id(1) * 3 + 2],
                        init_positions[id(2) * 3 + 2]);
            float r1 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(1));
            float r2 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(2));
            float r3 = fromDepthToParam(depth(1)) - fromDepthToParam(depth(2));
            regu_error += r1 * r1 + r2 * r2 + r3 * r3;
        }
        init_error += (mesh_vo::mapping_regu_weight / numParams) * regu_error;
    }

    /*
    if (mesh_vo::mapping_prior_weight > 0.0)
    {
        Vecxf res = params - init_params;
        Vecxf conv_dot_res = init_invcovariance * res;
        float weight = mesh_vo::mapping_prior_weight / numParams;
        float priorError = weight * (res.dot(conv_dot_res));

        init_error += priorError;
    }
    */

    positions = init_positions;
    indices = init_indices;
    params = init_params;
    error = init_error;

    if (printLog)
        std::cout << "mapOptimizer initial error " << init_error << " " << lvl << std::endl;

    reached_convergence_ = false;
}

void MapOptimizer::step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int lvl)
{
    int numParams = kframe.mesh().vertex_count();

    DenseLinearProblemx problem(numParams);
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        DenseLinearProblemx fhg = compute_problem_(frames[i], kframe, cam, lvl);
        if (fhg.count() > 0)
        {
            fhg.scale(1.0 / fhg.count());
            problem += fhg;
        }
    }
    problem.scale(1.0 / frames.size());

    if (mesh_vo::mapping_regu_weight > 0.0)
    {
        for (size_t i = 0; i < indices.size(); i += 3)
        {
            Vec3<int> ids(indices[i + 0],
                          indices[i + 1],
                          indices[i + 2]);
            Vec3<float> depths(positions[ids(0) * 3 + 2],
                               positions[ids(1) * 3 + 2],
                               positions[ids(2) * 3 + 2]);
            // regu_error += (depth(0) - depth(1)) * (depth(0) - depth(1)) + (depth(1) - depth(2)) * (depth(1) - depth(2));

            float r1 = fromDepthToParam(depths(0)) - fromDepthToParam(depths(1));
            Vec3<float> jac1(1.0, -1.0, 0.0);
            problem.add(jac1, r1, mesh_vo::mapping_regu_weight / numParams, ids);

            float r2 = fromDepthToParam(depths(0)) - fromDepthToParam(depths(2));
            Vec3<float> jac2(1.0, 0.0, -1.0);
            problem.add(jac2, r2, mesh_vo::mapping_regu_weight / numParams, ids);

            float r3 = fromDepthToParam(depths(1)) - fromDepthToParam(depths(2));
            Vec3<float> jac3(0.0, 1.0, -1.0);
            problem.add(jac3, r3, mesh_vo::mapping_regu_weight / numParams, ids);
        }
    }

    if (mesh_vo::mapping_prior_weight > 0.0)
    {
        Vecxf res = init_invcovariancesqrt * (params - init_params);
        Matxf jacobian = init_invcovariancesqrt;
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

        Vecxf inc = problem.solve(lambda);

        Vecxf new_params = params + inc;
        std::vector<float> new_positions;

        // auto pos_map = kframe.mesh().MapWritePositions();
        for (size_t i = 0; i < numParams; i++)
        {
            Vec3<float> pos(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]);

            Vec3<float> pos_up = (pos / pos(2)) * fromParamToDepth(new_params(i));
            // best_map_pos.push_back(pos);
            // pos_map[i * 3 + 0] = pos_up(0);
            // pos_map[i * 3 + 1] = pos_up(1);
            // pos_map[i * 3 + 2] = pos_up(2);
            new_positions.push_back(pos_up(0));
            new_positions.push_back(pos_up(1));
            new_positions.push_back(pos_up(2));
        }

        kframe.mesh().set_positions(new_positions);

        float new_error = 0;
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            Error fe = compute_error_(frames[i], kframe, cam, lvl);
            if (fe.getCount() < 0.5 * frames[i].image().width(lvl) * frames[i].image().height(lvl))
            {
                new_error += init_error * 2.0;
            }
            else
            {
                new_error += fe.getError() / fe.getCount();
            }
        }
        new_error *= 1.0 / frames.size();

        if (mesh_vo::mapping_regu_weight > 0.0)
        {
            // new_error += (mesh_vo::mapping_regu_weight / numParams) * error_regu_(kframe.mesh());

            float regu_error = 0.0f;
            for (size_t i = 0; i < indices.size(); i += 3)
            {
                Vec3<int> id(indices[i + 0],
                             indices[i + 1],
                             indices[i + 2]);
                Vec3<float> depth(new_positions[id(0) * 3 + 2],
                                  new_positions[id(1) * 3 + 2],
                                  new_positions[id(2) * 3 + 2]);
                float r1 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(1));
                float r2 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(2));
                float r3 = fromDepthToParam(depth(1)) - fromDepthToParam(depth(2));
                regu_error += r1 * r1 + r2 * r2 + r3 * r3;
            }
            new_error += (mesh_vo::mapping_regu_weight / numParams) * regu_error;
        }

        if (mesh_vo::mapping_prior_weight > 0.0)
        {
            Vecxf res = new_params - init_params;
            Vecxf conv_dot_res = init_invcovariance * res;
            float weight = mesh_vo::mapping_prior_weight / numParams;
            float priorError = weight * (res.dot(conv_dot_res));

            new_error += priorError;
        }

        if (printLog)
            std::cout << "mapOptimizer new error " << new_error << " " << lambda << " " << n_try << " lvl: " << lvl << " mesh_regu: " << mesh_vo::mapping_regu_weight << std::endl;

        if (new_error <= error)
        {
            float p = new_error / error;
            error = new_error;
            positions = new_positions;
            params = new_params;

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
            kframe.mesh().set_positions(positions);

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

DenseLinearProblemx MapOptimizer::compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl)
{
    int numMapParams = kframe.mesh().vertex_count();

    jmaprenderer_.Render(kframe.mesh(), frame.local_pose(), cam, lvl, lvl, kframe.frame().image(), frame.image(), frame.didxy(), jmap_texture_, pids_texture_, r_texture_);
    return hgmapreducer_.reduce(lvl, numMapParams, jmap_texture_, pids_texture_, r_texture_);
}
