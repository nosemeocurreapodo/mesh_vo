#include "optimizers/mapOptimizer.h"

MapOptimizer::MapOptimizer(int w, int h, bool _printLog)
    : BaseOptimizer(w, h),
      jmap_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jexp_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      pids_texture_(w, h, Vec3<PidType>(-1, -1, -1))
{
    printLog = _printLog;
}

void MapOptimizer::init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
    int num_depths = kframe.mesh().vertex_count();
    int numParams = num_depths;

    init_depths = get_depths(kframe.mesh());
    init_triangles = get_indices(kframe.mesh());

    invCovariance = Matxf::Identity(numParams, numParams);
    init_params = Vecxf::Zero(numParams);

    for (size_t i = 0; i < num_depths; i++)
    {
        init_params(i) = fromDepthToParam(init_depths[i]);
        invCovariance(i, i) = 1.0 / mesh_vo::mapping_param_initial_var;
    }

    init_invcovariance = invCovariance;

    if (mesh_vo::mapping_prior_weight > 0.0)
        init_invcovariancesqrt = invCovariance.sqrt();

    init_error = 0;
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        Error ef = compute_error_(frames[i], kframe, cam, in_lvl, out_lvl);
        init_error += ef.getError() / ef.getCount();
    }
    init_error *= 1.0 / frames.size();

    if (mesh_vo::mapping_regu_weight > 0.0)
    {
        float regu_error = 0.0f;
        for (size_t i = 0; i < init_triangles.size(); i++)
        {
            Vec3i id = init_triangles[i];
            Vec3f depth(init_depths[id(0)],
                        init_depths[id(1)],
                        init_depths[id(2)]);
            float r1 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(1));
            float r2 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(2));
            float r3 = fromDepthToParam(depth(1)) - fromDepthToParam(depth(2));
            regu_error += r1 * r1 + r2 * r2 + r3 * r3;
        }
        init_error += (mesh_vo::mapping_regu_weight / num_depths) * regu_error;
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

    depths = init_depths;
    triangles = init_triangles;
    params = init_params;
    error = init_error;

    if (printLog)
        std::cout << "mapOptimizer initial error " << init_error << " " << in_lvl << " " << out_lvl << std::endl;

    reached_convergence_ = false;
}

void MapOptimizer::step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
    int num_depths = kframe.mesh().vertex_count();
    int numParams = kframe.mesh().vertex_count();

    DenseLinearProblemx problem(numParams);
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        DenseLinearProblemx fhg = compute_problem_(frames[i], kframe, cam, i, frames.size(), num_depths, in_lvl, out_lvl);
        if (fhg.count() > 0)
        {
            fhg.scale(1.0 / fhg.count());
            problem += fhg;
        }
    }
    problem.scale(1.0 / frames.size());

    if (mesh_vo::mapping_regu_weight > 0.0)
    {
        for (size_t i = 0; i < triangles.size(); i++)
        {
            Vec3<int> ids = triangles[i];
            Vec3<float> depth(depths[ids(0)],
                              depths[ids(1)],
                              depths[ids(2)]);
            // regu_error += (depth(0) - depth(1)) * (depth(0) - depth(1)) + (depth(1) - depth(2)) * (depth(1) - depth(2));

            float r1 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(1));
            Vec3<float> jac1(1.0, -1.0, 0.0);
            problem.add(jac1, r1, mesh_vo::mapping_regu_weight / num_depths, ids);

            float r2 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(2));
            Vec3<float> jac2(1.0, 0.0, -1.0);
            problem.add(jac2, r2, mesh_vo::mapping_regu_weight / num_depths, ids);

            float r3 = fromDepthToParam(depth(1)) - fromDepthToParam(depth(2));
            Vec3<float> jac3(0.0, 1.0, -1.0);
            problem.add(jac3, r3, mesh_vo::mapping_regu_weight / num_depths, ids);
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
        std::vector<float> new_depths;

        for (size_t i = 0; i < num_depths; i++)
        {
            float new_depth = fromParamToDepth(new_params(i));
            if (new_depth < RenderConstants::NEAR_PLANE)
                new_depth = RenderConstants::NEAR_PLANE;
            if (new_depth > RenderConstants::FAR_PLANE)
                new_depth = RenderConstants::FAR_PLANE;

            new_depths.push_back(new_depth);
        }

        set_depths(kframe.mesh(), new_depths);

        float new_error = 0;
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            Error fe = compute_error_(frames[i], kframe, cam, in_lvl, out_lvl);
            if (fe.getCount() < 0.5 * frames[i].image().width(out_lvl) * frames[i].image().height(out_lvl))
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
            float regu_error = 0.0f;
            for (size_t i = 0; i < triangles.size(); i++)
            {
                Vec3<int> id = triangles[i];
                Vec3<float> depth(new_depths[id(0)],
                                  new_depths[id(1)],
                                  new_depths[id(2)]);
                float r1 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(1));
                float r2 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(2));
                float r3 = fromDepthToParam(depth(1)) - fromDepthToParam(depth(2));
                regu_error += r1 * r1 + r2 * r2 + r3 * r3;
            }
            new_error += (mesh_vo::mapping_regu_weight / num_depths) * regu_error;
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
            std::cout << "mapOptimizer new error " << new_error << " " << lambda << " " << n_try << " lvl: " << in_lvl << " " << out_lvl << " mesh_regu: " << mesh_vo::mapping_regu_weight << std::endl;

        if (new_error <= error)
        {
            float p = new_error / error;
            error = new_error;
            depths = new_depths;
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
            set_depths(kframe.mesh(), depths);

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

DenseLinearProblemx MapOptimizer::compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int frame_id, int num_frames, int num_vertices, int in_lvl, int out_lvl)
{
    jmaprenderer_.Render(kframe.mesh(), frame.local_pose(), frame.local_exposure(), cam, in_lvl, out_lvl, kframe.frame().image(), frame.image(), kframe.frame().didxy(), jmap_texture_, jexp_texture_, pids_texture_, r_texture_);
    // jmaprenderer_.Render(kframe.mesh(), frame.local_pose(), cam, in_lvl, out_lvl, kframe.frame().image(), frame.image(), kframe.frame().didxy(), jmap_texture_, pids_texture_, r_texture_);
    return hgmapreducer_.reduce(out_lvl, frame_id, num_frames, num_vertices, jmap_texture_, pids_texture_, r_texture_, kframe.mesh());
}
