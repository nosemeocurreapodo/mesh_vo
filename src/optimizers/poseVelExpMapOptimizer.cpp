#include "optimizers/poseVelExpMapOptimizer.h"

PoseVelExpMapOptimizer::PoseVelExpMapOptimizer(int w, int h, bool _printLog)
    : BaseOptimizer(w, h),
      jtra_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jrot_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jtravel_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jrotvel_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jexp_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jmap_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      pids_texture_(w, h, Vec3<PidType>(-1, -1, -1))
{
    printLog = _printLog;
}

void PoseVelExpMapOptimizer::init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
    int num_depths = kframe.mesh().vertex_count();
    int numParams = num_depths + 14 * frames.size();

    init_positions = kframe.mesh().get_positions();
    init_indices = kframe.mesh().get_indices();

    init_poses.clear();
    init_vels.clear();
    init_exposures.clear();
    for (int i = 0; i < frames.size(); i++)
    {
        init_poses.push_back(frames[i].local_pose());
        init_vels.push_back(frames[i].local_vel());
        init_exposures.push_back(frames[i].local_exposure());
    }

    invCovariance = Matxf::Identity(numParams, numParams);

    for (size_t i = 0; i < num_depths; i++)
    {
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

    positions = init_positions;
    indices = init_indices;
    poses = init_poses;
    vels = init_vels;
    exposures = init_exposures;
    error = init_error;

    if (printLog)
        std::cout << "poseMapOptimizer initial error " << init_error << " " << in_lvl << " " << out_lvl << std::endl;

    reached_convergence_ = false;
}

void PoseVelExpMapOptimizer::step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
    int num_depths = kframe.mesh().vertex_count();
    int numParams = kframe.mesh().vertex_count() + 14 * frames.size();

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
            problem.add(jac1, r1, mesh_vo::mapping_regu_weight / num_depths, ids);

            float r2 = fromDepthToParam(depths(0)) - fromDepthToParam(depths(2));
            Vec3<float> jac2(1.0, 0.0, -1.0);
            problem.add(jac2, r2, mesh_vo::mapping_regu_weight / num_depths, ids);

            float r3 = fromDepthToParam(depths(1)) - fromDepthToParam(depths(2));
            Vec3<float> jac3(0.0, 1.0, -1.0);
            problem.add(jac3, r3, mesh_vo::mapping_regu_weight / num_depths, ids);
        }
    }

    /*
    if (mesh_vo::mapping_prior_weight > 0.0)
    {
        Vecxf res = init_invcovariancesqrt * (params - init_params);
        Matxf jacobian = init_invcovariancesqrt;
        float weight = mesh_vo::mapping_prior_weight / numParams;
        problem.add(jacobian, res, weight);
    }
    */

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

        std::vector<float> new_positions;
        std::vector<SE3f> new_poses;
        std::vector<Vec6f> new_vels;
        std::vector<Vec2f> new_exposures;

        for (size_t i = 0; i < num_depths; i++)
        {
            Vec3<float> pos(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]);

            float param_inc = inc(i);
            float new_param = fromDepthToParam(pos(2)) + param_inc;
            Vec3<float> pos_up = (pos / pos(2)) * fromParamToDepth(new_param);

            new_positions.push_back(pos_up(0));
            new_positions.push_back(pos_up(1));
            new_positions.push_back(pos_up(2));
        }

        for (int i = 0; i < frames.size(); i++)
        {
            Vec6f pose_inc(inc(num_depths + i * 14 + 0),
                           inc(num_depths + i * 14 + 1),
                           inc(num_depths + i * 14 + 2),
                           inc(num_depths + i * 14 + 3),
                           inc(num_depths + i * 14 + 4),
                           inc(num_depths + i * 14 + 5));
            SE3f new_pose = poses[i] * SE3f::exp(pose_inc); // SE3::exp(inc).inverse();
            new_poses.push_back(new_pose);

            Vec6f vel_inc(inc(num_depths + i * 14 + 6),
                          inc(num_depths + i * 14 + 7),
                          inc(num_depths + i * 14 + 8),
                          inc(num_depths + i * 14 + 9),
                          inc(num_depths + i * 14 + 10),
                          inc(num_depths + i * 14 + 11));

            Vec6f new_vel = vels[i] + vel_inc;
            new_vels.push_back(new_vel);

            Vec2f exp_inc(inc(num_depths + i * 14 + 12),
                          inc(num_depths + i * 14 + 13));

            Vec2f new_exp = exposures[i] + exp_inc;
            new_exposures.push_back(new_exp);
        }

        kframe.mesh().set_positions(new_positions);

        for (int i = 0; i < frames.size(); i++)
        {
            frames[i].local_pose() = new_poses[i];
            frames[i].local_exposure() = new_exposures[i];
            frames[i].local_vel() = new_vels[i];
        }

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
            new_error += (mesh_vo::mapping_regu_weight / num_depths) * regu_error;
        }

        /*
        if (mesh_vo::mapping_prior_weight > 0.0)
        {
            Vecxf res = new_params - init_params;
            Vecxf conv_dot_res = init_invcovariance * res;
            float weight = mesh_vo::mapping_prior_weight / numParams;
            float priorError = weight * (res.dot(conv_dot_res));

            new_error += priorError;
        }
        */

        if (printLog)
            std::cout << "poseMapOptimizer new error " << new_error << " " << lambda << " " << n_try << " lvl: " << in_lvl << " " << out_lvl << " mesh_regu: " << mesh_vo::mapping_regu_weight << std::endl;

        if (new_error <= error)
        {
            float p = new_error / error;
            error = new_error;
            positions = new_positions;
            poses = new_poses;
            vels = new_vels;
            exposures = new_exposures;

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

            for (int i = 0; i < frames.size(); i++)
            {
                frames[i].local_pose() = poses[i];
                frames[i].local_exposure() = exposures[i];
                frames[i].local_vel() = vels[i];
            }

            float incMag = inc.dot(inc) / numParams;

            if (incMag <= mesh_vo::mapping_convergence_m_v)
            {
                reached_convergence_ = true;
                if (printLog)
                    std::cout << "poseMapOptimizer too small " << incMag << std::endl;
                break;
            }
        }
    }
}

DenseLinearProblemx PoseVelExpMapOptimizer::compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int frame_id, int num_frames, int num_vertices, int in_lvl, int out_lvl)
{
    jposeexpmaprenderer_.Render(kframe.mesh(), frame.local_pose(), frame.local_vel(), frame.local_exposure(), cam, 30.0, in_lvl, out_lvl, kframe.frame().image(), frame.image(), kframe.frame().didxy(), jtra_texture_, jrot_texture_, jtravel_texture_, jrotvel_texture_, jexp_texture_, jmap_texture_, pids_texture_, r_texture_);
    return hgposeexpmapreducer_.reduce(out_lvl, frame_id, num_frames, num_vertices, jtra_texture_, jrot_texture_, jtravel_texture_, jrotvel_texture_, jexp_texture_, jmap_texture_, pids_texture_, r_texture_, kframe.mesh());
}
