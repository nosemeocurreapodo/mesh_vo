#include "optimizers/poseExpMapOptimizer.h"

PoseExpMapOptimizer::PoseExpMapOptimizer(int w, int h, bool _printLog)
    : BaseOptimizer(w, h),
      jtra_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jrot_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jexp_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jmap_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      pids_texture_(w, h, Vec3<PidType>(-1, -1, -1)),
      solver_(0)
{
    printLog_ = _printLog;
}

void PoseExpMapOptimizer::init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
    int num_depths = kframe.mesh().vertex_count();
    int numParams = num_depths + 8 * frames.size();

    init_depths_ = get_depths(kframe.mesh());
    init_triangles_ = get_indices(kframe.mesh());

    init_poses_.clear();
    init_exposures_.clear();
    for (int i = 0; i < frames.size(); i++)
    {
        init_poses_.push_back(frames[i].local_pose());
        init_exposures_.push_back(frames[i].local_exposure());
    }

    invCovariance_ = Matxf::Identity(numParams, numParams);

    for (size_t i = 0; i < num_depths; i++)
    {
        invCovariance_(i, i) = 1.0 / mesh_vo::mapping_param_initial_var;
    }

    init_invcovariance_ = invCovariance_;

    if (mesh_vo::mapping_prior_weight > 0.0)
        init_invcovariancesqrt_ = invCovariance_.sqrt();

    init_error_ = 0;
    Error err;
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        compute_error_(frames[i], kframe, cam, in_lvl, out_lvl, err);
        // init_error += ef.getError() / ef.getCount();
    }
    // init_error *= 1.0 / frames.size();
    init_error_ = err.getError() / err.getCount();

    if (mesh_vo::mapping_regu_weight > 0.0)
    {
        float regu_error = 0.0f;
        for (size_t i = 0; i < init_triangles_.size(); i++)
        {
            Vec3i id = init_triangles_[i];
            Vec3f depth(init_depths_[id(0)],
                        init_depths_[id(1)],
                        init_depths_[id(2)]);
            float r1 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(1));
            float r2 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(2));
            float r3 = fromDepthToParam(depth(1)) - fromDepthToParam(depth(2));
            regu_error += r1 * r1 + r2 * r2 + r3 * r3;
        }
        init_error_ += (mesh_vo::mapping_regu_weight / num_depths) * regu_error;
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

    depths_ = init_depths_;
    triangles_ = init_triangles_;
    poses_ = init_poses_;
    exposures_ = init_exposures_;
    error_ = init_error_;
    if (printLog_)
        std::cout << "poseMapOptimizer initial error " << init_error_ << " " << in_lvl << " " << out_lvl << std::endl;

    reached_convergence_ = false;
}

void PoseExpMapOptimizer::step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
    int num_depths = kframe.mesh().vertex_count();
    int numParams = kframe.mesh().vertex_count() + 8 * frames.size();

    DenseLinearProblemx problem(numParams);
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        compute_problem_(frames[i], kframe, cam, i, frames.size(), num_depths, in_lvl, out_lvl, problem);
        /*
        if (fhg.count() > 0)
         {
             fhg.scale(1.0 / fhg.count());
             problem += fhg;
         }
             */
    }
    // problem.scale(1.0 / frames.size());

    if (mesh_vo::mapping_regu_weight > 0.0)
    {
        for (size_t i = 0; i < triangles_.size(); i++)
        {
            Vec3<int> ids = triangles_[i];
            Vec3<float> depth(depths_[ids(0)],
                              depths_[ids(1)],
                              depths_[ids(2)]);
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

        solver_.compute(problem.Hp() + Matxf::Identity(numParams, numParams) * lambda);
        Vecxf inc = solver_.solve(-problem.G());

        std::vector<float> new_depths;
        std::vector<SE3f> new_poses;
        std::vector<Vec2f> new_exposures;

        for (size_t i = 0; i < num_depths; i++)
        {
            float new_param = fromDepthToParam(depths_[i]) + inc(i);
            float new_depth = fromParamToDepth(new_param);
            if (new_depth < RenderConstants::NEAR_PLANE)
                new_depth = RenderConstants::NEAR_PLANE;
            if (new_depth > RenderConstants::FAR_PLANE)
                new_depth = RenderConstants::FAR_PLANE;
            new_depths.push_back(new_depth);
        }

        for (int i = 0; i < frames.size(); i++)
        {
            Vec6f pose_inc(inc(num_depths + i * 8 + 0),
                           inc(num_depths + i * 8 + 1),
                           inc(num_depths + i * 8 + 2),
                           inc(num_depths + i * 8 + 3),
                           inc(num_depths + i * 8 + 4),
                           inc(num_depths + i * 8 + 5));
            SE3f new_pose = poses_[i] * SE3f::exp(pose_inc); // SE3::exp(inc).inverse();
            new_poses.push_back(new_pose);

            Vec2f exp_inc(inc(num_depths + i * 8 + 6),
                          inc(num_depths + i * 8 + 7));
            Vec2f new_exp = exposures_[i] + exp_inc;
            new_exposures.push_back(new_exp);
        }

        set_depths(kframe.mesh(), new_depths);

        for (int i = 0; i < frames.size(); i++)
        {
            frames[i].local_pose() = new_poses[i];
            frames[i].local_exposure() = new_exposures[i];
        }

        float new_error = 0;
        Error err;
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            compute_error_(frames[i], kframe, cam, in_lvl, out_lvl, err);
            /*
            if (fe.getCount() < 0.5 * frames[i].image().width(out_lvl) * frames[i].image().height(out_lvl))
            {
                new_error += init_error * 2.0;
            }
            else
            {
                new_error += fe.getError() / fe.getCount();
            }
                */
        }
        // new_error *= 1.0 / frames.size();
        new_error = err.getError() / err.getCount();

        if (mesh_vo::mapping_regu_weight > 0.0)
        {
            float regu_error = 0.0f;
            for (size_t i = 0; i < triangles_.size(); i++)
            {
                Vec3<int> id = triangles_[i];
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

        if (printLog_)
            std::cout << "poseMapOptimizer new error " << new_error << " " << lambda << " " << n_try << " lvl: " << in_lvl << " " << out_lvl << " mesh_regu: " << mesh_vo::mapping_regu_weight << std::endl;

        if (new_error <= error_)
        {
            float p = new_error / error_;
            error_ = new_error;
            depths_ = new_depths;
            poses_ = new_poses;
            exposures_ = new_exposures;

            if (p >= mesh_vo::mapping_convergence_p)
            {
                reached_convergence_ = true;
                if (printLog_)
                    std::cout << "poseMapOptimizer converged p:" << p << std::endl;
            }
            break;
        }
        else
        {
            set_depths(kframe.mesh(), depths_);

            for (int i = 0; i < frames.size(); i++)
            {
                frames[i].local_pose() = poses_[i];
                frames[i].local_exposure() = exposures_[i];
            }

            float incMag = inc.dot(inc) / numParams;

            if (incMag <= mesh_vo::mapping_convergence_m_v)
            {
                reached_convergence_ = true;
                if (printLog_)
                    std::cout << "poseMapOptimizer too small " << incMag << std::endl;
                break;
            }
        }
    }
}

void PoseExpMapOptimizer::compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int frame_id, int num_frames, int num_vertices, int in_lvl, int out_lvl, DenseLinearProblemx &total)
{
    jposeexpmaprenderer_.Render(kframe.mesh(), frame.local_pose(), frame.local_exposure(), cam, in_lvl, out_lvl, kframe.image(), frame.image(), kframe.didxy(), jtra_texture_, jrot_texture_, jexp_texture_, jmap_texture_, pids_texture_, r_texture_);
    hgposeexpmapreducer_.reduce(out_lvl, frame_id, num_frames, num_vertices, jtra_texture_, jrot_texture_, jexp_texture_, jmap_texture_, pids_texture_, r_texture_, kframe.mesh(), total);
}
