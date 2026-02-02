#include "optimizers/poseDepthOptimizer.h"

PoseDepthOptimizer::PoseDepthOptimizer(int w, int h, bool _printLog)
    : BaseOptimizer(w, h),
      jtra_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jrot_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jexp_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jdepth_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      pids_texture_(w, h, Vec3<PidType>(-1, -1, -1)),
      solver_(0),
      problem_(0)
{
    printLog_ = _printLog;
}

void PoseDepthOptimizer::init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
    int num_depths = kframe.mesh().vertex_count();
    int numParams = num_depths + 6 * frames.size();

    init_depths_ = get_depths(kframe.mesh());
    
    triangles_ = get_indices(kframe.mesh());
    edges_ = get_edges(kframe.mesh());

    init_poses_.clear();
    for (int i = 0; i < frames.size(); i++)
    {
        init_poses_.push_back(frames[i].local_pose());
    }

    invCovariance = Matxf::Identity(numParams, numParams);

    for (size_t i = 0; i < num_depths; i++)
    {
        invCovariance(i, i) = 1.0 / mesh_vo::mapping_param_initial_var;
    }

    init_invcovariance_ = invCovariance;

    if (mesh_vo::mapping_prior_weight > 0.0)
        init_invcovariancesqrt_ = invCovariance.sqrt();

    init_error_ = 0;
    Error err;
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        compute_error_(frames[i], kframe, cam, in_lvl, out_lvl, err);
    }
    init_error_ = err.getError() / err.getCount();

    if (mesh_vo::mapping_regu_weight > 0.0)
    {
        float regu_error = 0.0f;
        for (size_t i = 0; i < edges_.size(); i++)
        {
            Vec2<int> edge = edges_[i];

            float depth0 = init_depths_[edge(0)];
            float depth1 = init_depths_[edge(1)];

            float res = fromDepthToParam(depth0) - fromDepthToParam(depth1);
            float w = huber_weight(res, mesh_vo::huber_thresh_param);
            regu_error += w * res * res;
        }
        init_error_ += (mesh_vo::mapping_regu_weight / edges_.size()) * regu_error;
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
    poses_ = init_poses_;
    error_ = init_error_;

    solver_ = Solverx<float>(numParams);

    if (printLog_)
        std::cout << "poseMapOptimizer initial error " << init_error_ << " " << in_lvl << " " << out_lvl << std::endl;

    reached_convergence_ = false;
}

void PoseDepthOptimizer::step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
    int num_depths = kframe.mesh().vertex_count();
    int numParams = kframe.mesh().vertex_count() + 6 * frames.size();

    problem_.clear(numParams);
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        compute_problem_(frames[i], kframe, cam, i, frames.size(), num_depths, in_lvl, out_lvl, problem_);
    }
    problem_.scale(1.0 / problem_.count());

    if (mesh_vo::mapping_regu_weight > 0.0)
    {
        float regu_error = 0.0f;
        for (size_t i = 0; i < edges_.size(); i++)
        {
            Vec2<int> edge = edges_[i];

            float depth0 = depths_[edge(0)];
            float depth1 = depths_[edge(1)];

            float res = fromDepthToParam(depth0) - fromDepthToParam(depth1);
            float w = huber_weight(res, mesh_vo::huber_thresh_param);
            Vec2<float> jac(1.0, -1.0);
            problem_.add(jac, res, w * mesh_vo::mapping_regu_weight / edges_.size(), edge);
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

        Matxf Hp_lm = problem_.Hp();
        if (lambda > 0.0)
        {
            for (int i = 0; i < numParams; i++)
            {
                Hp_lm(i, i) += lambda;
                // Hp_lm(i, i) *= 1.0 + lambda;
            }
        }
        solver_.compute(Hp_lm);
        Vecxf inc = solver_.solve(-problem_.G());

        std::vector<float> new_depths;
        std::vector<SE3f> new_poses;

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
            Vec6f pose_inc(inc(num_depths + i * 6 + 0),
                           inc(num_depths + i * 6 + 1),
                           inc(num_depths + i * 6 + 2),
                           inc(num_depths + i * 6 + 3),
                           inc(num_depths + i * 6 + 4),
                           inc(num_depths + i * 6 + 5));
            SE3f new_pose = poses_[i] * SE3f::exp(pose_inc); // SE3::exp(inc).inverse();
            new_poses.push_back(new_pose);
        }

        set_depths(kframe.mesh(), new_depths);

        for (int i = 0; i < frames.size(); i++)
        {
            frames[i].local_pose() = new_poses[i];
        }

        float new_error = 0;
        Error err;
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            compute_error_(frames[i], kframe, cam, in_lvl, out_lvl, err);
        }
        new_error = err.getError() / err.getCount();

        if (mesh_vo::mapping_regu_weight > 0.0)
        {
            float regu_error = 0.0f;
            for (size_t i = 0; i < edges_.size(); i++)
            {
                Vec2<int> edge = edges_[i];

                float depth0 = new_depths[edge(0)];
                float depth1 = new_depths[edge(1)];

                float res = fromDepthToParam(depth0) - fromDepthToParam(depth1);
                float w = huber_weight(res, mesh_vo::huber_thresh_param);
                regu_error += w * res * res;
            }
            new_error += (mesh_vo::mapping_regu_weight / edges_.size()) * regu_error;
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

void PoseDepthOptimizer::compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int frame_id, int num_frames, int num_vertices, int in_lvl, int out_lvl, DenseLinearProblemx &total)
{
    jposemaprenderer_.Render(kframe.mesh(), frame.local_pose(), frame.local_exposure(), cam, in_lvl, out_lvl, kframe.image(), kframe.didxy(), image_texture_, jtra_texture_, jrot_texture_, jexp_texture_, jdepth_texture_, pids_texture_);
    hgposemapreducer_.reduce(out_lvl, frame_id, num_frames, num_vertices, jtra_texture_, jrot_texture_, jdepth_texture_, pids_texture_, image_texture_, frame.image(), kframe.mesh(), total);
}
