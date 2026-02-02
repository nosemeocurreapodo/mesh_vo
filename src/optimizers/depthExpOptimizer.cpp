#include "optimizers/depthExpOptimizer.h"

DepthExpOptimizer::DepthExpOptimizer(int w, int h, bool _printLog)
    : BaseOptimizer(w, h),
      jdepth_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jexp_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      pids_texture_(w, h, Vec3<PidType>(-1, -1, -1)),
      solver_(0),
      problem_(0)
{
    printLog_ = _printLog;
}

void DepthExpOptimizer::init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
    int num_depths = kframe.mesh().vertex_count();
    int numParams = num_depths + 2 * frames.size();

    init_depths_ = get_depths(kframe.mesh());

    triangles_ = get_indices(kframe.mesh());
    edges_ = get_edges(kframe.mesh());

    init_exposures_.clear();
    for (int i = 0; i < frames.size(); i++)
    {
        init_exposures_.push_back(frames[i].local_exposure());
    }

    invCovariance_ = Matxf::Identity(numParams, numParams);
    init_params_ = Vecxf::Zero(numParams);

    for (size_t i = 0; i < num_depths; i++)
    {
        init_params_(i) = fromDepthToParam(init_depths_[i]);
        invCovariance_(i, i) = 1.0 / mesh_vo::mapping_param_initial_var;
    }

    for (size_t i = 0; i < frames.size(); i++)
    {
        init_params_(num_depths + i * 2) = 0.0;
        init_params_(num_depths + i * 2 + 1) = 0.0;
        invCovariance_(num_depths + i * 2, num_depths + i * 2) = 1.0 / mesh_vo::mapping_param_initial_var;
        invCovariance_(num_depths + i * 2 + 1, num_depths + i * 2 + 1) = 1.0 / mesh_vo::mapping_param_initial_var;
    }

    init_invcovariance_ = invCovariance_;

    if (mesh_vo::mapping_prior_weight > 0.0)
        init_invcovariancesqrt_ = invCovariance_.sqrt();

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
    exposures_ = init_exposures_;
    params_ = init_params_;
    error_ = init_error_;

    if (printLog_)
        std::cout << "mapOptimizer initial error " << init_error_ << " " << in_lvl << " " << out_lvl << std::endl;

    reached_convergence_ = false;
}

void DepthExpOptimizer::step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
    int num_depths = kframe.mesh().vertex_count();
    int numParams = kframe.mesh().vertex_count() + 2 * frames.size();

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
            Vec2f jac(1.0f, -1.0f);
            problem_.add(jac, res, w * mesh_vo::mapping_regu_weight / edges_.size(), edge);
        }
    }

    if (mesh_vo::mapping_prior_weight > 0.0)
    {
        Vecxf res = init_invcovariancesqrt_ * (params_ - init_params_);
        Matxf jacobian = init_invcovariancesqrt_;
        float weight = mesh_vo::mapping_prior_weight / numParams;
        problem_.add(jacobian, res, weight);
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

        Vecxf new_params = params_ + inc;
        std::vector<float> new_depths;
        std::vector<Vec2f> new_exposures;

        for (size_t i = 0; i < num_depths; i++)
        {
            float new_depth = fromParamToDepth(new_params(i));
            if (new_depth < RenderConstants::NEAR_PLANE)
                new_depth = RenderConstants::NEAR_PLANE;
            if (new_depth > RenderConstants::FAR_PLANE)
                new_depth = RenderConstants::FAR_PLANE;

            new_depths.push_back(new_depth);
        }

        for (size_t i = 0; i < frames.size(); i++)
        {
            new_exposures.push_back(Vec2f(new_params(num_depths + i * 2 + 0), new_params(num_depths + i * 2 + 1)));
        }

        set_depths(kframe.mesh(), new_depths);

        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i].local_exposure() = new_exposures[i];
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

        if (mesh_vo::mapping_prior_weight > 0.0)
        {
            Vecxf res = new_params - init_params_;
            Vecxf conv_dot_res = init_invcovariance_ * res;
            float weight = mesh_vo::mapping_prior_weight / numParams;
            float priorError = weight * (res.dot(conv_dot_res));

            new_error += priorError;
        }

        if (printLog_)
            std::cout << "mapOptimizer new error " << new_error << " " << lambda << " " << n_try << " lvl: " << in_lvl << " " << out_lvl << " mesh_regu: " << mesh_vo::mapping_regu_weight << std::endl;

        if (new_error <= error_)
        {
            float p = new_error / error_;
            error_ = new_error;
            depths_ = new_depths;
            exposures_ = new_exposures;
            params_ = new_params;

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

            for (size_t i = 0; i < frames.size(); i++)
            {
                frames[i].local_exposure() = exposures_[i];
            }

            float incMag = inc.dot(inc) / numParams;

            if (incMag <= mesh_vo::mapping_convergence_m_v)
            {
                reached_convergence_ = true;
                if (printLog_)
                    std::cout << "mapOptimizer too small " << incMag << std::endl;
                break;
            }
        }
    }
}

void DepthExpOptimizer::compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int frame_id, int num_frames, int num_vertices, int in_lvl, int out_lvl, DenseLinearProblemx &total)
{
    jdepthrenderer_.Render(kframe.mesh(), frame.local_pose(), frame.local_exposure(), cam, in_lvl, out_lvl, kframe.image(), frame.didxy(), image_texture_, jdepth_texture_, jexp_texture_, pids_texture_);
    hgmapreducer_.reduce(out_lvl, frame_id, num_frames, num_vertices, jdepth_texture_, jexp_texture_, pids_texture_, image_texture_, frame.image(), kframe.mesh(), total);
}
