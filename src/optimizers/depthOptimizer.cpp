#include "optimizers/depthOptimizer.h"

DepthOptimizer::DepthOptimizer(int w, int h, bool printLog)
    : BaseOptimizer(w, h),
      jdepth_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jexp_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      pids_texture_(w, h, Vec3<PidType>(-1, -1, -1)),
      printLog_(printLog),
      problem_(0),
      solver_(0)
{
}

void DepthOptimizer::init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
    int num_depths = kframe.mesh().vertex_count();
    int numParams = num_depths;

    init_depths_ = get_depths(kframe.mesh());
    
    triangles_ = get_indices(kframe.mesh());
    edges_ = get_edges(kframe.mesh());

    invCovariance_ = Matxf::Identity(numParams, numParams);
    init_params_ = Vecxf::Zero(numParams);

    for (size_t i = 0; i < num_depths; i++)
    {
        init_params_(i) = fromDepthToParam(init_depths_[i]);
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
    }
    init_error_ = err.getError() / err.getCount();

    /*
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
    */

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
    params_ = init_params_;
    error_ = init_error_;

    solver_ = Solverx<float>(numParams);

    if (printLog_)
        std::cout << "mapOptimizer initial error " << init_error_ << " " << in_lvl << " " << out_lvl << std::endl;

    reached_convergence_ = false;
}

void DepthOptimizer::step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
    int num_depths = kframe.mesh().vertex_count();
    int numParams = kframe.mesh().vertex_count();

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

    /*
    if (mesh_vo::mapping_prior_weight > 0.0)
    {
        Vecxf res = init_invcovariancesqrt_ * (params_ - init_params_);
        Matxf jacobian = init_invcovariancesqrt_;
        float weight = mesh_vo::mapping_prior_weight / numParams;
        problem_.add(jacobian, res, weight);
    }
    */

    if (mesh_vo::mapping_prior_weight > 0.0)
    {
        for (std::size_t i = 0; i < depths_.size(); i++)
        {
            float res = fromDepthToParam(depths_[i]) - fromDepthToParam(init_depths_[i]);
            float w = huber_weight(res, mesh_vo::huber_thresh_param);
            Vec<float, 1> jac;
            jac(0) = 1.0f;
            Vec<int, 1> id;
            id(0) = i;
            problem_.add(jac, res, w * mesh_vo::mapping_prior_weight / depths_.size(), id);
        }
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

        // if (printLog_)
        //{
        //     timer_.tic();
        // }
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

        // if (printLog_)
        //{
        //     float t_solve = timer_.toc();
        //     std::cout << "mapOptimizer solve time: " << t_solve << " ms" << std::endl;
        // }

        Vecxf new_params = params_ + inc;
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
            float prior_error = 0.0f;
            for (std::size_t i = 0; i < depths_.size(); i++)
            {
                float res = fromDepthToParam(depths_[i]) - fromDepthToParam(init_depths_[i]);
                float w = huber_weight(res, mesh_vo::huber_thresh_param);
                prior_error += w * res * res;
            }
            new_error += (mesh_vo::mapping_prior_weight / edges_.size()) * prior_error;
        }

        if (printLog_)
            std::cout << "mapOptimizer new error " << new_error << " " << lambda << " " << n_try << " lvl: " << in_lvl << " " << out_lvl << " mesh_regu: " << mesh_vo::mapping_regu_weight << std::endl;

        if (new_error <= error_)
        {
            float p = new_error / error_;
            error_ = new_error;
            depths_ = new_depths;
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

void DepthOptimizer::compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int frame_id, int num_frames, int num_vertices, int in_lvl, int out_lvl, DenseLinearProblemx &total)
{
    // if (printLog_)
    //{
    //     timer_.tic();
    // }
    jmaprenderer_.Render(kframe.mesh(), frame.local_pose(), frame.local_exposure(), cam, in_lvl, out_lvl, kframe.image(), kframe.didxy(), image_texture_, jdepth_texture_, jexp_texture_, pids_texture_);
    // jmap_texture_.generate_mipmaps(out_lvl);
    // jexp_texture_.generate_mipmaps(out_lvl);
    // pids_texture_.generate_mipmaps(out_lvl);
    // r_texture_.generate_mipmaps(out_lvl);
    // if (printLog_)
    //{
    //    float t_render = timer_.toc();
    //    std::cout << "mapOptimizer render time: " << t_render << " ms" << std::endl;
    //    timer_.tic();
    //}
    hgmapreducer_.reduce(out_lvl, frame_id, num_frames, num_vertices, jdepth_texture_, pids_texture_, image_texture_, frame.image(), kframe.mesh(), total);
    // if (printLog_)
    //{
    //     float t_reduce = timer_.toc();
    //     std::cout << "mapOptimizer reduce time: " << t_reduce << " ms" << std::endl;
    // }
}
