#include "optimizers/vertexOptimizer.h"

VertexOptimizer::VertexOptimizer(int w, int h, bool printLog)
    : BaseOptimizer(w, h),
      jv0_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jv1_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jv2_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      jexp_texture_(w, h, Vec3<float>(0.0, 0.0, 0.0)),
      pids_texture_(w, h, Vec3<PidType>(-1, -1, -1)),
      problem_(0),
      printLog_(printLog),
      solver_(0)
{
}

void VertexOptimizer::init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
    int num_vertices = kframe.mesh().vertex_count();
    int numParams = num_vertices * 3;

    init_vertices_ = get_vertices(kframe.mesh());
    init_triangles_ = get_indices(kframe.mesh());

    invCovariance_ = Matxf::Identity(numParams, numParams);
    init_params_ = Vecxf::Zero(numParams);

    for (size_t i = 0; i < num_vertices; i++)
    {
        init_params_(i * 3 + 0) = init_vertices_[i](0);
        init_params_(i * 3 + 1) = init_vertices_[i](1);
        init_params_(i * 3 + 2) = init_vertices_[i](2);
        invCovariance_(i * 3 + 0, i * 3 + 0) = 1.0 / mesh_vo::mapping_param_initial_var;
        invCovariance_(i * 3 + 1, i * 3 + 1) = 1.0 / mesh_vo::mapping_param_initial_var;
        invCovariance_(i * 3 + 2, i * 3 + 2) = 1.0 / mesh_vo::mapping_param_initial_var;
    }

    init_invcovariance_ = invCovariance_;

    if (mesh_vo::mapping_prior_weight > 0.0)
        init_invcovariancesqrt_ = invCovariance_.sqrt();

    init_error_ = 0;
    Error err;
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        compute_error_(frames[i], kframe, cam, in_lvl, out_lvl, err);
        // init_error += err.getError() / err.getCount();
    }
    // init_error *= 1.0 / frames.size();
    init_error_ = err.getError() / err.getCount();

    /*
    if (mesh_vo::mapping_regu_weight > 0.0)
    {
        float regu_error = 0.0f;
        for (size_t i = 0; i < init_triangles_.size(); i++)
        {
            Vec3i id = init_triangles_[i];
            Vec3<float> v0 = init_vertices_[id(0)];
            Vec3<float> v1 = init_vertices_[id(1)];
            Vec3<float> v2 = init_vertices_[id(2)];
            float area = area3d(v0, v1, v2);

            regu_error += area;
        }
        init_error_ += (mesh_vo::mapping_regu_weight / num_vertices) * regu_error;
    }
    */

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

    vertices_ = init_vertices_;
    triangles_ = init_triangles_;
    params_ = init_params_;
    error_ = init_error_;

    solver_ = Solverx<float>(numParams);

    if (printLog_)
        std::cout << "mapOptimizer initial error " << init_error_ << " " << in_lvl << " " << out_lvl << std::endl;

    reached_convergence_ = false;
}

void VertexOptimizer::step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
    int num_vertices = kframe.mesh().vertex_count();
    int numParams = kframe.mesh().vertex_count() * 3;

    problem_.clear(numParams);
    for (std::size_t i = 0; i < frames.size(); i++)
    {
        // DenseLinearProblemx fhg(numParams);
        compute_problem_(frames[i], kframe, cam, i, frames.size(), num_vertices, in_lvl, out_lvl, problem_);

        // if (fhg.count() > 0)
        //{
        //  fhg.scale(1.0 / fhg.count());
        //    problem += fhg;
        //}
    }
    // problem.scale(1.0 / frames.size());
    problem_.scale(1.0 / problem_.count());

    /*
    if (mesh_vo::mapping_regu_weight > 0.0)
    {
        for (size_t i = 0; i < triangles_.size(); i++)
        {
            Vec3i ids = triangles_[i];
            Vec3f v0 = vertices_[ids(0)];
            Vec3f v1 = vertices_[ids(1)];
            Vec3f v2 = vertices_[ids(2)];

            float r1 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(1));
            Vec3<float> jac1(1.0, -1.0, 0.0);
            problem_.add(jac1, r1, mesh_vo::mapping_regu_weight / num_depths, ids);

            float r2 = fromDepthToParam(depth(0)) - fromDepthToParam(depth(2));
            Vec3<float> jac2(1.0, 0.0, -1.0);
            problem_.add(jac2, r2, mesh_vo::mapping_regu_weight / num_depths, ids);

            float r3 = fromDepthToParam(depth(1)) - fromDepthToParam(depth(2));
            Vec3<float> jac3(0.0, 1.0, -1.0);
            problem_.add(jac3, r3, mesh_vo::mapping_regu_weight / num_depths, ids);
        }
    }
    */

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

        if (printLog_)
        {
            timer_.tic();
        }
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

        if (printLog_)
        {
            float t_solve = timer_.toc();
            std::cout << "vertexOptimizer solve time: " << t_solve << " ms" << std::endl;
        }

        Vecxf new_params = params_ + inc;
        std::vector<Vec3<float>> new_vertices;

        for (size_t i = 0; i < num_vertices; i++)
        {
            Vec3<float> new_vertice;
            new_vertice(0) = new_params(i * 3 + 0);
            new_vertice(1) = new_params(i * 3 + 1);
            new_vertice(2) = new_params(i * 3 + 2);

            new_vertices.push_back(new_vertice);
        }

        set_vertices(kframe.mesh(), new_vertices);

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

        /*
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
        */

        if (mesh_vo::mapping_prior_weight > 0.0)
        {
            Vecxf res = new_params - init_params_;
            Vecxf conv_dot_res = init_invcovariance_ * res;
            float weight = mesh_vo::mapping_prior_weight / numParams;
            float priorError = weight * (res.dot(conv_dot_res));

            new_error += priorError;
        }

        if (printLog_)
            std::cout << "vertexOptimizer new error " << new_error << " " << lambda << " " << n_try << " lvl: " << in_lvl << " " << out_lvl << " mesh_regu: " << mesh_vo::mapping_regu_weight << std::endl;

        if (new_error <= error_)
        {
            float p = new_error / error_;
            error_ = new_error;
            vertices_ = new_vertices;
            params_ = new_params;

            if (p >= mesh_vo::mapping_convergence_p)
            {
                reached_convergence_ = true;
                if (printLog_)
                    std::cout << "vertexOptimizer converged p:" << p << std::endl;
            }
            break;
        }
        else
        {
            set_vertices(kframe.mesh(), vertices_);

            float incMag = inc.dot(inc) / numParams;

            if (incMag <= mesh_vo::mapping_convergence_m_v)
            {
                reached_convergence_ = true;
                if (printLog_)
                    std::cout << "vertexOptimizer too small " << incMag << std::endl;
                break;
            }
        }
    }
}

void VertexOptimizer::compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int frame_id, int num_frames, int num_vertices, int in_lvl, int out_lvl, DenseLinearProblemx &total)
{
    if (printLog_)
    {
        timer_.tic();
    }
    jvertexrenderer_.Render(kframe.mesh(), frame.local_pose(), frame.local_exposure(), cam, in_lvl, out_lvl, kframe.image(), kframe.didxy(), image_texture_, jv0_texture_, jv1_texture_, jv2_texture_, jexp_texture_, pids_texture_);
    // jmap_texture_.generate_mipmaps(out_lvl);
    // jexp_texture_.generate_mipmaps(out_lvl);
    // pids_texture_.generate_mipmaps(out_lvl);
    // r_texture_.generate_mipmaps(out_lvl);
    if (printLog_)
    {
        float t_render = timer_.toc();
        std::cout << "vertexOptimizer render time: " << t_render << " ms" << std::endl;
        timer_.tic();
    }

    hgmapreducer_.reduce(out_lvl, frame_id, num_frames, num_vertices, jv0_texture_, jv1_texture_, jv2_texture_, pids_texture_, image_texture_, frame.image(), kframe.mesh(), total);
    if (printLog_)
    {
        float t_reduce = timer_.toc();
        std::cout << "vertexOptimizer reduce time: " << t_reduce << " ms" << std::endl;
    }
}
