#pragma once

#include <span>
#include "params.h"
#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "common/reducer.h"
#include "common/DenseLinearProblem.h"
#include "mpdr/backends/cpu/renderercpu.h"
// #include "cpu/OpenCVDebug.h"

static float regu_depth(const std::vector<float> &depths, const std::vector<Vec2i> &edges)
{
    float regu_error = 0.0f;
    for (size_t i = 0; i < edges.size(); i++)
    {
        Vec2<int> edge = edges[i];

        float param0 = fromDepthToParam(depths[edge(0)]);
        float param1 = fromDepthToParam(depths[edge(1)]);

        float res = param0 - param1;
        float w = huber_weight(res, mesh_vo::huber_thresh_param);
        regu_error += w * res * res;
    }
    return (mesh_vo::mapping_regu_weight / edges.size()) * regu_error;
}

static void regu_depth_jacobian(const std::vector<float> &depths, const std::vector<Vec2i> &edges, DenseLinearProblemx<float> &problem)
{
    float regu_error = 0.0f;
    for (size_t i = 0; i < edges.size(); i++)
    {
        Vec2<int> edge = edges[i];

        float param0 = fromDepthToParam(depths[edge(0)]);
        float param1 = fromDepthToParam(depths[edge(1)]);

        float res = param0 - param1;
        float w = huber_weight(res, mesh_vo::huber_thresh_param);
        Vec2f jac(1.0f, -1.0f);
        problem.add(jac, res, w * mesh_vo::mapping_regu_weight / edges.size(), edge);
    }
}

static float prior_depth(const std::vector<float> &depths,
                         const std::vector<float> &depths_init,
                         const std::vector<float> &params_lambda)
{
    float regu_error = 0.0f;
    const size_t n = depths.size();

    for (size_t i = 0; i < n; ++i)
    {
        float param = fromDepthToParam(depths[i]);
        float param_init = fromDepthToParam(depths_init[i]);
        float param_lambda = params_lambda[i];

        float res = (param - param_init) * param_lambda;
        float w = huber_weight(res, mesh_vo::huber_thresh_param);

        regu_error += w * res * res;
    }

    return (mesh_vo::mapping_prior_weight / n) * regu_error;
}

static void prior_depth_jacobian(const std::vector<float> &depths,
                                 const std::vector<float> &depths_init,
                                 const std::vector<float> &params_lambda,
                                 DenseLinearProblemx<float> &problem)
{
    const size_t n = depths.size();

    for (size_t i = 0; i < n; ++i)
    {
        float param = fromDepthToParam(depths[i]);
        float param_init = fromDepthToParam(depths_init[i]);
        float param_lambda = params_lambda[i];

        float res = (param - param_init) * param_lambda;
        float w = huber_weight(res, mesh_vo::huber_thresh_param);

        Vec<float, 1> jac(param_lambda);
        problem.add(jac,
                    res,
                    w * mesh_vo::mapping_prior_weight / n,
                    Vec<int, 1>(static_cast<int>(i)));
    }
}

static float prior_pose(const SE3f &pose,
                        const SE3f &pose_init,
                        const Mat6f &pose_lambda)
{
    Vec6f res = SE3f(pose_init.inverse() * pose).log();

    float err = (pose_lambda * res).dot(res);
    // float err = 0.0f;
    // for (int k = 0; k < 6; ++k)
    //     err += res(k) * res(k) * pose_lambda(k);

    return mesh_vo::tracking_prior_weight * err;
}

static void prior_pose_jacobian(const SE3f &pose,
                                const SE3f &pose_init,
                                const Mat6f &pose_lambda,
                                DenseLinearProblem<float, 6> &problem)
{
    Vec6f res = SE3f(pose_init.inverse() * pose).log();

    // Matx<float> J(6, 6);
    // J.setZero();
    // for (int k = 0; k < 6; ++k)
    //     J(k, k) = 1.0f;

    problem.add(pose_lambda,
                res,
                mesh_vo::tracking_prior_weight);
}

static float prior_pose_exp(const SE3f &pose,
                            const Vec2f &exp,
                            const SE3f &pose_init,
                            const Vec2f &exp_init,
                            const Mat8f &lambda)
{
    Vec6f pose_res = SE3f(pose_init.inverse() * pose).log();
    Vec2f exp_res = exp_init - exp;
    Vec<float, 8> res;
    res(0) = pose_res(0);
    res(1) = pose_res(1);
    res(2) = pose_res(2);
    res(3) = pose_res(3);
    res(4) = pose_res(4);
    res(5) = pose_res(5);
    res(6) = exp_res(0);
    res(7) = exp_res(1);

    float err = (lambda * res).dot(res);
    // float err = 0.0f;
    // for (int k = 0; k < 6; ++k)
    //     err += res(k) * res(k) * pose_lambda(k);

    return mesh_vo::tracking_prior_weight * err;
}

static void prior_pose_exp_jacobian(const SE3f &pose,
                                    const Vec2f &exp,
                                    const SE3f &pose_init,
                                    const Vec2f &exp_init,
                                    const Mat8f &lambda,
                                    DenseLinearProblem<float, 8> &problem)
{
    Vec6f pose_res = SE3f(pose_init.inverse() * pose).log();
    Vec2f exp_res = exp_init - exp;
    Vec<float, 8> res;
    res(0) = pose_res(0);
    res(1) = pose_res(1);
    res(2) = pose_res(2);
    res(3) = pose_res(3);
    res(4) = pose_res(4);
    res(5) = pose_res(5);
    res(6) = exp_res(0);
    res(7) = exp_res(1);

    // Matx<float> J(6, 6);
    // J.setZero();
    // for (int k = 0; k < 6; ++k)
    //     J(k, k) = 1.0f;

    problem.add(lambda,
                res,
                mesh_vo::tracking_prior_weight);
}

template <class Derived, typename HessianType, typename GradType, typename ErrorType, typename Problem, typename Solver>
class BaseOptimizer
{
public:
    BaseOptimizer(bool printlog)
        : reached_convergence_(false),
          printlog_(printlog)
    {
    }

    bool converged() const
    {
        return reached_convergence_;
    }

    void init(const Frame *frame, const KeyFrame &kframe, const Cameraf &cam, int in_lvl, int out_lvl)
    {
        init(std::span{&frame, 1}, kframe, cam, in_lvl, out_lvl);
    }

    void init(std::span<const Frame *const> frames, const KeyFrame &kframe, const Cameraf &cam, int in_lvl, int out_lvl)
    {
        derived().reset(frames, kframe, problem_, solver_);

        init_error_.setZero();
        init_error_ += derived().compute_error(frames, kframe, cam, in_lvl, out_lvl);
        init_error_ *= 1.0f / init_error_.getCount();

        init_error_ += derived().regu_error();
        // init_error_ += derived().prior_error();

        if (printlog_)
            std::cout << "optimizer " << in_lvl << " " << out_lvl << " initial error: " << init_error_() << std::endl;

        error_ = init_error_;

        reached_convergence_ = false;
    }

    void update(Frame *frame, KeyFrame &kframe, Cameraf &cam)
    {
        update(std::span{&frame, 1}, kframe, cam);
    }

    void update(std::span<Frame *const> frames, KeyFrame &kframe, Cameraf &cam)
    {
        derived().update_params(frames, kframe, problem_);
    }

    void step(Frame *frame, KeyFrame &kframe, Cameraf &cam, int in_lvl, int out_lvl)
    {
        step(std::span{&frame, 1}, kframe, cam, in_lvl, out_lvl);
    }

    void step(std::span<Frame *const> frames, KeyFrame &kframe, Cameraf &cam, int in_lvl, int out_lvl)
    {
        problem_.clear();
        derived().compute_problem(frames, kframe, cam, in_lvl, out_lvl, problem_);
        problem_.scale();
        derived().regu_jacobian(problem_);
        derived().prior_jacobian(problem_);

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

            HessianType Hp_lm = problem_.Hp();

            if (lambda > 0.0)
            {
                for (int i = 0; i < derived().numParams(); i++)
                {
                    Hp_lm(i, i) += lambda;
                    // Hp_lm(i, i) *= 1.0 + lambda;
                }
            }

            solver_.compute(Hp_lm);
            GradType inc = solver_.solve(-problem_.G());

            derived().apply_inc(frames, kframe, inc);

            ErrorType new_error = derived().compute_error(frames, kframe, cam, in_lvl, out_lvl);
            new_error *= 1.0f / new_error.getCount();

            new_error += derived().regu_error();
            new_error += derived().prior_error();

            if (printlog_)
                std::cout << "optimizer " << in_lvl << " " << out_lvl << " new error: " << new_error() << " lambda: " << lambda << std::endl;

            if (new_error() <= error_())
            {
                // if (printlog_)
                //     std::cout << "accepted " << std::endl;

                float p = new_error() / error_();
                error_ = new_error;

                derived().update_best_params();

                if (p >= mesh_vo::mapping_convergence_p)
                {
                    reached_convergence_ = true;
                    if (printlog_)
                        std::cout << "optimizer " << in_lvl << " " << out_lvl << " converged p:" << p << std::endl;
                }
                break;
            }
            else
            {
                // if (printlog_)
                //     std::cout << "rejected" << std::endl;

                derived().restore_best_params(frames, kframe);

                float incMag = inc.dot(inc) / derived().numParams();

                if (incMag <= mesh_vo::mapping_convergence_m_v)
                {
                    reached_convergence_ = true;
                    if (printlog_)
                        std::cout << "optimizer " << in_lvl << " " << out_lvl << " too small " << incMag << std::endl;
                    break;
                }
            }
        }
    }

protected:
    Derived &derived() { return static_cast<Derived &>(*this); }

    Problem problem_;
    Solver solver_;

    ErrorType init_error_;
    ErrorType error_;

    bool reached_convergence_;

    bool printlog_;
};
