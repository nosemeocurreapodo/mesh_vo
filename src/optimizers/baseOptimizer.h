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

static float regu_depth(const std::vector<float> &depths, const std::vector<Vec2<int>> &edges)
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

static void regu_depth_jacobian(const std::vector<float> &depths, const std::vector<Vec2<int>> &edges, DenseLinearProblemx &problem)
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

template <class Derived, typename HessianType, typename GradType, typename Problem, typename Solver>
class BaseOptimizer
{
public:
    BaseOptimizer(int w, int h, bool printlog)
        : image_texture_(w, h, 0),
          reached_convergence_(false),
          printlog_(printlog)
    {
    }

    bool converged() const
    {
        return reached_convergence_;
    }

    void init(const Frame* frame, const KeyFrame &kframe, const Camera &cam, int in_lvl, int out_lvl)
    {
        init(std::span{&frame, 1}, kframe, cam, in_lvl, out_lvl);
    }

    void init(std::span<const Frame* const> frames, const KeyFrame &kframe, const Camera &cam, int in_lvl, int out_lvl)
    {
        derived().reset(frames, kframe, problem_, solver_);

        init_error_.setZero();
        init_error_ += derived().compute_error(frames, kframe, cam, in_lvl, out_lvl);
        init_error_ *= 1.0f / init_error_.getCount();

        init_error_ += derived().regu_error();

        if (printlog_)
            std::cout << "optimizer " << in_lvl << " " << out_lvl << " initial error: " << init_error_() << std::endl;

        error_ = init_error_;

        reached_convergence_ = false;
    }

    void update(Frame *frame, KeyFrame &kframe, Camera &cam)
    {
        update(std::span{&frame, 1}, kframe, cam);
    }

    void update(std::span<Frame* const> frames, KeyFrame &kframe, Camera &cam)
    {
        derived().update_params(frames, kframe);
    }

    void step(Frame *frame, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
    {
        step(std::span{&frame, 1}, kframe, cam, in_lvl, out_lvl);
    }

    void step(std::span<Frame* const> frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
    {
        problem_.clear();
        derived().compute_problem(frames, kframe, cam, in_lvl, out_lvl, problem_);
        problem_.scale(1.0 / problem_.count());
        derived().regu_jacobian(problem_);

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

            Error new_error = derived().compute_error(frames, kframe, cam, in_lvl, out_lvl);
            new_error *= 1.0f / new_error.getCount();

            new_error += derived().regu_error();

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

    ImageRenderer imagerenderer_;
    ResidualReducerCPU residualreducer_;

    Texture<ImageType> image_texture_;

    Problem problem_;
    Solver solver_;

    Error init_error_;
    Error error_;

    bool reached_convergence_;

    bool printlog_;
};
