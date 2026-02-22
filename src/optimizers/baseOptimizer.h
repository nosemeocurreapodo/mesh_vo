#pragma once

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
    BaseOptimizer(bool printlog)
        : image_texture_(1, 1, 0),
          printlog_(printlog)
    {
    }

    void init(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
    {
        derived().init(frames, kframe, problem_, solver_);

        init_error_ = 0;
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            init_error_ += compute_error_(frames[i], kframe, cam, in_lvl, out_lvl);
        }
        init_error_ += derived().regu_error_();

        error_ = init_error_;

        reached_convergence_ = false;
    }

    void step(std::vector<Frame> &frames, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
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

            derived().update_params(frames, kframe, inc);

            float new_error = 0;
            for (std::size_t i = 0; i < frames.size(); i++)
            {
                new_error += compute_error_(frames[i], kframe, cam, in_lvl, out_lvl);
            }
            new_error /= frames.size();

            new_error += derived().regu_error();

            if (new_error <= error_)
            {
                float p = new_error / error_;
                error_ = new_error;

                derived().update_best_params();

                if (p >= mesh_vo::mapping_convergence_p)
                {
                    reached_convergence_ = true;
                    if (printlog_)
                        std::cout << "poseMapOptimizer converged p:" << p << std::endl;
                }
                break;
            }
            else
            {
                derived().restore_best_params(frames, kframe);

                float incMag = inc.dot(inc) / derived().numParams();

                if (incMag <= mesh_vo::mapping_convergence_m_v)
                {
                    reached_convergence_ = true;
                    if (printlog_)
                        std::cout << "mapOptimizer too small " << incMag << std::endl;
                    break;
                }
            }
        }
    }

protected:
    float compute_error_(const Frame &frame, const KeyFrame &kframe, const Camera &cam, int in_lvl, int out_lvl)
    {
        // imagerenderer_.Render(kframe.mesh(), frame.local_pose() * kframe.frame().local_pose().inverse(), cam, lvl, lvl, kframe.frame().image(), e_texture_);
        // return errorreducer_.reduce(lvl, frame.image(), e_texture_);

        assert(frame.keyframe_id() == kframe.id());
        assert(frame.image().width(0) == kframe.image().width(0) &&
               frame.image().height(0) == kframe.image().height(0));

        if (frame.image().width(0) != image_texture_.width(0) || frame.image().height(0) != image_texture_.height(0))
        {
            image_texture_ = Texture<ImageType>(frame.image().width(0), frame.image().height(0), 0);
        }

        imagerenderer_.Render(kframe.mesh(), frame.local_pose(), frame.local_exposure(), cam, in_lvl, out_lvl, kframe.image(), image_texture_);
        Error total;
        residualreducer_.reduce(out_lvl, image_texture_, frame.image(), total);
        return total.getError() / total.getCount();
    }

    Derived& derived() { return static_cast<Derived&>(*this); }

    ImageRenderer imagerenderer_;
    ResidualReducerCPU residualreducer_;

    Texture<ImageType> image_texture_;

    Problem problem_;
    Solver solver_;

    float init_error_;
    float error_;

    bool reached_convergence_;

    bool printlog_;
};
