#include "optimizers/poseOptimizerCPU.h"

poseOptimizerCPU::poseOptimizerCPU(int width, int height)
    : baseOptimizerCPU(width, height),
      j_buffer(width, height, vec6f::Zero())
{
    invCovariance = mat6f::Identity() / mesh_vo::tracking_pose_initial_var;
}

void poseOptimizerCPU::optimize(frameCPU &frame, keyFrameCPU &kframe, camera &cam)
{
    vec6f init_pose = frame.getLocalPose().log();
    mat6f init_invcovariance = invCovariance;
    mat6f init_invcovariancesqrt;

    if (mesh_vo::tracking_prior_weight > 0.0)
        init_invcovariancesqrt = invCovariance.sqrt();

    for (int lvl = mesh_vo::tracking_ini_lvl; lvl >= mesh_vo::tracking_fin_lvl; lvl--)
    {
        float last_error = 0;
        Error er = computeError(frame, kframe, cam, lvl);
        last_error = er.getError() / er.getCount();

        if (mesh_vo::tracking_prior_weight > 0.0)
        {
            vec6f res = frame.getLocalPose().log() - init_pose;
            vec6f conv_dot_res = init_invcovariance * res;
            float weight = mesh_vo::tracking_prior_weight / 6;
            last_error += weight * (res.dot(conv_dot_res));
        }

        std::cout << "initial error " << last_error << " " << lvl << std::endl;
        std::vector<frameCPU> frames;
        frames.push_back(frame);
        plotDebug(kframe, frames, cam, "poseOptimizerCPU");

        float lambda = 0.0;
        bool keepIterating = true;
        while (keepIterating)
        {
            DenseLinearProblem problem = computeProblem(frame, kframe, cam, lvl);
            problem *= 1.0 / problem.getCount();

            if (mesh_vo::tracking_prior_weight > 0.0)
            {
                // error = diff * (H * diff)
                // jacobian = ones * (H * diff) + diff ( H * ones)
                vec6f res = init_invcovariancesqrt * (frame.getLocalPose().log() - init_pose);
                mat6f jacobian = init_invcovariancesqrt;
                float weight = mesh_vo::tracking_prior_weight / 6;
                // vec6<float> res(_res);
                // mat6<float> jacobian(_jacobian);
                problem.add(jacobian, res, weight);
            }

            int n_try = 0;
            lambda = 0.0;
            while (true)
            {
                if (n_try > 0)
                {
                    if (lambda < mesh_vo::min_lambda)
                        lambda = mesh_vo::min_lambda;
                    lambda *= std::pow(2.0, n_try);
                }
                n_try++;

                if (!problem.prepareH(lambda))
                    continue;

                vecxf inc = problem.solve();

                SE3f best_pose = frame.getLocalPose();
                SE3f new_pose = frame.getLocalPose() * SE3f::exp(inc).inverse();
                frame.setLocalPose(new_pose);

                float new_error = 0;
                Error ne = computeError(frame, kframe, cam, lvl);
                if (ne.getCount() < 0.5 * frame.getRawImage(lvl).width * frame.getRawImage(lvl).height)
                {
                    // too few pixels, unreliable, set to large error
                    new_error += last_error;
                }
                else
                {
                    new_error += ne.getError() / ne.getCount();
                }

                if (mesh_vo::tracking_prior_weight > 0.0)
                {
                    vec6f res = frame.getLocalPose().log() - init_pose;
                    vec6f conv_dot_res = init_invcovariance * res;
                    float weight = mesh_vo::tracking_prior_weight / 6;
                    new_error += weight * (res.dot(conv_dot_res));
                }

                std::cout << "new error " << new_error << " " << lambda << " " << " " << lvl << std::endl;
                std::vector<frameCPU> frames;
                frames.push_back(frame);
                plotDebug(kframe, frames, cam, "poseOptimizerCPU");

                if (new_error <= last_error)
                {
                    float p = new_error / last_error;

                    last_error = new_error;

                    if (p > mesh_vo::tracking_convergence_p)
                    {
                        keepIterating = false;
                    }
                    // if update accepted, do next iteration
                    break;
                }
                else
                {
                    frame.setLocalPose(best_pose);

                    if (inc.dot(inc) < mesh_vo::tracking_convergence_v)
                    {
                        // std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations and " << t_try << " total tries, with lambda " << lambda << std::endl;
                        // if too small, do next level!
                        keepIterating = false;
                        break;
                    }
                }
            }
        }
    }
}

DenseLinearProblem poseOptimizerCPU::computeProblem(frameCPU &frame, keyFrameCPU &kframe, camera &cam, int lvl)
{
    j_buffer.setToNoData(lvl);
    error_buffer.setToNoData(lvl);

    renderer.renderJPoseParallel(kframe, frame, j_buffer, error_buffer, cam, lvl);
    DenseLinearProblem problem = reducer.reduceHGPoseParallel(j_buffer.get(lvl), error_buffer.get(lvl));
    return problem;
}
