#include "optimizers/poseOptimizerCPU.h"

poseOptimizerCPU::poseOptimizerCPU(int width, int height, bool _printLog)
    : baseOptimizerCPU(width, height),
      j_buffer(width, height, Vec6::Zero())
{
    invCovariance = mat6f::Identity() / mesh_vo::tracking_pose_initial_var;
    printLog = _printLog;
}

void poseOptimizerCPU::init(Frame &frame, keyFrameCPU &kframe, cameraType &cam, int lvl)
{
    init_pose = frame.getLocalPose().log();
    init_invcovariance = invCovariance;

    if (mesh_vo::tracking_prior_weight > 0.0)
        init_invcovariancesqrt = invCovariance.sqrt();

    Error er = computeError(frame, kframe, cam, lvl);
    init_error = er.getError() / er.getCount();

    if (mesh_vo::tracking_prior_weight > 0.0)
    {
        vec6f res = frame.getLocalPose().log() - init_pose;
        vec6f conv_dot_res = init_invcovariance * res;
        float weight = mesh_vo::tracking_prior_weight / 6;
        init_error += weight * (res.dot(conv_dot_res));
    }

    if (printLog)
        std::cout << "poseOptimizer initial error " << init_error << " " << lvl << std::endl;

    reachedConvergence = false;
}

void poseOptimizerCPU::step(FrameCPU &frame, KeyFrameCPU &kframe, CameraType &cam, int lvl)
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
            new_error += init_error * 2.0;
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

        if (printLog)
            std::cout << "poseOptimizer new error " << new_error << " " << lambda << " " << " " << lvl << std::endl;

        if (new_error <= init_error)
        {
            float p = new_error / init_error;

            init_error = new_error;

            if (p >= mesh_vo::tracking_convergence_p)
            {
                reachedConvergence = true;

                if (printLog)
                    std::cout << "poseOptimizer converged p:" << p << " lvl: " << lvl << std::endl;
            }
            // if update accepted, do next iteration
            break;
        }
        else
        {
            frame.setLocalPose(best_pose);

            float poseIncMag = inc.dot(inc) / 6.0;

            if (poseIncMag <= mesh_vo::tracking_convergence_v)
            {
                // std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations and " << t_try << " total tries, with lambda " << lambda << std::endl;
                // if too small, do next level!
                reachedConvergence = true;

                if (printLog)
                    std::cout << "poseOptimizer too small " << poseIncMag << " lvl: " << lvl << std::endl;

                break;
            }
        }
    }
}

DenseLinearProblem poseOptimizerCPU::computeProblem(Frame &frame, KeyFrame &kframe, CameraType &cam, int lvl)
{
    jposerenderer.Render(kframe.mesh, frame.pose * kframe.pose.inverse(), cam, frame.didxy, j_buffer, lvl, lvl);
    imagerenderer.Render(kframe.mesh, frame.pose * kframe.pose.inverse(), cam, kframe.image, i_buffer, lvl, lvl);

    DenseLinearProblem problem = hgposereducer.reduce(j_buffer.get(lvl), frame.image.get(lvl), i_buffer);

    return problem;
}

