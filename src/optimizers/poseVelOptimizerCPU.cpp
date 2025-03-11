#include "optimizers/poseVelOptimizerCPU.h"

poseVelOptimizerCPU::poseVelOptimizerCPU(int width, int height)
    : baseOptimizerCPU(width, height),
      jpose_buffer(width, height, jposeType::Zero()),
      jvel_buffer(width, height, jvelType::Zero())
{
}

void poseVelOptimizerCPU::init(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl)
{
    int numPoseParams = 6;
    int numParams = 6;

    invCovariance = matxf::Identity(numParams, numParams);

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

    std::cout << "poseVelOptimizer initial error " << init_error << " " << lvl << std::endl;
    reachedConvergence = false;
}

void poseVelOptimizerCPU::step(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl)
{
    int numPoseParams = 6;
    int numVelParams = 6;
    int numParams = numPoseParams + numVelParams;

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

        jposeType poseInc = inc.segment(0, numPoseParams);
        jvelType velInc = inc.segment(numPoseParams, numVelParams);

        SE3f best_pose = frame.getLocalPose();
        SE3f new_pose = frame.getLocalPose() * SE3f::exp(poseInc).inverse();
        jvelType new_vel = frame.getLocalVel() - velInc;
        frame.setLocalPose(new_pose);
        frame.setLocalVel(new_vel);

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
            float weight = mesh_vo::tracking_prior_weight / numParams;
            new_error += weight * (res.dot(conv_dot_res));
        }

        std::cout << "poseVelOptimizer new error " << new_error << " " << lambda << " " << " " << lvl << std::endl;

        if (new_error <= init_error)
        {
            float p = new_error / init_error;

            init_error = new_error;

            if (p >= mesh_vo::tracking_convergence_p)
            {
                reachedConvergence = true;
                std::cout << "poseVelOptimizer converged p:" << p << " lvl: " << lvl << std::endl;
            }
            // if update accepted, do next iteration
            break;
        }
        else
        {
            frame.setLocalPose(best_pose);

            float poseIncMag = inc.dot(inc)/numParams;

            if (poseIncMag <= mesh_vo::tracking_convergence_v)
            {
                // std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations and " << t_try << " total tries, with lambda " << lambda << std::endl;
                // if too small, do next level!
                reachedConvergence = true;
                std::cout << "poseVelOptimizer too small " << poseIncMag << " lvl: " << lvl << std::endl;
                break;
            }
        }
    }
}

DenseLinearProblem poseVelOptimizerCPU::computeProblem(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl)
{
    jpose_buffer.setToNoData(lvl);
    jvel_buffer.setToNoData(lvl);
    error_buffer.setToNoData(lvl);

    renderer.renderJPoseVelParallel(kframe, frame, jpose_buffer, jvel_buffer, error_buffer, cam, lvl);
    DenseLinearProblem problem = reducer.reduceHGPoseVelParallel(jpose_buffer.get(lvl), jvel_buffer.get(lvl), error_buffer.get(lvl));
    return problem;
}

std::vector<dataCPU<float>> poseVelOptimizerCPU::getDebugData(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl)
{
    std::vector<dataCPU<float>> toShow;

    toShow.push_back(kframe.getRawImage(lvl).convert<float>());

    depth_buffer.setToNoData(lvl);
    //weight_buffer.setToNoData(lvl);

    renderer.renderDepthParallel(kframe, SE3f(), depth_buffer, cam, lvl);
    //renderer.renderWeightParallel(kframe, SE3f(), weight_buffer, cam, lvl);

    depth_buffer.get(lvl).invert();

    toShow.push_back(depth_buffer.get(lvl));
    //toShow.push_back(weight_buffer.get(lvl));

    error_buffer.setToNoData(lvl);
    // depth_buffer.setToNoData(lvl);
    renderer.renderResidualParallel(kframe, frame, error_buffer, cam, lvl);
    // renderer.renderDepthParallel(kframe, frames[i].getLocalPose(), depth_buffer, cam, lvl);
    toShow.push_back(frame.getRawImage(lvl).convert<float>());
    toShow.push_back(error_buffer.get(lvl).convert<float>());
    // toShow.push_back(depth_buffer.get(lvl));

    return toShow;
}
