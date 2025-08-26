#include "optimizers/poseOptimizer.h"

PoseOptimizer::PoseOptimizer(int w, int h, bool print_log)
	: BaseOptimizer(w, h),
	  jac_buffer_(w, h, Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
{
	inv_covariance_ = Mat6::Identity() / mesh_vo::tracking_pose_initial_var;
	print_log_ = print_log;
}

void PoseOptimizer::init(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl)
{
	init_pose_ = frame.local_pose().log();
	init_invcovariance_ = inv_covariance_;

	if (mesh_vo::tracking_prior_weight > 0.0)
		init_invcovariancesqrt_ = inv_covariance_.sqrt();

	Error er = computeError(frame, kframe, cam, lvl);
	init_error_ = er.getError() / er.getCount();

	if (mesh_vo::tracking_prior_weight > 0.0)
	{
		Vec6 res = frame.local_pose().log() - init_pose_;
		Vec6 conv_dot_res = init_invcovariance_ * res;
		float weight = mesh_vo::tracking_prior_weight / 6;
		init_error_ += weight * (res.dot(conv_dot_res));
	}

	if (print_log_)
		std::cout << "poseOptimizer initial error " << init_error_ << " " << lvl << std::endl;

	reached_convergence_ = false;
}

void PoseOptimizer::step(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl)
{
	DenseLinearProblem problem = computeProblem(frame, kframe, cam, lvl);
	// problem *= 1.0 / problem.count();

	/*
	if (mesh_vo::tracking_prior_weight > 0.0)
	{
		// error = diff * (H * diff)
		// jacobian = ones * (H * diff) + diff ( H * ones)
		Vec6 res = init_invcovariancesqrt_ * (frame.local_pose().log() - init_pose_);
		Mat6 jacobian = init_invcovariancesqrt_;
		float weight = mesh_vo::tracking_prior_weight / 6;
		// vec6<float> res(_res);
		// mat6<float> jacobian(_jacobian);
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

		Vecx inc = problem.solve(lambda);

		SE3 best_pose = frame.local_pose();
		SE3 new_pose = frame.local_pose() * SE3::exp(inc); //SE3::exp(inc).inverse();
		frame.local_pose() = new_pose;

		float new_error = 0;
		Error ne = computeError(frame, kframe, cam, lvl);
		if (ne.getCount() < 0.5 * frame.image().size(lvl))
		{
			// too few pixels, unreliable, set to large error
			new_error += init_error_ * 2.0;
		}
		else
		{
			new_error += ne.getError() / ne.getCount();
		}

		if (mesh_vo::tracking_prior_weight > 0.0)
		{
			Vec6 res = frame.local_pose().log() - init_pose_;
			Vec6 conv_dot_res = init_invcovariance_ * res;
			float weight = mesh_vo::tracking_prior_weight / 6;
			new_error += weight * (res.dot(conv_dot_res));
		}

		if (print_log_)
			std::cout << "poseOptimizer new error " << new_error << " " << lambda << " " << " " << lvl << std::endl;

		if (new_error <= init_error_)
		{
			float p = new_error / init_error_;

			init_error_ = new_error;

			if (p >= mesh_vo::tracking_convergence_p)
			{
				reached_convergence_ = true;

				if (print_log_)
					std::cout << "poseOptimizer converged p:" << p << " lvl: " << lvl << std::endl;
			}
			// if update accepted, do next iteration
			break;
		}
		else
		{
			frame.local_pose() = best_pose;
			frame.global_pose() = kframe.localPoseToGlobal(best_pose);

			float poseIncMag = inc.dot(inc) / 6.0;

			if (poseIncMag <= mesh_vo::tracking_convergence_v)
			{
				// std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations and " << t_try << " total tries, with lambda " << lambda << std::endl;
				// if too small, do next level!
				reached_convergence_ = true;

				if (print_log_)
					std::cout << "poseOptimizer too small " << poseIncMag << " lvl: " << lvl << std::endl;

				break;
			}
		}
	}
}

DenseLinearProblem PoseOptimizer::computeProblem(Frame &frame, KeyFrame &kframe, Camera &cam, int lvl)
{
	imagerenderer_.Render(kframe.mesh(), frame.local_pose() * kframe.frame().local_pose().inverse(), cam, kframe.frame().image(), image_buffer_, lvl, lvl);
	jposerenderer_.Render(kframe.mesh(), frame.local_pose() * kframe.frame().local_pose().inverse(), cam, frame.didxy(), jac_buffer_, lvl, lvl);
	return hgposereducer_.reduce(frame.image(), image_buffer_, jac_buffer_, lvl);
}
