#include "optimizers/poseOptimizer.h"

PoseOptimizer::PoseOptimizer(int w, int h, bool print_log)
	: BaseOptimizer(w, h),
	  jtra_texture_(w, h, Vec3f(0.0, 0.0, 0.0)),
	  jrot_texture_(w, h, Vec3f(0.0, 0.0, 0.0)),
	  jexp_texture_(w, h, Vec3f(0.0, 0.0, 0.0))
{
	inv_covariance_ = Mat6f::Identity() / mesh_vo::tracking_pose_initial_var;
	print_log_ = print_log;
}

void PoseOptimizer::init(Frame &frame, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
	init_pose_ = frame.local_pose();
	init_invcovariance_ = inv_covariance_;

	if (mesh_vo::tracking_prior_weight > 0.0)
		init_invcovariancesqrt_ = inv_covariance_.sqrt();

	Error er = compute_error_(frame, kframe, cam, in_lvl, out_lvl);
	init_error_ = er.getError() / er.getCount();

	if (mesh_vo::tracking_prior_weight > 0.0)
	{
		Vec6f res = frame.local_pose().log() - init_pose_.log();
		Vec6f conv_dot_res = init_invcovariance_ * res;
		float weight = mesh_vo::tracking_prior_weight / 6;
		init_error_ += weight * (res.dot(conv_dot_res));
	}

	pose_ = init_pose_;
	error_ = init_error_;

	if (print_log_)
		std::cout << "poseOptimizer initial error " << init_error_ << " " << in_lvl << " " << out_lvl << std::endl;

	reached_convergence_ = false;
}

void PoseOptimizer::step(Frame &frame, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
	DenseLinearProblem<6> problem = compute_problem_(frame, kframe, cam, in_lvl, out_lvl);
	// problem *= 1.0 / problem.count();
	/*
	if (mesh_vo::tracking_prior_weight > 0.0)
	{
		// error = diff * (H * diff)
		// jacobian = ones * (H * diff) + diff ( H * ones)
		Vec6f res = init_invcovariancesqrt_ * (pose_.log() - init_pose_.log());
		Mat6f jacobian = init_invcovariancesqrt_;
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

		Vec6f inc = problem.solve(lambda);
		Vec6f pose_inc;
		pose_inc(0) = inc(0);
		pose_inc(1) = inc(1);
		pose_inc(2) = inc(2);
		pose_inc(3) = inc(3);
		pose_inc(4) = inc(4);
		pose_inc(5) = inc(5);

		SE3f new_pose = pose_ * SE3f::exp(pose_inc); // SE3::exp(inc).inverse();

		frame.local_pose() = new_pose;

		float new_error = 0;
		Error ne = compute_error_(frame, kframe, cam, in_lvl, out_lvl);
		if (ne.getCount() < 0.5 * frame.image().width(out_lvl) * frame.image().height(out_lvl))
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
			Vec6f res = new_pose.log() - init_pose_.log();
			Vec6f conv_dot_res = init_invcovariance_ * res;
			float weight = mesh_vo::tracking_prior_weight / 6;
			new_error += weight * (res.dot(conv_dot_res));
		}

		if (print_log_)
			std::cout << "poseOptimizer new error " << new_error << " " << lambda << " " << in_lvl << " " << out_lvl << std::endl;

		if (new_error <= error_)
		{
			float p = new_error / error_;

			pose_ = new_pose;
			error_ = new_error;

			if (p >= mesh_vo::tracking_convergence_p)
			{
				reached_convergence_ = true;

				if (print_log_)
					std::cout << "poseOptimizer converged p:" << p << " lvl: " << in_lvl << " " << out_lvl << std::endl;
			}
			// if update accepted, do next iteration
			break;
		}
		else
		{
			frame.local_pose() = pose_;

			float poseIncMag = inc.dot(inc) / 6.0;

			if (poseIncMag <= mesh_vo::tracking_convergence_v)
			{
				// std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations and " << t_try << " total tries, with lambda " << lambda << std::endl;
				// if too small, do next level!
				reached_convergence_ = true;

				if (print_log_)
					std::cout << "poseOptimizer too small " << poseIncMag << " lvl: " << in_lvl << " " << out_lvl << std::endl;

				break;
			}
		}
	}
}

DenseLinearProblem<6> PoseOptimizer::compute_problem_(Frame &frame, KeyFrame &kframe, Camera &cam, int in_lvl, int out_lvl)
{
	jposerenderer_.Render(kframe.mesh(), frame.local_pose(), frame.local_exposure(), cam, in_lvl, out_lvl, kframe.frame().image(), frame.image(), kframe.frame().didxy(), jtra_texture_, jrot_texture_, jexp_texture_, r_texture_);
	return hgposereducer_.reduce(out_lvl, jtra_texture_, jrot_texture_, r_texture_);
}
