#pragma once

#include "params.h"
#include "common/camera.h"
#include "common/types.h"
#include "common/Error.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/reduceCPU.h"
#include "cpu/SceneMesh.h"
#include "common/DenseLinearProblem.h"
#include "optimizers/baseOptimizerCPU.h"
#include "cpu/OpenCVDebug.h"

class poseOptimizerCPU : public baseOptimizerCPU
{
public:
    poseOptimizerCPU(camera &_cam)
        : baseOptimizerCPU(_cam),
          j_buffer(_cam.width, _cam.height, vec6f::Zero())
    {
        priorWeight = 0.0;
        invCovariance = mat6f::Identity() / (INITIAL_POSE_STD * INITIAL_POSE_STD);
    }

    void optimize(frameCPU &frame, frameCPU &kframe, sceneType &scene)
    {
        vec6f init_pose = frame.getPose().log();
        mat6f init_invcovariance = invCovariance;
        mat6f init_invcovariancesqrt;
        
        if(priorWeight > 0.0)
            init_invcovariancesqrt = invCovariance.sqrt();

        for (int lvl = 2; lvl >= 2; lvl--)
        {
            Error last_error = computeError(frame, kframe, scene, lvl);
            last_error *= 1.0 / last_error.getCount();

            if (priorWeight > 0.0)
            {
                vec6f res = frame.getPose().log() - init_pose;
                vec6f conv_dot_res = init_invcovariance * res;
                float weight = priorWeight / 6;
                last_error += weight * (res.dot(conv_dot_res));
            }

            std::cout << "initial error " << last_error.getError() << " " << lvl << std::endl;
            std::vector<frameCPU> frames;
            frames.push_back(frame);
            plotDebug(scene, kframe, frames);

            float lambda = 0.0;
            bool keepIterating = true;
            while (keepIterating)
            {
                DenseLinearProblem problem = computeProblem(frame, kframe, scene, lvl);
                problem *= 1.0 / problem.getCount();

                if (priorWeight > 0.0)
                {
                    // error = diff * (H * diff)
                    // jacobian = ones * (H * diff) + diff ( H * ones)
                    vec6f res = init_invcovariancesqrt * (frame.getPose().log() - init_pose);
                    mat6f jacobian = init_invcovariancesqrt;
                    float weight = priorWeight / 6;
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
                        if (lambda < MIN_LAMBDA)
                            lambda = MIN_LAMBDA;
                        lambda *= std::pow(2.0, n_try);
                    }
                    n_try++;

                    if (!problem.prepareH(lambda))
                        continue;

                    vecxf inc = problem.solve();

                    SE3f best_pose = frame.getPose();
                    SE3f new_pose = frame.getPose() * SE3f::exp(inc).inverse();
                    frame.setPose(new_pose);

                    Error new_error = computeError(frame, kframe, scene, lvl);
                    if (new_error.getCount() < 0.5 * cam[lvl].width * cam[lvl].height)
                    {
                        // too few pixels, unreliable, set to large error
                        new_error.setZero();
                        new_error += last_error;
                    }
                    new_error *= 1.0 / new_error.getCount();

                    if (priorWeight > 0.0)
                    {
                        vec6f res = frame.getPose().log() - init_pose;
                        vec6f conv_dot_res = init_invcovariance * res;
                        float weight = priorWeight / 6;
                        new_error += weight * (res.dot(conv_dot_res));
                    }

                    std::cout << "new error " << new_error.getError() << " " << lambda << " " << " " << lvl << std::endl;
                    std::vector<frameCPU> frames;
                    frames.push_back(frame);
                    plotDebug(scene, kframe, frames);

                    if (new_error.getError() < last_error.getError())
                    {
                        float p = new_error.getError() / last_error.getError();

                        last_error = new_error;

                        if (p > 0.999f)
                        {
                            keepIterating = false;
                        }
                        // if update accepted, do next iteration
                        break;
                    }
                    else
                    {
                        frame.setPose(best_pose);

                        if (inc.dot(inc) < 1e-16)
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

private:
    DenseLinearProblem computeProblem(frameCPU &frame, frameCPU &kframe, sceneType &scene, int lvl)
    {
        j_buffer.set(j_buffer.nodata, lvl);
        error_buffer.setToNoData(lvl);

        renderer.renderJPoseParallel(scene, kframe.getRawImage(lvl), kframe.getExposure(), kframe.getPose(), frame.getRawImage(lvl), frame.getExposure(), frame.getdIdpixImage(lvl), frame.getPose(), cam[lvl], j_buffer.get(lvl), error_buffer.get(lvl));
        DenseLinearProblem problem = reducer.reduceHGPoseParallel(j_buffer.get(lvl), error_buffer.get(lvl));
        return problem;
    }

    float priorWeight;
    dataMipMapCPU<vec6f> j_buffer;
    mat6f invCovariance;
};
