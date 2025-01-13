#pragma once

#include <Eigen/Core>
// #include <Eigen/CholmodSupport>
//  #include <Eigen/SPQRSupport>
// #include <thread>

#include "common/camera.h"
#include "common/types.h"
#include "common/DenseLinearProblem.h"
// #include "common/SparseLinearProblem.h"
//  #include "common/HGMapped.h"
#include "common/Error.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/reduceCPU.h"
// #include "cpu/SceneBase.h"
// #include "cpu/ScenePatches.h"
#include "cpu/SceneMesh.h"
// #include "cpu/SceneSurfels.h"
// #include "cpu/SceneMeshSmooth.h"
#include "optimizers/baseOptimizerCPU.h"
#include "cpu/OpenCVDebug.h"
#include "params.h"

template <typename sceneType>
class poseOptimizerCPU : public baseOptimizerCPU<sceneType>
{
public:
    poseOptimizerCPU(camera &_cam)
        : baseOptimizerCPU<sceneType>(_cam),
          j_buffer(_cam.width, _cam.height, vec6<float>(0.0))
    {
        priorWeight = 10.0;
    }

    void optimize(frameCPU &frame, frameCPU &kframe, sceneType &scene)
    {
        Eigen::Matrix<float, 6, 1> init_pose = frame.getPose().log();
        Eigen::Matrix<float, 6, 6> init_invcovariance = invCovariance;

        for (int lvl = 2; lvl >= 2; lvl--)
        {
            Error last_error = baseOptimizerCPU<sceneType>::computeError(frame, kframe, scene, lvl);
            last_error *= 1.0 / last_error.getCount();

            if (priorWeight > 0.0)
            {
                Eigen::Matrix<float, 6, 1> res = frame.getPose().log() - init_pose;
                Eigen::Matrix<float, 6, 1> conv_dot_res = init_invcovariance * res;
                last_error += priorWeight * (res.dot(conv_dot_res));
            }

            std::cout << "initial error " << last_error.getError() << " " << lvl << std::endl;
            std::vector<frameCPU> frames;
            frames.push_back(frame);
            baseOptimizerCPU<sceneType>::plotDebug(scene, kframe, frames);

            float lambda = 0.0;
            bool keepIterating = true;
            while (keepIterating)
            {
                DenseLinearProblem problem = computeProblem(frame, kframe, scene, lvl);
                problem *= 1.0 / problem.getCount();

                invCovariance = problem.getH();

                if (priorWeight > 0.0)
                {
                    Eigen::Matrix<float, 6, 1> _res = frame.getPose().log() - init_pose;
                    Eigen::Matrix<float, 6, 1> _jacobian = init_invcovariance * frame.getPose().log();
                    vec6<float> res(_res(0), _res(1), _res(2), _res(3), _res(4), _res(5));
                    vec6<float> jacobian(_jacobian(0), _jacobian(1), _jacobian(2), _jacobian(3), _jacobian(4), _jacobian(5));
                    vec6<int> ids(0, 1, 2, 3, 4, 5);
                    problem.add(jacobian, res, 1.0, ids);
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

                    Eigen::VectorXf poseUpdate = problem.solve();

                    Sophus::SE3f best_pose = frame.getPose();
                    Sophus::SE3f new_pose = frame.getPose() * Sophus::SE3f::exp(poseUpdate).inverse();
                    frame.setPose(new_pose);

                    float update_mag = poseUpdate.dot(poseUpdate);

                    Error new_error = baseOptimizerCPU<sceneType>::computeError(frame, kframe, scene, lvl);
                    if (new_error.getCount() < 0.5 * baseOptimizerCPU<sceneType>::cam[lvl].width * baseOptimizerCPU<sceneType>::cam[lvl].height)
                    {
                        // too few pixels, unreliable, set to large error
                        new_error.setZero();
                        new_error += last_error;
                    }
                    new_error *= 1.0 / new_error.getCount();

                    std::cout << "new error " << new_error.getError() << " " << lambda << " " << " " << lvl << std::endl;
                    std::vector<frameCPU> frames;
                    frames.push_back(frame);
                    baseOptimizerCPU<sceneType>::plotDebug(scene, kframe, frames);

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

                        if (update_mag < 1e-16)
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
        baseOptimizerCPU<sceneType>::error_buffer.setToNoData(lvl);

        baseOptimizerCPU<sceneType>::renderer.renderJPoseParallel(scene, kframe.getRawImage(lvl), kframe.getAffine(), kframe.getPose(), frame.getRawImage(lvl), frame.getAffine(), frame.getdIdpixImage(lvl), frame.getPose(), baseOptimizerCPU<sceneType>::cam[lvl], j_buffer.get(lvl), baseOptimizerCPU<sceneType>::error_buffer.get(lvl));
        DenseLinearProblem problem = baseOptimizerCPU<sceneType>::reducer.reduceHGPoseParallel(j_buffer.get(lvl), baseOptimizerCPU<sceneType>::error_buffer.get(lvl));
        return problem;
    }

    float priorWeight;
    dataMipMapCPU<vec6<float>> j_buffer;
    Eigen::Matrix<float, 6, 6> invCovariance;
};
