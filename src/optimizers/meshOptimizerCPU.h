#pragma once

#include <Eigen/Core>
#include <Eigen/CholmodSupport>
// #include <Eigen/SPQRSupport>
#include <thread>

#include "common/camera.h"
#include "common/types.h"
#include "common/HGEigenDense.h"
#include "common/HGEigenSparse.h"
// #include "common/HGMapped.h"
#include "common/Error.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/reduceCPU.h"
#include "cpu/SceneBase.h"
#include "cpu/ScenePatches.h"
#include "cpu/SceneMesh.h"
// #include "cpu/SceneSurfels.h"
// #include "cpu/SceneMeshSmooth.h"
#include "cpu/OpenCVDebug.h"
#include "params.h"

template <typename sceneType, typename shapeType, typename jmapType, typename idsType>
class meshOptimizerCPU
{
public:
    meshOptimizerCPU(camera &_cam);
    void initKeyframe(frameCPU &frame, int lvl);

    void initKeyframe(frameCPU &frame, dataCPU<float> &idepth, dataCPU<float> &ivar, int lvl)
    {
        kscene.init(cam[lvl], idepth, ivar, lvl);

        /*
        for(int i = 0; i < 1; i++)
        {
            renderer.renderIdepth(&kscene, cam[lvl], &idepth_buffer, lvl);
            dataCPU<float> diff = idepth.sub(idepth_buffer, lvl);

            int added_vertices = kscene.updateMeshGivenErrorAndThresh(frame, cam[lvl], diff, 0.1, lvl);
            if(added_vertices == 0)
                break;
        }
        */

        kimage = frame.getRawImage();
        kpose = frame.getPose();
        // kframe.setAffine(vec2<float>(0.0, 0.0));
    }

    void initKeyframe(frameCPU &frame, std::vector<vec2<float>> &texcoords, std::vector<float> &idepths, int lvl)
    {
        kscene.init(cam[lvl], texcoords, idepths);
        kimage = frame.getRawImage();
        kpose = frame.getPose();
    }

    void normalizeDepth()
    {
        vec2<float> affine = kscene.meanStdDepth();
        affine(1) = 0.0;
        kscene.scaleDepth(affine);
        kDepthAffine = affine;
    }

    void optLightAffine(frameCPU &frame)
    {
        int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

        for (int lvl = 3; lvl >= 1; lvl--)
        {
            vec2<float> best_affine = frame.getAffine();
            Error e = computeError(&frame, lvl, false);
            float last_error = e.getError();

            std::cout << "initial error " << last_error << " " << lvl << std::endl;

            for (int it = 0; it < maxIterations[lvl]; it++)
            {
                // HGPose hg = computeHGPose(idepth_buffer, keyframe, frame, lvl);
                HGEigenDense<2> hg = computeHGLightAffine(&frame, lvl, false);

                Eigen::VectorXf G = hg.getG();
                Eigen::Matrix<float, 2, 2> H = hg.getH();

                float lambda = 0.0;
                int n_try = 0;
                while (true)
                {
                    Eigen::Matrix<float, 2, 2> H_lambda;
                    H_lambda = H;

                    for (int j = 0; j < 2; j++)
                        H_lambda(j, j) *= 1.0 + lambda;

                    // Eigen::Matrix<float, 6, 1> inc = H_lambda.ldlt().solve(G);
                    Eigen::VectorXf inc = Eigen::VectorXf::Zero(2);
                    inc = H_lambda.ldlt().solve(G);

                    // Sophus::SE3f new_pose = frame.pose * Sophus::SE3f::exp(inc_pose);
                    vec2<float> new_affine = best_affine - vec2<float>(inc(0), inc(1));
                    frame.setAffine(new_affine);
                    // Sophus::SE3f new_pose = Sophus::SE3f::exp(inc_pose).inverse() * frame.pose;

                    // e.setZero();
                    // e = computeError(idepth_buffer, keyframe, frame, lvl);
                    e = computeError(&frame, lvl, false);
                    float error = e.getError();
                    // std::cout << "new error " << error << " time " << t.toc() << std::endl;

                    std::cout << "new error " << error << " " << lambda << " " << it << " " << lvl << std::endl;

                    if (error < last_error)
                    {
                        // accept update, decrease lambda

                        best_affine = new_affine;
                        float p = error / last_error;

                        // if (lambda < 0.2f)
                        //     lambda = 0.0f;
                        // else
                        // lambda *= 0.5;
                        lambda = 0.0;

                        last_error = error;

                        if (p > 0.999f)
                        {
                            // if converged, do next level
                            it = maxIterations[lvl];
                        }
                        // if update accepted, do next iteration
                        break;
                    }
                    else
                    {
                        frame.setAffine(new_affine);

                        n_try++;

                        if (lambda == 0.0)
                            lambda = 0.2f;
                        else
                            lambda *= 2.0; // std::pow(2.0, n_try);

                        // reject update, increase lambda, use un-updated data
                        // std::cout << "update rejected " << std::endl;

                        if (!(inc.dot(inc) > 1e-16))
                        {
                            // std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations and " << t_try << " total tries, with lambda " << lambda << std::endl;
                            // if too small, do next level!
                            it = maxIterations[lvl];
                            break;
                        }
                    }
                }
            }
        }
    }

    void optPose(frameCPU &frame)
    {
        int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

        for (int lvl = 4; lvl >= 1; lvl--)
        {
            // std::cout << "*************************lvl " << lvl << std::endl;
            Sophus::SE3f best_pose = frame.getPose();
            vec2<float> best_affine = frame.getAffine();
            // Error e = computeError(idepth_buffer, keyframe, frame, lvl);
            Error e = computeError(&frame, lvl, false);
            float last_error = e.getError();

            std::cout << "initial error " << last_error << " " << lvl << std::endl;

            for (int it = 0; it < maxIterations[lvl]; it++)
            {
                // HGPose hg = computeHGPose(idepth_buffer, keyframe, frame, lvl);
                HGEigenDense hg = computeHGPose(&frame, lvl, false);

                // std::vector<int> pIds = hg.G.getParamIds();
                // Eigen::VectorXf G = hg.G.toEigen(pIds);
                // Eigen::SparseMatrix<float> H = hg.H.toEigen(pIds);

                Eigen::VectorXf G = hg.getG();
                Eigen::Matrix<float, 8, 8> H = hg.getH();

                float lambda = 0.0;
                int n_try = 0;
                while (true)
                {
                    Eigen::Matrix<float, 8, 8> H_lambda;
                    H_lambda = H;

                    for (int j = 0; j < 8; j++)
                        H_lambda(j, j) *= 1.0 + lambda;

                    // Eigen::Matrix<float, 6, 1> inc = H_lambda.ldlt().solve(G);
                    Eigen::VectorXf inc = Eigen::VectorXf::Zero(8);
                    inc = H_lambda.ldlt().solve(G);

                    // Sophus::SE3f new_pose = frame.pose * Sophus::SE3f::exp(inc_pose);
                    Sophus::SE3f new_pose = best_pose * Sophus::SE3f::exp(inc.segment(0, 6)).inverse();
                    vec2<float> new_affine = best_affine; // - vec2<float>(inc(6), inc(7));
                    frame.setPose(new_pose);
                    frame.setAffine(new_affine);
                    // Sophus::SE3f new_pose = Sophus::SE3f::exp(inc_pose).inverse() * frame.pose;

                    // e.setZero();
                    // e = computeError(idepth_buffer, keyframe, frame, lvl);
                    e = computeError(&frame, lvl, false);
                    float error = e.getError();
                    // std::cout << "new error " << error << " time " << t.toc() << std::endl;

                    std::cout << "new error " << error << " " << lambda << " " << it << " " << lvl << std::endl;

                    if (error < last_error)
                    {
                        // accept update, decrease lambda

                        best_pose = new_pose;
                        best_affine = new_affine;
                        float p = error / last_error;

                        // if (lambda < 0.2f)
                        //     lambda = 0.0f;
                        // else
                        // lambda *= 0.5;
                        lambda = 0.0;

                        last_error = error;

                        if (p > 0.999f)
                        {
                            // if converged, do next level
                            it = maxIterations[lvl];
                        }
                        // if update accepted, do next iteration
                        break;
                    }
                    else
                    {
                        frame.setPose(best_pose);
                        frame.setAffine(new_affine);

                        n_try++;

                        if (lambda == 0.0)
                            lambda = 0.2f;
                        else
                            lambda *= 2.0; // std::pow(2.0, n_try);

                        // reject update, increase lambda, use un-updated data
                        // std::cout << "update rejected " << std::endl;

                        if (!(inc.dot(inc) > 1e-16))
                        {
                            // std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations and " << t_try << " total tries, with lambda " << lambda << std::endl;
                            // if too small, do next level!
                            it = maxIterations[lvl];
                            break;
                        }
                    }
                }
            }
        }
    }

    void optMap(std::vector<frameCPU> &frames, dataCPU<float> &mask)
    {
        Error e;
        Error e_regu;
        Error e_init;

        HGEigenSparse hg(kscene.getNumParams());
        HGEigenSparse hg_regu(kscene.getNumParams());
        HGEigenSparse hg_init(kscene.getNumParams());

        // HGMapped hg;
        // HGMapped hg_regu;
        // HGMapped hg_init;

        for (int lvl = 4; lvl >= 1; lvl--)
        {
            e.setZero();
            for (std::size_t i = 0; i < frames.size(); i++)
            {
                e += computeError(&frames[i], lvl);
            }

            float last_error = e.getError();

            if (meshRegularization > 0.0)
            {
                e_regu = kscene.errorRegu(cam[lvl]);
                last_error += meshRegularization * e_regu.getError();
            }

            // e_init = keyframeScene.errorInitial(initialScene, initialInvVar);
            //  e_init.error /= e_init.count;

            std::cout << "optMap initial error " << last_error << " " << lvl << std::endl;

            int maxIterations = 1000;
            float lambda = 0.0;
            for (int it = 0; it < maxIterations; it++)
            {
                hg.setZero();
                for (std::size_t i = 0; i < frames.size(); i++)
                {
                    HGEigenSparse _hg = computeHGMap2(&frames[i], &mask, lvl);
                    hg += _hg;
                }

                // saveH(hg, "H.png");

                std::map<int, int> paramIds = hg.getObservedParamIds();
                // std::map<int, int> paramIds = hg.getParamIds();

                Eigen::VectorXf G = hg.getG(paramIds);
                Eigen::SparseMatrix<float> H = hg.getH(paramIds);

                if (meshRegularization > 0.0)
                {
                    hg_regu = kscene.HGRegu(cam[lvl]);

                    Eigen::VectorXf G_regu = hg_regu.getG(paramIds);
                    Eigen::SparseMatrix<float> H_regu = hg_regu.getH(paramIds);

                    H += meshRegularization * H_regu; // + meshInitial * H_init;
                    G += meshRegularization * G_regu; // + meshInitial * G_init;
                }

                // hg_init = HGInitial(initialScene, initialInvVar);

                // Eigen::VectorXf G_init = hg_init.G.toEigen(ids);
                // Eigen::SparseMatrix<float> H_init = hg_init.H.toEigen(ids);

                // H += meshInitial * H_init;
                // G += meshInitial * G_init;

                int n_try = 0;
                while (true)
                {
                    Eigen::SparseMatrix<float> H_lambda = H;

                    for (int j = 0; j < H_lambda.rows(); j++)
                    {
                        H_lambda.coeffRef(j, j) *= (1.0 + lambda);
                    }

                    bool solverSucceded = true;

                    H_lambda.makeCompressed();
                    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
                    Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
                    //   Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;

                    // Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
                    // Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;

                    // Eigen::CholmodDecomposition<Eigen::SparseMatrix<float>, Eigen::Lower> solver;
                    // solver.setMode(Eigen::CholmodSupernodalLLt);

                    // Eigen::SPQR<Eigen::SparseMatrix<float>> solver;

                    solver.compute(H_lambda);
                    // solver.analyzePattern(H_lambda);
                    // solver.factorize(H_lambda);

                    if (solver.info() != Eigen::Success)
                    {
                        // some problem i have still to debug
                        solverSucceded = false;
                    }
                    // std::cout << solver.lastErrorMessage() << std::endl;

                    Eigen::VectorXf inc = solver.solve(G);
                    // Eigen::VectorXf inc = G / (1.0 + lambda);
                    //    inc_depth = -acc_H_depth_lambda.llt().solve(acc_J_depth);
                    //    inc_depth = - acc_H_depth_lambda.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(acc_J_depth);
                    //    inc_depth = -acc_H_depth_lambda.colPivHouseholderQr().solve(acc_J_depth);

                    if (solver.info() != Eigen::Success)
                    {
                        // solving failed
                        solverSucceded = false;
                    }

                    std::vector<float> best_params;
                    float map_inc_mag = 0.0;
                    int map_inc_mag_count = 0;
                    for (auto id : paramIds)
                    {
                        float best_param = kscene.getParam(id.first);
                        float inc_param = inc(id.second);
                        float new_param = best_param - inc_param;

                        // if (std::fabs(inc_param / best_param) > 0.1)
                        // if(new_param <= 0.0)
                        if (false)
                        {
                            solverSucceded = false;
                            // break;
                        }

                        float weight = H.coeffRef(id.second, id.second);
                        best_params.push_back(best_param);
                        // the derivative is with respecto to the keyframe pose
                        // the update should take this into account
                        kscene.setParam(new_param, id.first);
                        kscene.setParamWeight(weight, id.first);
                        map_inc_mag += inc_param * inc_param;
                        map_inc_mag_count += 1;
                    }
                    map_inc_mag /= map_inc_mag_count;

                    e.setZero();
                    for (std::size_t i = 0; i < frames.size(); i++)
                    {
                        e += computeError(&frames[i], lvl);
                    }

                    float error = e.getError();

                    if (meshRegularization > 0.0)
                    {
                        e_regu = kscene.errorRegu(cam[lvl]);
                        error += meshRegularization * e_regu.getError();
                    }

                    // e_init = errorInitial(initialMesh, initialInvVar);
                    // e_init.error /= e_init.count;

                    std::cout << "new error " << error << " " << lambda << " " << it << " " << lvl << std::endl;

                    if (error < last_error && solverSucceded)
                    {
                        // accept update, decrease lambda
                        float p = error / last_error;

                        // if (lambda < 0.2f)
                        //     lambda = 0.0f;
                        // else
                        lambda *= 0.5;
                        // lambda = 0.0;

                        last_error = error;

                        if (p > 0.999f)
                        {
                            // std::cout << "lvl " << lvl << " converged after " << it << " itarations with lambda " << lambda << std::endl;
                            //  if converged, do next level
                            it = maxIterations;
                        }

                        // if update accepted, do next iteration
                        break;
                    }
                    else
                    {
                        for (auto id : paramIds)
                            kscene.setParam(best_params[id.second], id.first);

                        n_try++;

                        if (lambda == 0.0f)
                            lambda = 0.01f;
                        else
                            lambda *= 4.0; // std::pow(2.0, n_try);

                        // reject update, increase lambda, use un-updated data

                        if (map_inc_mag < 1e-16)
                        {
                            // if too small, do next level!
                            it = maxIterations;
                            break;
                        }
                    }
                }
            }
        }
    }

    void optPoseMap(std::vector<frameCPU> &frames)
    {
        Error e;
        Error e_regu;
        HGEigenSparse hg(kscene.getNumParams() + frames.size() * 8);
        HGEigenSparse hg_regu(kscene.getNumParams() + frames.size() * 8);

        for (int lvl = 4; lvl >= 1; lvl--)
        {
            e.setZero();
            for (std::size_t i = 0; i < frames.size(); i++)
            {
                e += computeError(&frames[i], lvl);
            }

            float last_error = e.getError();

            if (meshRegularization > 0.0)
            {
                e_regu = kscene.errorRegu(cam[lvl]);
                last_error += meshRegularization * e_regu.getError();
            }

            std::cout << "optPoseMap initial error " << last_error << " lvl: " << lvl << std::endl;

            int maxIterations = 1000;
            float lambda = 0.0;
            for (int it = 0; it < maxIterations; it++)
            {
                hg.setZero();
                for (std::size_t i = 0; i < frames.size(); i++)
                {
                    HGEigenSparse _hg = computeHGPoseMap2(&frames[i], i, frames.size(), lvl);
                    hg += _hg;
                }

                // saveH(hg, "H.png");

                // map from param id to param index paramIndex = obsParamIds[paramId]
                std::map<int, int> paramIds = hg.getObservedParamIds();
                // std::map<int, int> paramIds = hg.getParamIds();

                Eigen::VectorXf G = hg.getG(paramIds);
                Eigen::SparseMatrix<float> H = hg.getH(paramIds);

                if (meshRegularization > 0.0)
                {
                    hg_regu = kscene.HGRegu(cam[lvl], frames.size());

                    Eigen::VectorXf G_regu = hg_regu.getG(paramIds);
                    Eigen::SparseMatrix<float> H_regu = hg_regu.getH(paramIds);

                    H += meshRegularization * H_regu;
                    G += meshRegularization * G_regu;
                }

                int n_try = 0;
                while (true)
                {
                    Eigen::SparseMatrix<float> H_lambda = H;

                    for (int j = 0; j < H_lambda.rows(); j++)
                    {
                        H_lambda.coeffRef(j, j) *= (1.0 + lambda);
                    }

                    bool solverSucceded = true;

                    H_lambda.makeCompressed();
                    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
                    Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
                    //   Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;

                    // Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
                    // Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;

                    // Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<float>, Eigen::Lower> solver;
                    // Eigen::SPQR<Eigen::SparseMatrix<float>> solver;

                    solver.compute(H_lambda);
                    // solver.analyzePattern(H_lambda);
                    // solver.factorize(H_lambda);
                    if (solver.info() != Eigen::Success)
                    {
                        solverSucceded = false;
                    }
                    // std::cout << solver.lastErrorMessage() << std::endl;
                    Eigen::VectorXf inc = solver.solve(G);
                    // Eigen::VectorXf inc = G / (1.0 + lambda);
                    //  inc_depth = -acc_H_depth_lambda.llt().solve(acc_J_depth);
                    //  inc_depth = - acc_H_depth_lambda.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(acc_J_depth);
                    //  inc_depth = -acc_H_depth_lambda.colPivHouseholderQr().solve(acc_J_depth);

                    if (solver.info() != Eigen::Success)
                    {
                        solverSucceded = false;
                    }

                    // update pose
                    std::vector<Sophus::SE3f> best_poses;
                    std::vector<vec2<float>> best_affines;
                    float pose_inc_mag = 0.0;
                    for (size_t i = 0; i < frames.size(); i++)
                    {
                        Eigen::Matrix<float, 8, 1> pose_inc;
                        // if ids are in order, like this I get the correct pose increment
                        // have to fix it some better way
                        for (int j = 0; j < 8; j++)
                        {
                            int paramId = kscene.getNumParams() + i * 8 + j;
                            int index = paramIds[paramId];
                            pose_inc(j) = inc(index);
                        }
                        pose_inc_mag += pose_inc.dot(pose_inc);
                        best_poses.push_back(frames[i].getPose());
                        best_affines.push_back(frames[i].getAffine());
                        Sophus::SE3f new_pose = frames[i].getPose() * Sophus::SE3f::exp(pose_inc.segment(0, 6)).inverse();
                        if (i == frames.size() - 1)
                            new_pose.translation() = frames[i].getPose().translation();
                        vec2<float> new_affine = frames[i].getAffine(); // - vec2<float>(pose_inc(6), pose_inc(7));
                        frames[i].setPose(new_pose);
                        frames[i].setAffine(new_affine);
                        // frames[i].pose = Sophus::SE3f::exp(pose_inc).inverse() * frames[i].pose;
                    }
                    pose_inc_mag /= frames.size();

                    // update map
                    std::map<unsigned int, float> best_params;
                    float map_inc_mag = 0.0;
                    int map_inc_mag_count = 0;
                    for (auto id : paramIds)
                    {
                        // negative ids are for the poses
                        if (id.first >= kscene.getNumParams())
                            continue;

                        float best_param = kscene.getParam(id.first);
                        float inc_param = inc(id.second);
                        float new_param = best_param - inc_param;
                        float weight = H.coeffRef(id.second, id.second);
                        // if(std::fabs(inc_param/best_param) > 0.4)
                        //     solverSucceded = false;
                        best_params[id.first] = best_param;
                        kscene.setParam(new_param, id.first);
                        kscene.setParamWeight(weight, id.first);
                        map_inc_mag += inc_param * inc_param;
                        map_inc_mag_count += 1;
                    }
                    map_inc_mag /= map_inc_mag_count;

                    e.setZero();
                    for (std::size_t i = 0; i < frames.size(); i++)
                    {
                        e += computeError(&frames[i], lvl);
                    }

                    float error = e.getError();

                    if (meshRegularization > 0.0)
                    {
                        e_regu = kscene.errorRegu(cam[lvl]);
                        error += meshRegularization * e_regu.getError();
                    }

                    std::cout << "new error " << error << " " << it << " " << lambda << " lvl: " << lvl << std::endl;

                    if (error < last_error && solverSucceded)
                    {
                        // accept update, decrease lambda
                        float p = error / last_error;

                        // if (lambda < 0.001f)
                        //     lambda = 0.0f;
                        // else
                        lambda *= 0.5;
                        // lambda = 0.0;

                        last_error = error;

                        if (p > 0.999f)
                        {
                            // std::cout << "lvl " << lvl << " converged after " << it << " itarations with lambda " << lambda << std::endl;
                            //  if converged, do next level
                            it = maxIterations;
                        }

                        // if update accepted, do next iteration
                        break;
                    }
                    else
                    {
                        for (size_t index = 0; index < frames.size(); index++)
                        {
                            frames[index].setPose(best_poses[index]);
                            frames[index].setAffine(best_affines[index]);
                        }

                        for (auto id : paramIds)
                        {
                            // negative ids are for the poses
                            if (id.first >= kscene.getNumParams())
                                continue;

                            kscene.setParam(best_params[id.first], id.first);
                        }

                        n_try++;

                        if (lambda == 0.0f)
                            lambda = 0.01f;
                        else
                            lambda *= 4.0; // std::pow(2.0, n_try);

                        // reject update, increase lambda, use un-updated data

                        // std::cout << "pose inc mag " << pose_inc_mag << " map inc mag " << map_inc_mag << std::endl;

                        if (pose_inc_mag < 1e-16 || map_inc_mag < 1e-16)
                        {
                            // if too small, do next level!
                            it = maxIterations;
                            break;
                        }
                    }
                }
            }
        }
    }

    void setMeshRegu(float mr)
    {
        meshRegularization = mr;
    }

    float meanViewAngle(frameCPU *frame1, frameCPU *frame2)
    {
        int lvl = 1;

        kscene.project(cam[lvl]);

        sceneType scene1 = kscene.clone();
        scene1.transform(frame1->getPose());
        scene1.project(cam[lvl]);

        sceneType scene2 = kscene.clone();
        scene2.transform(frame2->getPose());
        scene2.project(cam[lvl]);

        Sophus::SE3f frame1PoseInv = frame1->getPose().inverse();
        Sophus::SE3f frame2PoseInv = frame2->getPose().inverse();

        Eigen::Vector3f frame1Tra = frame1PoseInv.translation();
        Eigen::Vector3f frame2Tra = frame2PoseInv.translation();

        vec3<float> frame1Translation(frame1Tra(0), frame1Tra(1), frame1Tra(2));
        vec3<float> frame2Translation(frame2Tra(0), frame2Tra(1), frame2Tra(2));

        std::vector<int> sIds = kscene.getShapesIds();

        float accAngle = 0;
        int count = 0;
        for (auto sId : sIds)
        {
            shapeType shape = kscene.getShape(sId);
            vec2<float> centerPix = shape.getCenterPix();
            float centerDepth = shape.getDepth(centerPix);

            shapeType shape1 = scene1.getShape(sId);
            shapeType shape2 = scene2.getShape(sId);

            vec2<float> pix1 = shape1.getCenterPix();
            vec2<float> pix2 = shape2.getCenterPix();

            if (!cam[lvl].isPixVisible(pix1) || !cam[lvl].isPixVisible(pix2))
                continue;

            vec3<float> centerRay = cam[lvl].pixToRay(centerPix);
            vec3<float> centerPoint = centerRay * centerDepth;

            vec3<float> diff1 = frame1Translation - centerPoint;
            vec3<float> diff2 = frame2Translation - centerPoint;
            vec3<float> diff1Normalized = diff1 / diff1.norm();
            vec3<float> diff2Normalized = diff2 / diff2.norm();

            float cos_angle = diff1Normalized.dot(diff2Normalized);
            float angle = std::acos(cos_angle);

            accAngle += std::fabs(angle);
            count += 1;
        }

        return accAngle / count;
    }

    float getViewPercent(frameCPU &frame)
    {
        int lvl = 1;
        sceneType scene = kscene.clone();
        scene.transform(frame.getPose());
        scene.project(cam[lvl]);
        std::vector<int> shapeIds = scene.getShapesIds();

        int numVisible = 0;
        for (auto shapeId : shapeIds)
        {
            shapeType shape = scene.getShape(shapeId);
            vec2<float> pix = shape.getCenterPix();
            float depth = shape.getDepth(pix);
            if (depth <= 0.0)
                continue;
            if (cam[lvl].isPixVisible(pix))
                numVisible++;
        }

        return float(numVisible) / shapeIds.size();
    }

    float checkInfo(frameCPU &frame)
    {
        int lvl = 2;
        dataCPU<float> mask(cam[0].width, cam[0].height, -1);
        HGEigenSparse hg = computeHGMap2(&frame, &mask, lvl);
        std::map<int, int> ids = hg.getObservedParamIds();
        Eigen::SparseMatrix<float> H = hg.getH(ids);
        Eigen::VectorXf G = hg.getG(ids);

        // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > solver;
        // Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
        Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
        // Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;
        // Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;

        solver.compute(H);
        // solver.analyzePattern(H_lambda);
        // solver.factorize(H_lambda);

        if (solver.info() != Eigen::Success)
        {
            return 0.0;
        }
        // std::cout << solver.lastErrorMessage() << std::endl;
        Eigen::VectorXf inc = solver.solve(G);
        // inc_depth = -acc_H_depth_lambda.llt().solve(acc_J_depth);
        // inc_depth = - acc_H_depth_lambda.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(acc_J_depth);
        // inc_depth = -acc_H_depth_lambda.colPivHouseholderQr().solve(acc_J_depth);

        if (solver.info() != Eigen::Success)
        {
            return 0.0;
        }

        float relative_error = (H * inc - G).norm() / G.size(); // norm() is L2 norm

        return relative_error;
    }

    dataCPU<float> getIdepth(Sophus::SE3f pose, int lvl)
    {
        idepth_buffer.set(idepth_buffer.nodata, lvl);

        renderer.renderIdepthParallel(&kscene, pose, cam[lvl], &idepth_buffer, lvl);
        // renderer.renderInterpolate(cam[lvl], &idepth_buffer, lvl);
        return idepth_buffer;
    }

    /*
    dataCPU<float> getImage(frameCPU &frame, Sophus::SE3f &pose, int lvl)
    {
        image_buffer.set(image_buffer.nodata, lvl);
        renderer.renderImage(cam[lvl], frame, pose, image_buffer, lvl);
        return image_buffer;
    }
    */

    void plotDebug(frameCPU &frame, std::vector<frameCPU> frames = std::vector<frameCPU>())
    {
        idepth_buffer.set(idepth_buffer.nodata, 1);
        image_buffer.set(image_buffer.nodata, 1);
        error_buffer.set(error_buffer.nodata, 1);
        ivar_buffer.set(ivar_buffer.nodata, 1);
        debug.set(debug.nodata, 0);

        // renderer.renderIdepth(cam[1], frame.pose, idepth_buffer, 1);
        renderer.renderIdepthParallel(&kscene, frame.getPose(), cam[1], &idepth_buffer, 1);
        renderer.renderWeightParallel(&kscene, frame.getPose(), cam[1], &ivar_buffer, 1);
        renderer.renderResidualParallel(&kscene, &kimage, &frame, cam[1], &error_buffer, 1);

        show(frame.getRawImage(), "frame image", false, false, 1);
        show(kimage, "keyframe image", false, false, 1);
        show(error_buffer, "frame error", false, true, 1);
        show(idepth_buffer, "frame idepth", true, true, 1);
        show(ivar_buffer, "ivar", true, false, 1);

        show(frame.getdIdpixImage(), "frame dx image", false, true, 0, 1);
        show(jpose_buffer, "jpose image", false, true, 0, 1);
        show(jmap_buffer, "jmap image", false, true, 0, 1);

        if (frames.size() > 0)
        {
            dataCPU<float> frames_buffer(cam[0].width * frames.size(), cam[0].height, -1);
            dataCPU<float> residual_buffer(cam[0].width * frames.size(), cam[0].height, -1);
            for (int i = 0; i < frames.size(); i++)
            {
                for (int y = 0; y < cam[1].height; y++)
                {
                    for (int x = 0; x < cam[1].width; x++)
                    {
                        float pix_val = frames[i].getRawImage().get(y, x, 1);
                        float res_val = frames[i].getResidualImage().get(y, x, 1);
                        frames_buffer.set(pix_val, y, x + i * cam[1].width, 1);
                        residual_buffer.set(res_val, y, x + i * cam[1].width, 1);
                    }
                }
            }

            show(frames_buffer, "frames", false, false, 1);
            show(residual_buffer, "residuals", false, false, 1);
        }

        renderer.renderDebugParallel(&kscene, &kimage, Sophus::SE3f(), cam[0], &debug, 0);
        show(debug, "frame debug", false, false, 0);

        idepth_buffer.set(idepth_buffer.nodata, 1);
        // renderer.renderIdepth(cam[1], frame.pose, idepth_buffer, 1);
        renderer.renderIdepthParallel(&kscene, Sophus::SE3f(), cam[1], &idepth_buffer, 1);
        show(idepth_buffer, "keyframe idepth", true, false, 1);
    }

    void changeKeyframe(frameCPU &frame)
    {
        int lvl = 1;

        idepth_buffer.set(idepth_buffer.nodata, lvl);
        ivar_buffer.set(ivar_buffer.nodata, lvl);

        renderer.renderIdepthParallel(&kscene, frame.getPose(), cam[lvl], &idepth_buffer, lvl);
        renderer.renderWeightParallel(&kscene, frame.getPose(), cam[lvl], &ivar_buffer, lvl);
        // renderer.renderRandom(cam[lvl], &idepth, lvl);
        renderer.renderInterpolate(cam[lvl], &idepth_buffer, lvl);
        renderer.renderInterpolate(cam[lvl], &ivar_buffer, lvl);

        initKeyframe(frame, idepth_buffer, ivar_buffer, lvl);

        // kscene.transform(frame.pose);
        // kscene.project(cam[lvl]);
        // kscene.complete(frame, cam[lvl], idepth_buffer, lvl);
        // scene = kscene.clone();

        /*
        //method 2
        //build frame mesh
        //remove ocluded
        //devide big triangles
        //complete with random points
        MeshCPU frameMesh = buildFrameMesh(frame, lvl);
        keyframeMesh = frameMesh.getCopy();
        keyframe = frame;
        */
    }

    sceneType kscene;
    // frameCPU kframe;
    dataCPU<float> kimage;
    Sophus::SE3f kpose;
    vec2<float> kDepthAffine;

    std::vector<camera> cam;

private:
    Error computeError(frameCPU *frame, int lvl, bool useWeights=false)
    {
        // error_buffer.set(error_buffer.nodata, lvl);
        frame->getResidualImage().set(frame->getResidualImage().nodata, lvl);
        ivar_buffer.set(ivar_buffer.nodata, lvl);

        // renderer.renderResidualParallel(&kscene, &kframe, frame, cam[lvl], &error_buffer, lvl);
        renderer.renderResidualParallel(&kscene, &kimage, frame, cam[lvl], &(frame->getResidualImage()), lvl);
        if (useWeights)
            renderer.renderWeightParallel(&kscene, frame->getPose(), cam[lvl], &ivar_buffer, lvl);

        Error e = reducer.reduceErrorParallel(frame->getResidualImage(), ivar_buffer, lvl);

        return e;
    }

    HGEigenDense<2> computeHGLightAffine(frameCPU *frame, int lvl, bool useWeights)
    {
        idepth_buffer.set(idepth_buffer.nodata, lvl);
        jlightaffine_buffer.set(jlightaffine_buffer.nodata, lvl);
        error_buffer.set(error_buffer.nodata, lvl);
        ivar_buffer.set(ivar_buffer.nodata, lvl);

        renderer.renderJLightAffineParallel(&kscene, &kimage, frame, cam[lvl], &jlightaffine_buffer, &error_buffer, lvl);
        if (useWeights)
            renderer.renderWeightParallel(&kscene, frame->getPose(), cam[lvl], &ivar_buffer, lvl);

        HGEigenDense<2> hg = reducer.reduceHGLightAffineParallel(jlightaffine_buffer, error_buffer, ivar_buffer, lvl);

        return hg;
    }

    HGEigenDense<8> computeHGPose(frameCPU *frame, int lvl, bool useWeights)
    {
        idepth_buffer.set(idepth_buffer.nodata, lvl);
        jpose_buffer.set(jpose_buffer.nodata, lvl);
        error_buffer.set(error_buffer.nodata, lvl);
        ivar_buffer.set(ivar_buffer.nodata, lvl);

        renderer.renderJPoseParallel(&kscene, &kimage, frame, cam[lvl], &jpose_buffer, &error_buffer, lvl);
        if (useWeights)
            renderer.renderWeightParallel(&kscene, frame->getPose(), cam[lvl], &ivar_buffer, lvl);

        HGEigenDense<8> hg = reducer.reduceHGPoseParallel(jpose_buffer, error_buffer, ivar_buffer, lvl);

        return hg;
    }

    /*
    HGMapped computeHGMap(frameCPU *frame, int lvl)
    {
        error_buffer.set(error_buffer.nodata, lvl);
        jmap_buffer.set(jmap_buffer.nodata, lvl);
        pId_buffer.set(pId_buffer.nodata, lvl);

        // renderer.renderJMap(scene, cam[lvl], kframe, frame, jmap_buffer, error_buffer, pId_buffer, lvl);
        renderer.renderJMapParallel(&kscene, &kimage, frame, cam[lvl], &jmap_buffer, &error_buffer, &pId_buffer, lvl);
        // HGMapped hg = reducer.reduceHGMap(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);
        HGMapped hg = reducer.reduceHGMapParallel(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);

        return hg;
    }
    */

    HGEigenSparse computeHGMap2(frameCPU *frame, dataCPU<float> *mask, int lvl)
    {
        error_buffer.set(error_buffer.nodata, lvl);
        jmap_buffer.set(jmap_buffer.nodata, lvl);
        pId_buffer.set(pId_buffer.nodata, lvl);

        // renderer.renderJMap(scene, cam[lvl], kframe, frame, jmap_buffer, error_buffer, pId_buffer, lvl);
        renderer.renderJMapParallel(&kscene, &kimage, frame, cam[lvl], &jmap_buffer, &error_buffer, &pId_buffer, lvl);
        // HGMapped hg = reducer.reduceHGMap(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);
        // HGEigen hg = reducer.reduceHGMapParallel(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);
        HGEigenSparse hg = reducer.reduceHGMap2(kscene.getNumParams(), jmap_buffer, error_buffer, pId_buffer, *mask, lvl);

        return hg;
    }
    /*
    HGMapped computeHGPoseMap(frameCPU *frame, int frameIndex, int numFrames, int lvl)
    {
        jpose_buffer.set(jpose_buffer.nodata, lvl);
        jmap_buffer.set(jmap_buffer.nodata, lvl);
        error_buffer.set(error_buffer.nodata, lvl);
        pId_buffer.set(pId_buffer.nodata, lvl);

        renderer.renderJPoseMapParallel(&kscene, &kimage, frame, cam[lvl], &jpose_buffer, &jmap_buffer, &error_buffer, &pId_buffer, lvl);
        HGMapped hg = reducer.reduceHGPoseMapParallel(cam[lvl], frameIndex, numFrames, kscene.getNumParams(), jpose_buffer, jmap_buffer, error_buffer, pId_buffer, lvl);

        return hg;
    }
    */

    HGEigenSparse computeHGPoseMap2(frameCPU *frame, int frameIndex, int numFrames, int lvl)
    {
        jpose_buffer.set(jpose_buffer.nodata, lvl);
        jmap_buffer.set(jmap_buffer.nodata, lvl);
        error_buffer.set(error_buffer.nodata, lvl);
        pId_buffer.set(pId_buffer.nodata, lvl);

        renderer.renderJPoseMapParallel(&kscene, &kimage, frame, cam[lvl], &jpose_buffer, &jmap_buffer, &error_buffer, &pId_buffer, lvl);
        // HGEigenSparse hg = reducer.reduceHGPoseMap2(cam[lvl], frameIndex, numFrames, scene->getNumParams(), jpose_buffer, jmap_buffer, error_buffer, pId_buffer, lvl);
        HGEigenSparse hg = reducer.reduceHGPoseMapParallel2(frameIndex, numFrames, kscene.getNumParams(), jpose_buffer, jmap_buffer, error_buffer, pId_buffer, lvl);

        return hg;
    }

    // params
    bool multiThreading;
    float meshRegularization;
    float meshInitial;

    dataCPU<float> image_buffer;
    dataCPU<float> idepth_buffer;
    dataCPU<float> ivar_buffer;
    dataCPU<float> error_buffer;

    dataCPU<vec2<float>> jlightaffine_buffer;
    dataCPU<vec8<float>> jpose_buffer;

    dataCPU<jmapType> jmap_buffer;
    dataCPU<idsType> pId_buffer;

    // debug
    dataCPU<float> debug;
    dataCPU<float> idepthVar;

    renderCPU<sceneType, shapeType> renderer;
    reduceCPU reducer;
};
