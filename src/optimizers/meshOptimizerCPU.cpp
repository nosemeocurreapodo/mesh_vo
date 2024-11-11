#include "optimizers/meshOptimizerCPU.h"
#include <math.h>
#include "utils/tictoc.h"

meshOptimizerCPU::meshOptimizerCPU(camera &_cam)
    : kframe(_cam.width, _cam.height),
      image_buffer(_cam.width, _cam.height, -1.0),
      idepth_buffer(_cam.width, _cam.height, -1.0),
      ivar_buffer(_cam.width, _cam.height, -1.0),
      error_buffer(_cam.width, _cam.height, -1.0),
      jlightaffine_buffer(_cam.width, _cam.height, {0.0, 0.0}),
      jpose_buffer(_cam.width, _cam.height, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
      jmap_buffer(_cam.width, _cam.height, {0.0, 0.0, 0.0}),
      pId_buffer(_cam.width, _cam.height, {-1, -1, -1}),
      // jmap_buffer(_cam.width, _cam.height, 0.0),
      //  pId_buffer(_cam.width, _cam.height, -1),
      debug(_cam.width, _cam.height, -1.0),
      idepthVar(_cam.width, _cam.height, -1.0),
      renderer(_cam.width, _cam.height)
{
    cam[0] = _cam;
    for (int i = 1; i < MAX_LEVELS; i++)
    {
        cam[i] = _cam;
        cam[i].resize(1.0 / std::pow(2.0, i));
    }

    multiThreading = false;
    meshRegularization = 10.0;
    meshInitial = 0.0;
}

void meshOptimizerCPU::initKeyframe(frameCPU &frame, int lvl)
{
    idepth_buffer.set(idepth_buffer.nodata, lvl);
    // renderer.renderRandom(cam[lvl], &idepth_buffer, lvl);
    renderer.renderSmooth(cam[lvl], &idepth_buffer, lvl, 0.5, 1.5);
    ivar_buffer.set(ivar_buffer.nodata, lvl);
    renderer.renderSmooth(cam[lvl], &ivar_buffer, lvl, initialIvar(), initialIvar());
    kscene.init(frame, cam[lvl], idepth_buffer, ivar_buffer, lvl);
    kframe = frame;
}

void meshOptimizerCPU::initKeyframe(frameCPU &frame, dataCPU<float> &idepth, dataCPU<float> &ivar, int lvl)
{
    kscene.init(frame, cam[lvl], idepth, ivar, lvl);

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

    kframe = frame;
}

void meshOptimizerCPU::initKeyframe(frameCPU &frame, std::vector<vec2<float>> &texcoords, std::vector<float> &idepths, int lvl)
{
    kscene.init(frame, cam[lvl], texcoords, idepths, lvl);
    kframe = frame;
}

Error meshOptimizerCPU::computeError(frameCPU *frame, int lvl, bool useWeights)
{
    error_buffer.set(error_buffer.nodata, lvl);
    ivar_buffer.set(ivar_buffer.nodata, lvl);

    renderer.renderResidualParallel(&kscene, &kframe, frame, cam[lvl], &error_buffer, lvl);
    if (useWeights)
        renderer.renderWeightParallel(&kscene, frame->getPose(), cam[lvl], &ivar_buffer, lvl);

    Error e = reducer.reduceErrorParallel(cam[lvl], error_buffer, ivar_buffer, lvl);

    return e;
}

/*
Error meshOptimizerCPU::computeError(dataCPU<float> &fIdepth, frameCPU &kframe, frameCPU &frame, int lvl)
{
    image_buffer.set(image_buffer.nodata, lvl);

    renderer.renderImage(fIdepth, cam[lvl], kframe, frame.pose, image_buffer, lvl);
    Error e = reducer.reduceError(cam[lvl], image_buffer, frame.image, lvl);

    return e;
}
*/

HGEigenDense<2> meshOptimizerCPU::computeHGLightAffine(frameCPU *frame, int lvl, bool useWeights)
{
    idepth_buffer.set(idepth_buffer.nodata, lvl);
    jlightaffine_buffer.set(jlightaffine_buffer.nodata, lvl);
    error_buffer.set(error_buffer.nodata, lvl);
    ivar_buffer.set(ivar_buffer.nodata, lvl);

    renderer.renderJLightAffineParallel(&kscene, &kframe, frame, cam[lvl], &jlightaffine_buffer, &error_buffer, lvl);
    if (useWeights)
        renderer.renderWeightParallel(&kscene, frame->getPose(), cam[lvl], &ivar_buffer, lvl);

    HGEigenDense<2> hg = reducer.reduceHGLightAffineParallel(cam[lvl], jlightaffine_buffer, error_buffer, ivar_buffer, lvl);

    return hg;
}

HGEigenDense<8> meshOptimizerCPU::computeHGPose(frameCPU *frame, int lvl, bool useWeights)
{
    idepth_buffer.set(idepth_buffer.nodata, lvl);
    jpose_buffer.set(jpose_buffer.nodata, lvl);
    error_buffer.set(error_buffer.nodata, lvl);
    ivar_buffer.set(ivar_buffer.nodata, lvl);

    renderer.renderJPoseParallel(&kscene, &kframe, frame, cam[lvl], &jpose_buffer, &error_buffer, lvl);
    if (useWeights)
        renderer.renderWeightParallel(&kscene, frame->getPose(), cam[lvl], &ivar_buffer, lvl);

    HGEigenDense<8> hg = reducer.reduceHGPoseParallel(cam[lvl], jpose_buffer, error_buffer, ivar_buffer, lvl);

    return hg;
}

/*
HGPose meshOptimizerCPU::computeHGPose(dataCPU<float> &fIdepth, frameCPU &kframe, frameCPU &frame, int lvl)
{
    error_buffer.set(error_buffer.nodata, lvl);
    jpose_buffer.set(jpose_buffer.nodata, lvl);

    //renderer.renderJPose(fIdepth, cam[lvl], kframe, frame, jpose_buffer, error_buffer, lvl);
    renderer.renderJPoseParallel(fIdepth, cam[lvl], kframe, frame, jpose_buffer, error_buffer, lvl);
    //HGPose hg =reducer.reduceHGPose(cam[lvl], jpose_buffer, error_buffer, lvl);
    HGPose hg =reducer.reduceHGPoseParallel(cam[lvl], jpose_buffer, error_buffer, lvl);
    return hg;
}
*/

HGMapped meshOptimizerCPU::computeHGMap(frameCPU *frame, int lvl)
{
    error_buffer.set(error_buffer.nodata, lvl);
    jmap_buffer.set(jmap_buffer.nodata, lvl);
    pId_buffer.set(pId_buffer.nodata, lvl);

    // renderer.renderJMap(scene, cam[lvl], kframe, frame, jmap_buffer, error_buffer, pId_buffer, lvl);
    renderer.renderJMapParallel(&kscene, &kframe, frame, cam[lvl], &jmap_buffer, &error_buffer, &pId_buffer, lvl);
    // HGMapped hg = reducer.reduceHGMap(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);
    HGMapped hg = reducer.reduceHGMapParallel(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);

    return hg;
}

HGEigenSparse meshOptimizerCPU::computeHGMap2(frameCPU *frame, dataCPU<float> *mask, int lvl)
{
    error_buffer.set(error_buffer.nodata, lvl);
    jmap_buffer.set(jmap_buffer.nodata, lvl);
    pId_buffer.set(pId_buffer.nodata, lvl);

    // renderer.renderJMap(scene, cam[lvl], kframe, frame, jmap_buffer, error_buffer, pId_buffer, lvl);
    renderer.renderJMapParallel(&kscene, &kframe, frame, cam[lvl], &jmap_buffer, &error_buffer, &pId_buffer, lvl);
    // HGMapped hg = reducer.reduceHGMap(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);
    // HGEigen hg = reducer.reduceHGMapParallel(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);
    HGEigenSparse hg = reducer.reduceHGMap2(cam[lvl], kscene.getNumParams(), jmap_buffer, error_buffer, pId_buffer, *mask, lvl);

    return hg;
}

HGMapped meshOptimizerCPU::computeHGPoseMap(frameCPU *frame, int frameIndex, int numFrames, int lvl)
{
    jpose_buffer.set(jpose_buffer.nodata, lvl);
    jmap_buffer.set(jmap_buffer.nodata, lvl);
    error_buffer.set(error_buffer.nodata, lvl);
    pId_buffer.set(pId_buffer.nodata, lvl);

    renderer.renderJPoseMapParallel(&kscene, &kframe, frame, cam[lvl], &jpose_buffer, &jmap_buffer, &error_buffer, &pId_buffer, lvl);
    HGMapped hg = reducer.reduceHGPoseMapParallel(cam[lvl], frameIndex, numFrames, kscene.getNumParams(), jpose_buffer, jmap_buffer, error_buffer, pId_buffer, lvl);

    return hg;
}

HGEigenSparse meshOptimizerCPU::computeHGPoseMap2(frameCPU *frame, int frameIndex, int numFrames, int lvl)
{
    jpose_buffer.set(jpose_buffer.nodata, lvl);
    jmap_buffer.set(jmap_buffer.nodata, lvl);
    error_buffer.set(error_buffer.nodata, lvl);
    pId_buffer.set(pId_buffer.nodata, lvl);

    renderer.renderJPoseMapParallel(&kscene, &kframe, frame, cam[lvl], &jpose_buffer, &jmap_buffer, &error_buffer, &pId_buffer, lvl);
    // HGEigenSparse hg = reducer.reduceHGPoseMap2(cam[lvl], frameIndex, numFrames, scene->getNumParams(), jpose_buffer, jmap_buffer, error_buffer, pId_buffer, lvl);
    HGEigenSparse hg = reducer.reduceHGPoseMapParallel2(cam[lvl], frameIndex, numFrames, kscene.getNumParams(), jpose_buffer, jmap_buffer, error_buffer, pId_buffer, lvl);

    return hg;
}

void meshOptimizerCPU::optLightAffine(frameCPU &frame)
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

void meshOptimizerCPU::optPose(frameCPU &frame)
{
    int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

    for (int lvl = 3; lvl >= 1; lvl--)
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
                vec2<float> new_affine = best_affine - vec2<float>(inc(6), inc(7));
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

void meshOptimizerCPU::optMap(std::vector<frameCPU> &frames, dataCPU<float> &mask)
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

    for (int lvl = 1; lvl >= 1; lvl--)
    {
        kscene.project(cam[lvl]);
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
                Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
                // Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
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

                    //if (std::fabs(inc_param / best_param) > 0.1)
                    //if(new_param <= 0.0)
                    if(false)
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
                    map_inc_mag += inc(id.second) * inc(id.second);
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

                    if (map_inc_mag < 1e-8)
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

void meshOptimizerCPU::optPoseMap(std::vector<frameCPU> &frames)
{
    Error e;
    Error e_regu;
    HGEigenSparse hg(kscene.getNumParams() + frames.size() * 8);
    HGEigenSparse hg_regu(kscene.getNumParams() + frames.size() * 8);

    for (int lvl = 3; lvl >= 1; lvl--)
    {
        kscene.project(cam[lvl]);
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
            //std::map<int, int> paramIds = hg.getObservedParamIds();
            std::map<int, int> paramIds = hg.getParamIds();

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
                Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
                // Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
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
                    vec2<float> new_affine = frames[i].getAffine() - vec2<float>(pose_inc(6), pose_inc(7));
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
                    //if(std::fabs(inc_param/best_param) > 0.4)
                    //    solverSucceded = false;
                    best_params[id.first] = best_param;
                    kscene.setParam(new_param, id.first);
                    kscene.setParamWeight(weight, id.first);
                    map_inc_mag += inc(id.second) * inc(id.second);
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

                    //std::cout << "pose inc mag " << pose_inc_mag << " map inc mag " << map_inc_mag << std::endl;

                    if (pose_inc_mag < 1e-16 || map_inc_mag < 1e-8)
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
