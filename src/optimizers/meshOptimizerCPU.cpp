#include "optimizers/meshOptimizerCPU.h"
#include <math.h>
#include "utils/tictoc.h"

meshOptimizerCPU::meshOptimizerCPU(camera &_cam)
    : kframe(_cam.width, _cam.height),
      image_buffer(_cam.width, _cam.height, -1.0),
      idepth_buffer(_cam.width, _cam.height, -1.0),
      error_buffer(_cam.width, _cam.height, -1.0),
      jpose_buffer(_cam.width, _cam.height, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
      jmap_buffer(_cam.width, _cam.height, {0.0, 0.0, 0.0}),
      pId_buffer(_cam.width, _cam.height, {-1, -1, -1}),
      // jmap_buffer(_cam.width, _cam.height, 0.0),
      // pId_buffer(_cam.width, _cam.height, -1),
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
    renderer.renderSmooth(cam[lvl], &idepth_buffer, lvl);
    kscene.init(frame, cam[lvl], idepth_buffer, lvl);
    kframe = frame;
    scene = kscene.clone();
}

void meshOptimizerCPU::initKeyframe(frameCPU &frame, dataCPU<float> &idepth, int lvl)
{
    kscene.init(frame, cam[lvl], idepth, lvl);
    kframe = frame;
    scene = kscene.clone();
}

Error meshOptimizerCPU::computeError(SceneBase *scene, frameCPU *frame, int lvl)
{
    image_buffer.set(image_buffer.nodata, lvl);
    idepth_buffer.set(idepth_buffer.nodata, lvl);

    renderer.renderImageParallel(&kscene, &kframe, scene, cam[lvl], &image_buffer, lvl);
    // renderer.renderIdepth(scene, cam[lvl], frame.pose, idepth_buffer, lvl);

    // renderer.renderIdepthParallel(scene, cam[lvl], frame.pose, idepth_buffer, lvl);
    // renderer.renderImageParallel(idepth_buffer, cam[lvl], kframe, frame.pose, image_buffer, lvl);
    Error e = reducer.reduceErrorParallel(cam[lvl], image_buffer, frame->image, lvl);

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

HGEigenDense meshOptimizerCPU::computeHGPose(SceneBase *scene, frameCPU *frame, int lvl)
{
    idepth_buffer.set(idepth_buffer.nodata, lvl);
    jpose_buffer.set(jpose_buffer.nodata, lvl);
    error_buffer.set(error_buffer.nodata, lvl);

    renderer.renderJPoseParallel(&kscene, &kframe, scene, frame, cam[lvl], &jpose_buffer, &error_buffer, lvl);
    // renderer.renderIdepth(scene, cam[lvl], frame.pose, idepth_buffer, lvl);
    // renderer.renderIdepthParallel(scene, cam[lvl], frame.pose, idepth_buffer, lvl);
    // renderer.renderJPose(idepth_buffer, cam[lvl], kframe, frame, jpose_buffer, error_buffer, lvl);
    //  renderer.renderJPoseParallel(idepth_buffer, cam[lvl], kframe, frame, j1_buffer, j2_buffer, error_buffer, lvl);

    HGEigenDense hg = reducer.reduceHGPoseParallel(cam[lvl], jpose_buffer, error_buffer, lvl);

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

HGMapped meshOptimizerCPU::computeHGMap(SceneBase *scene, frameCPU *frame, int lvl)
{
    error_buffer.set(error_buffer.nodata, lvl);
    jmap_buffer.set(jmap_buffer.nodata, lvl);
    pId_buffer.set(pId_buffer.nodata, lvl);

    // renderer.renderJMap(scene, cam[lvl], kframe, frame, jmap_buffer, error_buffer, pId_buffer, lvl);
    renderer.renderJMapParallel(&kscene, &kframe, scene, frame, cam[lvl], &jmap_buffer, &error_buffer, &pId_buffer, lvl);
    // HGMapped hg = reducer.reduceHGMap(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);
    HGMapped hg = reducer.reduceHGMapParallel(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);

    return hg;
}

HGEigenSparse meshOptimizerCPU::computeHGMap2(SceneBase *scene, frameCPU *frame, int lvl)
{
    error_buffer.set(error_buffer.nodata, lvl);
    jmap_buffer.set(jmap_buffer.nodata, lvl);
    pId_buffer.set(pId_buffer.nodata, lvl);

    // renderer.renderJMap(scene, cam[lvl], kframe, frame, jmap_buffer, error_buffer, pId_buffer, lvl);
    renderer.renderJMapParallel(&kscene, &kframe, scene, frame, cam[lvl], &jmap_buffer, &error_buffer, &pId_buffer, lvl);
    // HGMapped hg = reducer.reduceHGMap(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);
    // HGEigen hg = reducer.reduceHGMapParallel(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);
    HGEigenSparse hg = reducer.reduceHGMap2(cam[lvl], kscene.getNumParams(), jmap_buffer, error_buffer, pId_buffer, lvl);

    return hg;
}

HGMapped meshOptimizerCPU::computeHGPoseMap(SceneBase *scene, frameCPU *frame, int frameIndex, int numFrames, int lvl)
{
    jpose_buffer.set(jpose_buffer.nodata, lvl);
    jmap_buffer.set(jmap_buffer.nodata, lvl);
    error_buffer.set(error_buffer.nodata, lvl);
    pId_buffer.set(pId_buffer.nodata, lvl);

    renderer.renderJPoseMapParallel(&kscene, &kframe, scene, frame, cam[lvl], &jpose_buffer, &jmap_buffer, &error_buffer, &pId_buffer, lvl);
    HGMapped hg = reducer.reduceHGPoseMapParallel(cam[lvl], frameIndex, numFrames, kscene.getNumParams(), jpose_buffer, jmap_buffer, error_buffer, pId_buffer, lvl);

    return hg;
}

HGEigenSparse meshOptimizerCPU::computeHGPoseMap2(SceneBase *scene, frameCPU *frame, int frameIndex, int numFrames, int lvl)
{
    jpose_buffer.set(jpose_buffer.nodata, lvl);
    jmap_buffer.set(jmap_buffer.nodata, lvl);
    error_buffer.set(error_buffer.nodata, lvl);
    pId_buffer.set(pId_buffer.nodata, lvl);

    renderer.renderJPoseMapParallel(&kscene, &kframe, scene, frame, cam[lvl], &jpose_buffer, &jmap_buffer, &error_buffer, &pId_buffer, lvl);
    // HGEigenSparse hg = reducer.reduceHGPoseMap2(cam[lvl], frameIndex, numFrames, scene->getNumParams(), jpose_buffer, jmap_buffer, error_buffer, pId_buffer, lvl);
    HGEigenSparse hg = reducer.reduceHGPoseMapParallel2(cam[lvl], frameIndex, numFrames, scene->getNumParams(), jpose_buffer, jmap_buffer, error_buffer, pId_buffer, lvl);

    return hg;
}

void meshOptimizerCPU::optPose(frameCPU &frame)
{
    int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

    scene->transform(frame.pose);

    for (int lvl = 3; lvl >= 1; lvl--)
    {
        scene->project(cam[lvl]);
        kscene.project(cam[lvl]);
        // std::cout << "*************************lvl " << lvl << std::endl;
        Sophus::SE3f best_pose = frame.pose;
        // Error e = computeError(idepth_buffer, keyframe, frame, lvl);
        Error e = computeError(scene.get(), &frame, lvl);
        float last_error = e.getError();

        std::cout << "initial error " << last_error << " " << lvl << std::endl;

        for (int it = 0; it < maxIterations[lvl]; it++)
        {
            // HGPose hg = computeHGPose(idepth_buffer, keyframe, frame, lvl);
            HGEigenDense hg = computeHGPose(scene.get(), &frame, lvl);

            // std::vector<int> pIds = hg.G.getParamIds();
            // Eigen::VectorXf G = hg.G.toEigen(pIds);
            // Eigen::SparseMatrix<float> H = hg.H.toEigen(pIds);

            Eigen::VectorXf G = hg.getG();
            Eigen::Matrix<float, 6, 6> H = hg.getH();

            float lambda = 0.0;
            int n_try = 0;
            while (true)
            {
                Eigen::Matrix<float, 6, 6> H_lambda;
                H_lambda = H;

                for (int j = 0; j < 6; j++)
                    H_lambda(j, j) *= 1.0 + lambda;

                // Eigen::Matrix<float, 6, 1> inc = H_lambda.ldlt().solve(G);
                Sophus::Vector6f inc = H_lambda.ldlt().solve(G);

                // Sophus::SE3f new_pose = frame.pose * Sophus::SE3f::exp(inc_pose);
                frame.pose = best_pose * Sophus::SE3f::exp(inc).inverse();
                // Sophus::SE3f new_pose = Sophus::SE3f::exp(inc_pose).inverse() * frame.pose;
                scene->transform(frame.pose);
                scene->project(cam[lvl]);

                // e.setZero();
                // e = computeError(idepth_buffer, keyframe, frame, lvl);
                e = computeError(scene.get(), &frame, lvl);
                float error = e.getError();
                // std::cout << "new error " << error << " time " << t.toc() << std::endl;

                std::cout << "new error " << error << " " << lambda << " " << it << " " << lvl << std::endl;

                if (error < last_error)
                {
                    // accept update, decrease lambda

                    best_pose = frame.pose;
                    float p = error / last_error;

                    // if (lambda < 0.2f)
                    //     lambda = 0.0f;
                    // else
                    //lambda *= 0.5;
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
                    frame.pose = best_pose;
                    scene->transform(frame.pose);
                    scene->project(cam[lvl]);

                    n_try++;

                    if (lambda == 0.0)
                        lambda = 0.2f;
                    else
                        lambda *= 2.0;//std::pow(2.0, n_try);

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

void meshOptimizerCPU::optMap(std::vector<frameCPU> &frames)
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
            scene->transform(frames[i].pose);
            scene->project(cam[lvl]);
            e += computeError(scene.get(), &frames[i], lvl);
        }

        float last_error = e.getError();

        if (meshRegularization > 0.0)
        {
            e_regu = kscene.errorRegu(cam[lvl]);
            last_error += meshRegularization * e_regu.getError();
        }

        // e_init = keyframeScene.errorInitial(initialScene, initialInvVar);
        //  e_init.error /= e_init.count;

        std::cout << "initial error " << last_error << " " << lvl << std::endl;

        int maxIterations = 100;
        float lambda = 0.0;
        for (int it = 0; it < maxIterations; it++)
        {
            hg.setZero();
            for (std::size_t i = 0; i < frames.size(); i++)
            {
                scene->transform(frames[i].pose);
                scene->project(cam[lvl]);
                HGEigenSparse _hg = computeHGMap2(scene.get(), &frames[i], lvl);
                hg += _hg;
            }

            // saveH(hg, "H.png");

            // std::map<int, int> paramIds = hg.getObservedParamIds();
            std::map<int, int> paramIds = hg.getParamIds();

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

                H_lambda.makeCompressed();
                // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > solver;
                Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
                // Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
                //  Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;
                //  Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;

                solver.compute(H_lambda);
                // solver.analyzePattern(H_lambda);
                // solver.factorize(H_lambda);

                if (solver.info() != Eigen::Success)
                {
                    // some problem i have still to debug
                    it = maxIterations;
                    std::cout << "solver.compute not successfull " << solver.info() << std::endl;
                    break;
                }
                // std::cout << solver.lastErrorMessage() << std::endl;

                Eigen::VectorXf inc = solver.solve(G);
                // Eigen::VectorXf inc = G / (1.0 + lambda);
                //  inc_depth = -acc_H_depth_lambda.llt().solve(acc_J_depth);
                //  inc_depth = - acc_H_depth_lambda.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(acc_J_depth);
                //  inc_depth = -acc_H_depth_lambda.colPivHouseholderQr().solve(acc_J_depth);

                if (solver.info() != Eigen::Success)
                {
                    // solving failed
                    it = maxIterations;
                    std::cout << "solver.solve not successfull " << solver.info() << std::endl;
                    break;
                }

                std::vector<float> best_params;
                for (auto id : paramIds)
                {
                    float best_param = kscene.getParam(id.first);
                    float new_param = best_param - inc(id.second);
                    best_params.push_back(best_param);
                    // the derivative is with respecto to the keyframe pose
                    // the update should take this into account
                    kscene.setParam(new_param, id.first);
                }
                scene = kscene.clone();

                e.setZero();
                for (std::size_t i = 0; i < frames.size(); i++)
                {
                    scene->transform(frames[i].pose);
                    scene->project(cam[lvl]);
                    e += computeError(scene.get(), &frames[i], lvl);
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

                if (error < last_error)
                {
                    // accept update, decrease lambda
                    float p = error / last_error;

                    //if (lambda < 0.2f)
                    //    lambda = 0.0f;
                    //else
                    //    lambda *= 0.5;
                    lambda = 0.0;

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
                    scene = kscene.clone();

                    n_try++;

                    if (lambda == 0.0f)
                        lambda = 0.2f;
                    else
                        lambda *= 2.0;//std::pow(2.0, n_try);

                    // reject update, increase lambda, use un-updated data

                    if (inc.dot(inc) < 1e-16)
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
    HGEigenSparse hg(kscene.getNumParams() + frames.size() * 6);
    HGEigenSparse hg_regu(kscene.getNumParams() + frames.size() * 6);

    for (int lvl = 1; lvl >= 1; lvl--)
    {
        kscene.project(cam[lvl]);
        e.setZero();
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            scene->transform(frames[i].pose);
            scene->project(cam[lvl]);
            e += computeError(scene.get(), &frames[i], lvl);
        }

        float last_error = e.getError();

        if (meshRegularization > 0.0)
        {
            e_regu = kscene.errorRegu(cam[lvl]);
            last_error += meshRegularization * e_regu.getError();
        }

        std::cout << "initial error " << last_error << std::endl;

        int maxIterations = 100;
        float lambda = 0.0;
        for (int it = 0; it < maxIterations; it++)
        {
            hg.setZero();
            for (std::size_t i = 0; i < frames.size(); i++)
            {
                scene->transform(frames[i].pose);
                scene->project(cam[lvl]);
                HGEigenSparse _hg = computeHGPoseMap2(scene.get(), &frames[i], i, frames.size(), lvl);
                hg += _hg;
            }

            // saveH(hg, "H.png");

            // map from param id to param index paramIndex = obsParamIds[paramId]
            std::map<int, int> paramIds = hg.getObservedParamIds();

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

                H_lambda.makeCompressed();
                // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > solver;
                Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
                // Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
                //  Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;
                //  Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;

                solver.compute(H_lambda);
                // solver.analyzePattern(H_lambda);
                // solver.factorize(H_lambda);
                if (solver.info() != Eigen::Success)
                {
                    // some problem i have still to debug
                    it = maxIterations;
                    std::cout << "solver.compute not successfull " << solver.info() << std::endl;
                    break;
                }
                // std::cout << solver.lastErrorMessage() << std::endl;
                Eigen::VectorXf inc = solver.solve(G);
                // inc_depth = -acc_H_depth_lambda.llt().solve(acc_J_depth);
                // inc_depth = - acc_H_depth_lambda.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(acc_J_depth);
                // inc_depth = -acc_H_depth_lambda.colPivHouseholderQr().solve(acc_J_depth);

                if (solver.info() != Eigen::Success)
                {
                    // solving failed
                    it = maxIterations;
                    std::cout << "solver.compute not successfull " << solver.info() << std::endl;
                    break;
                }

                // update pose
                std::vector<Sophus::SE3f> best_poses;
                for (size_t i = 0; i < frames.size(); i++)
                {
                    Eigen::Matrix<float, 6, 1> pose_inc;
                    // if ids are in order, like this I get the correct pose increment
                    // have to fix it some better way
                    for (int j = 0; j < 6; j++)
                    {
                        int paramId = kscene.getNumParams() + i * 6 + j;
                        int index = paramIds[paramId];
                        pose_inc(j) = inc(index);
                    }
                    best_poses.push_back(frames[i].pose);
                    frames[i].pose = frames[i].pose * Sophus::SE3f::exp(pose_inc).inverse();
                    // frames[i].pose = Sophus::SE3f::exp(pose_inc).inverse() * frames[i].pose;
                }

                // update map
                std::map<unsigned int, float> best_params;
                for (auto id : paramIds)
                {
                    // negative ids are for the poses
                    if (id.first >= kscene.getNumParams())
                        continue;

                    float best_param = kscene.getParam(id.first);
                    float new_param = best_param - inc(id.second);
                    best_params[id.first] = best_param;
                    kscene.setParam(new_param, id.first);
                }
                scene = kscene.clone();

                e.setZero();
                for (std::size_t i = 0; i < frames.size(); i++)
                {
                    scene->transform(frames[i].pose);
                    scene->project(cam[lvl]);
                    e += computeError(scene.get(), &frames[i], lvl);
                }

                float error = e.getError();

                if (meshRegularization > 0.0)
                {
                    e_regu = kscene.errorRegu(cam[lvl]);
                    error += meshRegularization * e_regu.getError();
                }

                std::cout << "new error " << error << " " << it << " " << lambda << std::endl;

                if (error < last_error)
                {
                    // accept update, decrease lambda
                    float p = error / last_error;

                    // if (lambda < 0.001f)
                    //     lambda = 0.0f;
                    // else
                    //lambda *= 0.5;
                    lambda = 0.0;

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
                        frames[index].pose = best_poses[index];

                    for (auto id : paramIds)
                    {
                        // negative ids are for the poses
                        if (id.first >= kscene.getNumParams())
                            continue;

                        kscene.setParam(best_params[id.first], id.first);
                    }
                    scene = kscene.clone();

                    n_try++;

                    if (lambda == 0.0f)
                        lambda = 0.2f;
                    else
                        lambda *= 2.0;//std::pow(2.0, n_try);

                    // reject update, increase lambda, use un-updated data

                    if (inc.dot(inc) < 1e-8)
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
