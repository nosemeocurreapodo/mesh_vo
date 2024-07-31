#include "optimizers/meshOptimizerCPU.h"
#include <math.h>
#include "utils/tictoc.h"

meshOptimizerCPU::meshOptimizerCPU(camera &_cam)
    : image_buffer(_cam.width, _cam.height, -1.0),
      idepth_buffer(_cam.width, _cam.height, -1.0),
      error_buffer(_cam.width, _cam.height, -1.0),
      jpose_buffer(_cam.width, _cam.height, {0.0, 0.0, 0.0}),
      jmap_buffer(_cam.width, _cam.height, {0.0, 0.0, 0.0}),
      pId_buffer(_cam.width, _cam.height, {-1, -1, -1}),
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
    meshRegularization = 200.0;
    meshInitial = 0.0;
}

void meshOptimizerCPU::initKeyframe(frameCPU &frame, dataCPU<float> &idepth, dataCPU<float> &idepthVar, int lvl)
{
    sceneOptimized.init(frame, cam[lvl], idepth, lvl);
}

Error meshOptimizerCPU::computeError(frameCPU &kframe, frameCPU &frame, int lvl)
{
    image_buffer.set(image_buffer.nodata, lvl);
    idepth_buffer.set(idepth_buffer.nodata, lvl);

    renderer.renderImageParallel(cam[lvl], kframe, frame.pose, image_buffer, lvl);
    // renderer.renderIdepth(scene, cam[lvl], frame.pose, idepth_buffer, lvl);

    //renderer.renderIdepthParallel(scene, cam[lvl], frame.pose, idepth_buffer, lvl);
    //renderer.renderImageParallel(idepth_buffer, cam[lvl], kframe, frame.pose, image_buffer, lvl);
    Error e = reducer.reduceErrorParallel(cam[lvl], image_buffer, frame.image, lvl);

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

HGPose meshOptimizerCPU::computeHGPose(frameCPU &kframe, frameCPU &frame, int lvl)
{
    idepth_buffer.set(idepth_buffer.nodata, lvl);
    jpose_buffer.set(jpose_buffer.nodata, lvl);
    error_buffer.set(error_buffer.nodata, lvl);

    renderer.renderJPoseParallel(cam[lvl], kframe, frame, jpose_buffer, error_buffer, lvl);
    //renderer.renderIdepth(scene, cam[lvl], frame.pose, idepth_buffer, lvl);
    //renderer.renderIdepthParallel(scene, cam[lvl], frame.pose, idepth_buffer, lvl);
    //renderer.renderJPose(idepth_buffer, cam[lvl], kframe, frame, jpose_buffer, error_buffer, lvl);
    // renderer.renderJPoseParallel(idepth_buffer, cam[lvl], kframe, frame, j1_buffer, j2_buffer, error_buffer, lvl);

    HGPose hg = reducer.reduceHGPoseParallel(cam[lvl], jpose_buffer, error_buffer, lvl);

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

HGMapped meshOptimizerCPU::computeHGMap(frameCPU &kframe, frameCPU &frame, int lvl)
{
    error_buffer.set(error_buffer.nodata, lvl);
    jmap_buffer.set(jmap_buffer.nodata, lvl);
    pId_buffer.set(pId_buffer.nodata, lvl);

    //renderer.renderJMap<MESH_DOF>(scene, cam[lvl], kframe, frame, jmap_buffer, error_buffer, pId_buffer, lvl);
    renderer.renderJMapParallel<MESH_DOF>(cam[lvl], kframe, frame, jmap_buffer, error_buffer, pId_buffer, lvl);
    //HGMapped hg = reducer.reduceHGMap<MESH_DOF>(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);
    HGMapped hg = reducer.reduceHGMapParallel<MESH_DOF>(cam[lvl], jmap_buffer, error_buffer, pId_buffer, lvl);

    return hg;
}

HGMapped meshOptimizerCPU::computeHGPoseMap(frameCPU &kframe, frameCPU &frame, int lvl)
{
    jpose_buffer.set(jpose_buffer.nodata, lvl);
    jmap_buffer.set(jmap_buffer.nodata, lvl);
    error_buffer.set(error_buffer.nodata, lvl);
    pId_buffer.set(pId_buffer.nodata, lvl);

    renderer.renderJPoseMap<MESH_DOF>(cam[lvl], kframe, frame, jpose_buffer, jmap_buffer, error_buffer, pId_buffer, lvl);
    HGMapped hg = reducer.reduceHGPoseMap<MESH_DOF>(cam[lvl], frame.id, jpose_buffer, jmap_buffer, error_buffer, pId_buffer, lvl);

    return hg;
}

void meshOptimizerCPU::optPose(frameCPU &keyframe, frameCPU &frame)
{
    int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

    tic_toc t;

    // std::unique_ptr<SceneBase> keyframeScene = sceneOptimized.clone();
    // keyframeScene->transform(keyframe.pose);
    // sceneOptimized.transform(keyframe.pose);

    renderer.setScene(sceneOptimized);

    for (int lvl = 3; lvl >= 1; lvl--)
    {
        std::cout << "*************************lvl " << lvl << std::endl;
        t.tic();
        Sophus::SE3f best_pose = frame.pose;
        //Error e = computeError(idepth_buffer, keyframe, frame, lvl);
        Error e = computeError(keyframe, frame, lvl);
        float last_error = e.error / e.count;

        std::cout << "init error " << last_error << " time " << t.toc() << std::endl;

        for (int it = 0; it < maxIterations[lvl]; it++)
        {
            t.tic();
            //HGPose hg = computeHGPose(idepth_buffer, keyframe, frame, lvl);
            HGPose  hg = computeHGPose(keyframe, frame, lvl);

            // std::vector<int> pIds = hg.G.getParamIds();

            // Eigen::VectorXf G = hg.G.toEigen(pIds);
            // Eigen::SparseMatrix<float> H = hg.H.toEigen(pIds);

            Eigen::VectorXf G = hg.G;
            Eigen::Matrix<float, 6, 6> H = hg.H;

            H /= hg.count;
            G /= hg.count;

            std::cout << "HGPose time " << t.toc() << std::endl;

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

                t.tic();
                e.setZero();
                //e = computeError(idepth_buffer, keyframe, frame, lvl);
                e = computeError(keyframe, frame, lvl);
                float error = e.error / e.count;
                std::cout << "new error " << error << " time " << t.toc() << std::endl;

                if (error < last_error)
                {
                    // accept update, decrease lambda

                    best_pose = frame.pose;
                    float p = error / last_error;

                    if (lambda < 0.2f)
                        lambda = 0.0f;
                    else
                        lambda *= 0.5;

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

                    n_try++;

                    if (lambda < 0.2f)
                        lambda = 0.2f;
                    else
                        lambda *= std::pow(2.0, n_try);

                    // reject update, increase lambda, use un-updated data
                    // std::cout << "update rejected " << std::endl;

                    if (!(inc.dot(inc) > 1e-8))
                    // if(!(inc.dot(inc) > 1e-6))
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

void meshOptimizerCPU::optMap(frameCPU &keyframe, std::vector<frameCPU> &frames)
{
    tic_toc t;

    Error e;
    Error e_regu;
    Error e_init;

    HGMapped hg;
    HGMapped hg_regu;
    HGMapped hg_init;

    // the update will be done with respecto to the keyframe
    // so change the scene accordingly, then set it back to the original
    // pose (or maybe not)
    // std::unique_ptr<SceneBase> keyframeScene = sceneOptimized.clone();
    // keyframeScene->transform(keyframe.pose);
    sceneOptimized.transform(keyframe.pose);
    renderer.setScene(sceneOptimized);

    for (int lvl = 1; lvl >= 1; lvl--)
    {
        t.tic();
        // keyframeMesh.toRayIdepth();
        // std::vector<float> best_idepths = keyframeMesh.getVerticesIdepths();

        // PointSet* initialScene = (PointSet*)keyframeScene.clone();
        // MatrixMapped initialInvVar = invVar;

        e.setZero();
        for (std::size_t i = 0; i < frames.size(); i++)
            e += computeError(keyframe, frames[i], lvl);
        e.error /= e.count;

        e_regu = sceneOptimized.errorRegu();
        e_regu.error /= e_regu.count;

        // e_init = keyframeScene.errorInitial(initialScene, initialInvVar);
        //  e_init.error /= e_init.count;

        float last_error = e.error + meshRegularization * e_regu.error + meshInitial * e_init.error;

        std::cout << "--------lvl " << lvl << " initial error " << last_error << " " << t.toc() << std::endl;

        int maxIterations = 100;
        float lambda = 0.0;
        for (int it = 0; it < maxIterations; it++)
        {
            t.tic();

            hg.setZero();
            for (std::size_t i = 0; i < frames.size(); i++)
                hg += computeHGMap(keyframe, frames[i], lvl);

            std::vector<int> pIds = hg.G.getParamIds();

            Eigen::VectorXf G = hg.G.toEigen(pIds);
            Eigen::SparseMatrix<float> H = hg.H.toEigen(pIds);

            H /= hg.count;
            G /= hg.count;

            hg_regu = sceneOptimized.HGRegu();

            Eigen::VectorXf G_regu = hg_regu.G.toEigen(pIds);
            Eigen::SparseMatrix<float> H_regu = hg_regu.H.toEigen(pIds);

            H_regu /= hg_regu.count;
            G_regu /= hg_regu.count;

            // hg_init = HGInitial(initialScene, initialInvVar);

            // Eigen::VectorXf G_init = hg_init.G.toEigen(ids);
            // Eigen::SparseMatrix<float> H_init = hg_init.H.toEigen(ids);

            // H_init /= hg_init.count;
            // G_init /= hg_init.count;

            H += meshRegularization * H_regu; // + meshInitial * H_init;
            G += meshRegularization * G_regu; // + meshInitial * G_init;

            std::cout << "HG time " << t.toc() << std::endl;

            int n_try = 0;
            while (true)
            {
                Eigen::SparseMatrix<float> H_lambda = H;

                for (int j = 0; j < H_lambda.rows(); j++)
                {
                    H_lambda.coeffRef(j, j) *= (1.0 + lambda);
                }

                t.tic();

                H_lambda.makeCompressed();
                // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > solver;
                Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
                // Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;
                // Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;
                solver.analyzePattern(H_lambda);
                // std::cout << solver.info() << std::endl;
                solver.factorize(H_lambda);
                if (solver.info() != Eigen::Success)
                {
                    // some problem i have still to debug
                    it = maxIterations;
                    break;
                }
                // std::cout << solver.lastErrorMessage() << std::endl;
                Eigen::VectorXf inc = -solver.solve(G);
                // inc_depth = -acc_H_depth_lambda.llt().solve(acc_J_depth);
                // inc_depth = - acc_H_depth_lambda.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(acc_J_depth);
                // inc_depth = -acc_H_depth_lambda.colPivHouseholderQr().solve(acc_J_depth);

                std::cout << "solve time " << t.toc() << std::endl;

                std::vector<float> best_params;
                for (int index = 0; index < (int)pIds.size(); index++)
                {
                    float best_param = sceneOptimized.getParam(pIds[index]);
                    float new_param = best_param + inc(index);
                    best_params.push_back(best_param);
                    // the derivative is with respecto to the keyframe pose
                    // the update should take this into account
                    sceneOptimized.setParam(new_param, pIds[index]);
                }
                renderer.setScene(sceneOptimized);

                t.tic();

                e.setZero();
                for (std::size_t i = 0; i < frames.size(); i++)
                    e += computeError(keyframe, frames[i], lvl);
                e.error /= e.count;

                e_regu = sceneOptimized.errorRegu();
                e_regu.error /= e_regu.count;

                // e_init = errorInitial(initialMesh, initialInvVar);
                // e_init.error /= e_init.count;

                float error = e.error + meshRegularization * e_regu.error + meshInitial * e_init.error;

                std::cout << "new error " << error << " " << t.toc() << std::endl;

                if (error < last_error)
                {
                    // accept update, decrease lambda
                    float p = error / last_error;

                    if (lambda < 0.2f)
                        lambda = 0.0f;
                    else
                        lambda *= 0.5;

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
                    for (size_t index = 0; index < pIds.size(); index++)
                        sceneOptimized.setParam(best_params[index], pIds[index]);
                    renderer.setScene(sceneOptimized);

                    n_try++;

                    if (lambda < 0.2f)
                        lambda = 0.2f;
                    else
                        lambda *= 2.0; // std::pow(2.0, n_try);

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

void meshOptimizerCPU::optPoseMap(frameCPU &keyframe, std::vector<frameCPU> &frames)
{
    tic_toc t;

    Error e;
    Error e_regu;
    HGMapped hg;
    HGMapped hg_regu;

    // std::unique_ptr<SceneBase> keyframeScene = sceneOptimized.clone();
    // keyframeScene->transform(keyframe.pose);

    sceneOptimized.transform(keyframe.pose);
    renderer.setScene(sceneOptimized);

    for (int lvl = 1; lvl >= 1; lvl--)
    {
        t.tic();

        e.setZero();
        for (std::size_t i = 0; i < frames.size(); i++)
            e += computeError(keyframe, frames[i], lvl);
        e.error /= e.count;

        e_regu = sceneOptimized.errorRegu();
        e_regu.error /= e_regu.count;

        float last_error = e.error + meshRegularization * e_regu.error;

        std::cout << "--------lvl " << lvl << " initial error " << last_error << " " << t.toc() << std::endl;

        int maxIterations = 100;
        float lambda = 0.0;
        for (int it = 0; it < maxIterations; it++)
        {
            t.tic();

            hg.setZero();
            for (std::size_t i = 0; i < frames.size(); i++)
                hg += computeHGPoseMap(keyframe, frames[i], lvl);

            std::vector<int> pIds = hg.G.getParamIds();

            Eigen::VectorXf G = hg.G.toEigen(pIds);
            Eigen::SparseMatrix<float> H = hg.H.toEigen(pIds);

            H /= hg.count;
            G /= hg.count;

            hg_regu = sceneOptimized.HGRegu();

            Eigen::VectorXf G_regu = hg_regu.G.toEigen(pIds);
            Eigen::SparseMatrix<float> H_regu = hg_regu.H.toEigen(pIds);

            H_regu /= hg_regu.count;
            G_regu /= hg_regu.count;

            H += meshRegularization * H_regu;
            G += meshRegularization * G_regu;

            std::cout << "HG time " << t.toc() << std::endl;

            int n_try = 0;
            while (true)
            {
                Eigen::SparseMatrix<float> H_lambda = H;

                for (int j = 0; j < H_lambda.rows(); j++)
                {
                    H_lambda.coeffRef(j, j) *= (1.0 + lambda);
                }

                t.tic();

                H_lambda.makeCompressed();
                // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > solver;
                Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
                // Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;
                // Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;
                solver.analyzePattern(H_lambda);
                // std::cout << solver.info() << std::endl;
                solver.factorize(H_lambda);
                if (solver.info() != Eigen::Success)
                {
                    // some problem i have still to debug
                    it = maxIterations;
                    break;
                }
                // std::cout << solver.lastErrorMessage() << std::endl;
                Eigen::VectorXf inc = -solver.solve(G);
                // inc_depth = -acc_H_depth_lambda.llt().solve(acc_J_depth);
                // inc_depth = - acc_H_depth_lambda.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(acc_J_depth);
                // inc_depth = -acc_H_depth_lambda.colPivHouseholderQr().solve(acc_J_depth);

                std::cout << "solve time " << t.toc() << std::endl;

                // update pose
                std::vector<Sophus::SE3f> best_poses;
                for (size_t i = 0; i < frames.size(); i++)
                {
                    Eigen::Matrix<float, 6, 1> pose_inc;
                    // if ids are in order, like this I get the correct pose increment
                    // have to fix it some better way
                    for (int j = 0; j < 6; j++)
                    {
                        int paramId = j - (frames[i].id + 1) * 6;
                        int index = paramId + (frames[frames.size() - 1].id + 1) * 6;
                        pose_inc(j) = inc(index);
                    }
                    best_poses.push_back(frames[i].pose);
                    frames[i].pose = frames[i].pose * Sophus::SE3f::exp(pose_inc).inverse();
                }

                // update map

                std::map<unsigned int, float> best_params;
                for (int index = 0; index < (int)pIds.size(); index++)
                {
                    // negative ids are for the poses
                    if (pIds[index] < 0)
                        continue;

                    float best_param = sceneOptimized.getParam(pIds[index]);
                    float new_param = best_param + inc(index);
                    best_params[pIds[index]] = best_param;
                    sceneOptimized.setParam(new_param, pIds[index]);
                }
                renderer.setScene(sceneOptimized);

                t.tic();

                e.setZero();
                for (std::size_t i = 0; i < frames.size(); i++)
                    e += computeError(keyframe, frames[i], lvl);
                e.error /= e.count;

                e_regu = sceneOptimized.errorRegu();
                e_regu.error /= e_regu.count;

                float error = e.error + meshRegularization * e_regu.error;

                std::cout << "new error " << error << " " << t.toc() << std::endl;

                if (error < last_error)
                {
                    // accept update, decrease lambda
                    float p = error / last_error;

                    if (lambda < 0.2f)
                        lambda = 0.0f;
                    else
                        lambda *= 0.5;

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

                    for (size_t index = 0; index < pIds.size(); index++)
                    {
                        // negative ids are for the poses
                        if (pIds[index] < 0)
                            continue;
                        sceneOptimized.setParam(best_params[pIds[index]], pIds[index]);
                    }
                    renderer.setScene(sceneOptimized);

                    n_try++;

                    if (lambda < 0.2f)
                        lambda = 0.2f;
                    else
                        lambda *= 2.0; // std::pow(2.0, n_try);

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
