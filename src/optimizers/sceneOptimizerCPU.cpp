#include "optimizers/sceneOptimizerCPU.h"

template class sceneOptimizerCPU<SceneMesh, vec3<float>, vec3<int>>;
// template class meshOptimizerCPU<ScenePatches, vec1<float>, vec1<int>>;

template <typename sceneType, typename jmapType, typename idsType>
sceneOptimizerCPU<sceneType, jmapType, idsType>::sceneOptimizerCPU(camera &_cam)
    : cam(_cam),
      image_buffer(_cam.width, _cam.height, -1.0),
      idepth_buffer(_cam.width, _cam.height, -1.0),
      error_buffer(_cam.width, _cam.height, -1.0),
      jexp_buffer(_cam.width, _cam.height, vec2<float>(0.0)),
      jpose_buffer(_cam.width, _cam.height, vec6<float>(0.0)),
      jmap_buffer(_cam.width, _cam.height, jmapType(0.0)),
      pId_buffer(_cam.width, _cam.height, idsType(-1)),
      debug_buffer(_cam.width, _cam.height, -1.0),
      renderer(_cam.width, _cam.height)
{
    cam = _cam;

    sceneRegularization = 0.0;
    sceneInitial = 10.0;
    poseInitial = 0.0;


}

template <typename sceneType, typename jmapType, typename idsType>
void sceneOptimizerCPU<sceneType, jmapType, idsType>::optLightAffine(frameCPU &frame, frameCPU &kframe, sceneType &scene)
{
    int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

    for (int lvl = 3; lvl >= 1; lvl--)
    {
        vec2<float> best_affine = frame.getAffine();
        Error e = computeError(frame, kframe, scene, lvl);
        float last_error = e.getError();

        std::cout << "initial error " << last_error << " " << lvl << std::endl;

        for (int it = 0; it < maxIterations[lvl]; it++)
        {
            // HGPose hg = computeHGPose(idepth_buffer, keyframe, frame, lvl);
            DenseLinearProblem hg = computeHGLightAffine(frame, kframe, scene, lvl);

            float lambda = 0.0;
            int n_try = 0;
            while (true)
            {
                if (!hg.prepareH(lambda))
                {
                    n_try++;

                    if (lambda == 0.0f)
                        lambda = MIN_LAMBDA;
                    else
                        lambda *= std::pow(2.0, n_try);

                    continue;
                }
                Eigen::VectorXf inc = hg.solve();

                // Sophus::SE3f new_pose = frame.pose * Sophus::SE3f::exp(inc_pose);
                vec2<float> new_affine = best_affine - vec2<float>(inc(0), inc(1));
                frame.setAffine(new_affine);
                // Sophus::SE3f new_pose = Sophus::SE3f::exp(inc_pose).inverse() * frame.pose;

                // e.setZero();
                // e = computeError(idepth_buffer, keyframe, frame, lvl);
                e = computeError(frame, kframe, scene, lvl);
                float error = e.getError();
                // std::cout << "new error " << error << " time " << t.toc() << std::endl;

                std::cout << "new error " << error << " " << lambda << " " << it << " " << lvl << std::endl;

                if (error < last_error)
                {
                    // accept update, decrease lambda

                    best_affine = new_affine;
                    float p = error / last_error;

                    // if (lambda < MIN_LAMBDA)
                    //     lambda = 0.0f;
                    // else
                    //     lambda *= 0.5;
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
                        lambda = MIN_LAMBDA;
                    else
                        lambda *= std::pow(2.0, n_try);

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

template <typename sceneType, typename jmapType, typename idsType>
void sceneOptimizerCPU<sceneType, jmapType, idsType>::optPose(frameCPU &frame, frameCPU &kframe, sceneType &scene)
{
    for (int lvl = 2; lvl >= 2; lvl--)
    {
        Error e = computeError(frame, kframe, scene, lvl);
        e *= 1.0 / e.getCount();

        float last_error = e.getError();

        std::cout << "initial error " << last_error << " " << lvl << std::endl;
        std::vector<frameCPU> frames;
        frames.push_back(frame);
        plotDebug(scene, kframe, frames);

        float lambda = 0.0;
        bool keepIterating = true;
        while (keepIterating)
        {
            DenseLinearProblem hg = computeHGPose(frame, kframe, scene, lvl);
            hg *= 1.0 / hg.getCount();

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

                if (!hg.prepareH(lambda))
                    continue;

                Eigen::VectorXf inc = hg.solve();

                Sophus::SE3f best_pose = frame.getPose();
                Sophus::SE3f new_pose = frame.getPose() * Sophus::SE3f::exp(inc).inverse();

                frame.setPose(new_pose);

                float inc_mag = inc.dot(inc);

                e = computeError(frame, kframe, scene, lvl);
                if (e.getCount() < 0.5 * cam[lvl].width * cam[lvl].height)
                {
                    // too few pixels, unreliable, set to large error
                    e.setZero();
                    e += last_error;
                }
                e *= 1.0 / e.getCount();

                float error = e.getError();

                std::cout << "new error " << error << " " << lambda << " " << " " << lvl << std::endl;
                std::vector<frameCPU> frames;
                frames.push_back(frame);
                plotDebug(scene, kframe, frames);

                if (error < last_error)
                {
                    best_pose = new_pose;
                    float p = error / last_error;

                    last_error = error;

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

                    if (inc_mag < 1e-16)
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

template <typename sceneType, typename jmapType, typename idsType>
void sceneOptimizerCPU<sceneType, jmapType, idsType>::optMap(std::vector<frameCPU> &frames, frameCPU &kframe, sceneType &scene)
{
    std::vector<int> sceneParamsIds = scene.getParamIds();
    int numMapParams = sceneParamsIds.size();

    std::vector<float> init_sceneParams;
    for (int i = 0; i < sceneParamsIds.size(); i++)
    {
        float param = scene.getParam(sceneParamsIds[i]);
        init_sceneParams.push_back(param);
    }

    for (int lvl = 1; lvl >= 1; lvl--)
    {
        Error e;
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            Error ef = computeError(frames[i], kframe, scene, lvl);
            assert(ef.getCount() > 0.5 * cam[lvl].width * cam[lvl].height);
            e += ef;
        }
        e *= 1.0 / e.getCount();

        if (sceneRegularization > 0.0)
        {
            Error e_regu = scene.errorRegu();
            assert(e_regu.getCount() > 0);
            e_regu *= 1.0 / e_regu.getCount();
            e_regu *= sceneRegularization;
            e += e_regu;
        }

        if (sceneInitial > 0.0)
        {

            Error sceneInitialError;

            for (size_t index = 0; index < sceneParamsIds.size(); index++)
            {
                int sceneParamId = sceneParamsIds[index];
                float res = scene.getParam(sceneParamId) - init_sceneParams[sceneParamId];
                sceneInitialError += res;
            }

            sceneInitialError *= 1.0 / sceneParamsIds.size();
            sceneInitialError *= sceneInitial;
            e += sceneInitialError;
        }

        float last_error = e.getError();

        std::cout << "optMap initial error " << last_error << " " << lvl << std::endl;
        plotDebug(scene, kframe, frames);

        int maxIterations = 1000;
        float lambda = 0.0;
        for (int it = 0; it < maxIterations; it++)
        {
            DenseLinearProblem hg(numMapParams);
            for (std::size_t i = 0; i < frames.size(); i++)
            {
                DenseLinearProblem fhg = computeHGMap(frames[i], kframe, scene, lvl);
                assert(fhg.getCount() > 0.5 * cam[lvl].width * cam[lvl].height);
                hg += fhg;
            }
            hg *= 1.0 / hg.getCount();

            if (sceneRegularization > 0.0)
            {
                DenseLinearProblem hg_regu = scene.HGRegu(0);
                assert(hg_regu.getCount() > 0);
                hg_regu *= 1.0 / hg_regu.getCount();
                hg_regu *= sceneRegularization;
                hg += hg_regu;
            }

            if (sceneInitial > 0.0)
            {
                for (size_t index = 0; index < sceneParamsIds.size(); index++)
                {
                    int sceneParamId = sceneParamsIds[index];
                    int linearProblemParamId = sceneParamId;

                    float res = scene.getParam(sceneParamId) - init_sceneParams[sceneParamId];

                    vec1<float> jacobian = 1.0;
                    vec1<int> ids = linearProblemParamId;
                    hg.add(jacobian, res, 1.0, ids);
                }
            }

            // saveH(hg, "H.png");

            // std::vector<int> paramIds = hg.removeUnobservedParams();
            std::vector<int> linearProbleParamIds = hg.getParamIds();

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

                if (!hg.prepareH(lambda))
                    continue;

                Eigen::VectorXf inc = hg.solve();

                std::vector<float> best_params;
                float map_inc_mag = 0.0;
                int map_inc_mag_count = 0;
                for (size_t index = 0; index < linearProbleParamIds.size(); index++)
                {
                    int paramId = linearProbleParamIds[index];

                    float best_param = scene.getParam(paramId);
                    float inc_param = inc(index);
                    float new_param = best_param - inc_param;

                    best_params.push_back(best_param);
                    // the derivative is with respecto to the keyframe pose
                    // the update should take this into account
                    scene.setParam(new_param, paramId);
                    map_inc_mag += inc_param * inc_param;
                    map_inc_mag_count += 1;
                }
                map_inc_mag /= map_inc_mag_count;

                e.setZero();
                for (std::size_t i = 0; i < frames.size(); i++)
                {
                    Error fe = computeError(frames[i], kframe, scene, lvl);
                    if (fe.getCount() < 0.5 * cam[lvl].width * cam[lvl].height)
                    {
                        // too few pixels, unreliable, set to large error
                        fe.setZero();
                        fe += last_error;
                    }
                    e += fe;
                }
                e *= 1.0 / e.getCount();

                if (sceneRegularization > 0.0)
                {
                    Error e_regu = scene.errorRegu();
                    assert(e_regu.getCount() > 0);
                    e_regu *= 1.0 / e_regu.getCount();
                    e_regu *= sceneRegularization;
                    e += e_regu;
                }

                if (sceneInitial > 0.0)
                {
                    Error sceneInitialError;

                    for (size_t index = 0; index < sceneParamsIds.size(); index++)
                    {
                        int sceneParamId = sceneParamsIds[index];
                        float res = scene.getParam(sceneParamId) - init_sceneParams[sceneParamId];
                        sceneInitialError += res;
                    }

                    sceneInitialError *= 1.0 / sceneParamsIds.size();
                    sceneInitialError *= sceneInitial;
                    e += sceneInitialError;
                }

                float error = e.getError();

                std::cout << "new error " << error << " " << lambda << " " << it << " " << n_try << " lvl: " << lvl << std::endl;
                plotDebug(scene, kframe, frames);

                if (error < last_error)
                {
                    // accept update, decrease lambda
                    float p = error / last_error;

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
                    for (size_t index = 0; index < linearProbleParamIds.size(); index++)
                    {
                        scene.setParam(best_params[index], linearProbleParamIds[index]);
                    }

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

template <typename sceneType, typename jmapType, typename idsType>
void sceneOptimizerCPU<sceneType, jmapType, idsType>::optPoseMap(std::vector<frameCPU> &frames, frameCPU &kframe, sceneType &scene)
{
    assert(frames.size() > 0);

    std::vector<int> sceneParamsIds = scene.getParamIds();

    int numSceneParams = sceneParamsIds.size();
    int numFrameParams = frames.size() * 6;

    std::vector<Eigen::Matrix<float, 6, 1>> init_poses;
    for (auto &frame : frames)
    {
        init_poses.push_back(frame.getPose().log());
    }

    std::vector<float> init_sceneParams;
    for (int i = 0; i < sceneParamsIds.size(); i++)
    {
        float param = scene.getParam(sceneParamsIds[i]);
        init_sceneParams.push_back(param);
    }

    for (int lvl = 1; lvl >= 1; lvl--)
    {
        Error e;
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            Error fe = computeError(frames[i], kframe, scene, lvl);
            assert(fe.getCount() > 0.5 * cam[lvl].width * cam[lvl].height);
            e += fe;
        }
        e *= 1.0 / e.getCount();

        if (sceneRegularization > 0.0)
        {
            Error e_regu = scene.errorRegu();
            assert(e_regu.getCount() > 0);
            e_regu *= 1.0 / e_regu.getCount();
            e_regu *= sceneRegularization;
            e += e_regu;
        }

        if (poseInitial > 0.0)
        {
            Error poseInitialError;
            for (std::size_t i = 0; i < frames.size(); i++)
            {
                Eigen::Matrix<float, 6, 1> diff = frames[i].getPose().log() - init_poses[i];
                poseInitialError += diff.dot(diff);
            }
            poseInitialError *= 1.0 / frames.size();
            poseInitialError *= poseInitial;
            e += poseInitialError;
        }

        if (sceneInitial > 0.0)
        {
            Error sceneInitialError;

            for (size_t index = 0; index < sceneParamsIds.size(); index++)
            {
                int sceneParamId = sceneParamsIds[index];
                float res = scene.getParam(sceneParamId) - init_sceneParams[sceneParamId];
                sceneInitialError += res;
            }

            sceneInitialError *= 1.0 / sceneParamsIds.size();
            sceneInitialError *= sceneInitial;
            e += sceneInitialError;
        }

        float last_error = e.getError();

        std::cout << "optPoseMap initial error " << last_error << " lvl: " << lvl << std::endl;
        plotDebug(scene, kframe, frames);

        int maxIterations = 1000;
        float lambda = 0.0;
        for (int it = 0; it < maxIterations; it++)
        {
            DenseLinearProblem hg(numFrameParams + numSceneParams);
            for (std::size_t i = 0; i < frames.size(); i++)
            {
                DenseLinearProblem fhg = computeHGPoseMap(frames[i], kframe, scene, i, frames.size(), lvl);
                assert(fhg.getCount() > 0.5 * cam[lvl].width * cam[lvl].height);
                hg += fhg;
            }
            hg *= 1.0 / hg.getCount();

            if (sceneRegularization > 0.0)
            {
                DenseLinearProblem hg_regu = scene.HGRegu(frames.size());
                assert(hg_regu.getCount() > 0);
                hg_regu *= 1.0 / hg_regu.getCount();
                hg_regu *= sceneRegularization;
                hg += hg_regu;
            }

            if (poseInitial > 0.0)
            {
                for (std::size_t i = 0; i < frames.size(); i++)
                {
                    Eigen::Matrix<float, 6, 1> _pose_error = frames[i].getPose().log() - init_poses[i];
                    vec8<float> pose_error = vec8<float>(_pose_error(0), _pose_error(1), _pose_error(2), _pose_error(3),
                                                         _pose_error(4), _pose_error(5), 0.0, 0.0) *
                                             poseInitial / frames.size();
                    vec8<float> pose_jac = vec8<float>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0) * poseInitial / frames.size();
                    vec8<int> pose_ids;
                    for (int j = 0; j < 8; j++)
                        pose_ids(j) = i * 8 + j - frames.size() * 8;

                    hg.add(pose_jac, pose_error, 1.0, pose_ids);
                }
            }

            if (sceneInitial > 0.0)
            {
                for (size_t index = 0; index < sceneParamsIds.size(); index++)
                {
                    int sceneParamId = sceneParamsIds[index];
                    int linearProblemParamId = sceneParamId;

                    float res = scene.getParam(sceneParamId) - init_sceneParams[sceneParamId];

                    vec1<float> jacobian = 1.0;
                    vec1<int> ids = linearProblemParamId;
                    hg.add(jacobian, res, 1.0, ids);
                }
            }

            // std::vector<int> paramIds = hg.removeUnobservedParams();
            std::vector<int> linearProblemParamIds = hg.getParamIds();
            // saveH(hg, "H.png");

            int n_try = 0;
            // lambda = 0.0;
            lambda *= 0.75;
            while (true)
            {
                if (n_try > 0)
                {
                    if (lambda < MIN_LAMBDA)
                        lambda = MIN_LAMBDA;
                    lambda *= std::pow(2.0, n_try);
                }
                n_try++;

                if (!hg.prepareH(lambda))
                    continue;

                Eigen::VectorXf inc = hg.solve();

                // update pose
                std::vector<Sophus::SE3f> best_poses;
                std::vector<vec2<float>> best_affines;
                best_poses.reserve(frames.size());
                best_affines.reserve(frames.size());
                float pose_inc_mag = 0.0;
                for (size_t i = 0; i < frames.size(); i++)
                {
                    Eigen::Matrix<float, 8, 1> pose_inc = inc.segment(i * 8, 8);

                    pose_inc_mag += pose_inc.dot(pose_inc);

                    best_poses.push_back(frames[i].getPose());
                    best_affines.push_back(frames[i].getAffine());

                    Sophus::SE3f new_pose = frames[i].getPose() * Sophus::SE3f::exp(pose_inc.segment(0, 6)).inverse();
                    // Sophus::SE3f new_pose = ((frames[i].getPose() * kframe.getPose().inverse()) * Sophus::SE3f::exp(pose_inc.segment(0, 6)).inverse()) * kframe.getPose();

                    // if (i == frames.size() - 1)
                    //     new_pose.translation() = new_pose.translation().normalized() * frames[i].getPose().translation().norm();
                    vec2<float> new_affine = frames[i].getAffine() - vec2<float>(pose_inc(6), pose_inc(7));
                    frames[i].setPose(new_pose);
                    frames[i].setAffine(new_affine);
                }
                pose_inc_mag /= frames.size();

                // update map
                std::map<unsigned int, float> best_params;
                float map_inc_mag = 0.0;
                int map_inc_mag_count = 0;
                for (size_t index = 0; index < linearProblemParamIds.size(); index++)
                {
                    int paramId = linearProblemParamIds[index];
                    // negative ids are for the poses
                    if (paramId < numFrameParams)
                        continue;

                    int mapParamId = paramId - numFrameParams;

                    float best_param = scene.getParam(mapParamId);
                    float inc_param = inc(index);
                    float new_param = best_param - inc_param;
                    // if(std::fabs(inc_param/best_param) > 0.4)
                    //     solverSucceded = false;
                    best_params[mapParamId] = best_param;

                    scene.setParam(new_param, mapParamId);

                    map_inc_mag += inc_param * inc_param;
                    map_inc_mag_count += 1;
                }
                map_inc_mag /= map_inc_mag_count;

                e.setZero();
                for (std::size_t i = 0; i < frames.size(); i++)
                {
                    Error fe = computeError(frames[i], kframe, scene, lvl);
                    if (fe.getCount() < 0.5 * cam[lvl].width * cam[lvl].height)
                    {
                        // too few pixels, unreliable, set to large error
                        fe.setZero();
                        fe += last_error;
                    }
                    e += fe;
                }

                e *= 1.0 / e.getCount();

                if (sceneRegularization > 0.0)
                {
                    Error e_regu = scene.errorRegu();
                    e_regu *= 1.0 / e_regu.getCount();
                    e_regu *= sceneRegularization;
                    e += e_regu;
                }

                if (poseInitial > 0.0)
                {
                    Error poseInitialError;
                    for (std::size_t i = 0; i < frames.size(); i++)
                    {
                        Eigen::Matrix<float, 6, 1> diff = frames[i].getPose().log() - init_poses[i];
                        poseInitialError += diff.dot(diff);
                    }
                    poseInitialError *= 1.0 / frames.size();
                    poseInitialError *= poseInitial;
                    e += poseInitialError;
                }

                if (sceneInitial > 0.0)
                {
                    Error sceneInitialError;

                    for (size_t index = 0; index < sceneParamsIds.size(); index++)
                    {
                        int sceneParamId = sceneParamsIds[index];
                        float res = scene.getParam(sceneParamId) - init_sceneParams[sceneParamId];
                        sceneInitialError += res;
                    }

                    sceneInitialError *= 1.0 / sceneParamsIds.size();
                    sceneInitialError *= sceneInitial;
                    e += sceneInitialError;
                }

                float error = e.getError();

                std::cout << "new error " << error << " " << it << " " << lambda << " " << n_try << " lvl: " << lvl << std::endl;
                plotDebug(scene, kframe, frames);

                if (error < last_error)
                {
                    // accept update, decrease lambda
                    float p = error / last_error;

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

                    for (int index = 0; index < linearProblemParamIds.size(); index++)
                    {
                        int paramId = linearProblemParamIds[index];
                        if (paramId < numFrameParams)
                            continue;

                        int mapParamId = paramId - numFrameParams;

                        scene.setParam(best_params[mapParamId], mapParamId);
                    }

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
