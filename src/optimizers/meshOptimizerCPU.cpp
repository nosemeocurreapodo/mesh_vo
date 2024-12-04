#include "optimizers/meshOptimizerCPU.h"

template class meshOptimizerCPU<SceneMesh, vec3<float>, vec3<int>>;
// template class meshOptimizerCPU<ScenePatches, vec1<float>, vec1<int>>;

template <typename sceneType, typename jmapType, typename idsType>
meshOptimizerCPU<sceneType, jmapType, idsType>::meshOptimizerCPU(camera &_cam)
    : kimage(_cam.width, _cam.height, -1.0),
      image_buffer(_cam.width, _cam.height, -1.0),
      idepth_buffer(_cam.width, _cam.height, -1.0),
      ivar_buffer(_cam.width, _cam.height, -1.0),
      error_buffer(_cam.width, _cam.height, -1.0),
      jlightaffine_buffer(_cam.width, _cam.height, vec2<float>(0.0)),
      jpose_buffer(_cam.width, _cam.height, vec8<float>(0.0)),
      jmap_buffer(_cam.width, _cam.height, jmapType(0.0)),
      pId_buffer(_cam.width, _cam.height, idsType(-1)),
      debug(_cam.width, _cam.height, -1.0),
      idepthVar(_cam.width, _cam.height, -1.0),
      renderer(_cam.width, _cam.height)
{
    int lvl = 0;
    while (true)
    {
        camera lvlcam = _cam;
        lvlcam.resize(1.0 / std::pow(2.0, lvl));
        if (lvlcam.width == 0 || lvlcam.height == 0)
            break;
        cam.push_back(lvlcam);
        lvl++;
    }

    multiThreading = false;
    meshRegularization = 1.0;
    meshInitial = 0.0;
}

template <typename sceneType, typename jmapType, typename idsType>
void meshOptimizerCPU<sceneType, jmapType, idsType>::initKeyframe(frameCPU &frame, int lvl)
{
    idepth_buffer.set(idepth_buffer.nodata, lvl);
    // renderer.renderRandom(cam[lvl], &idepth_buffer, lvl, 0.5, 1.5);
    renderer.renderSmooth(cam[lvl], &idepth_buffer, lvl, 0.1, 2.0);
    ivar_buffer.set(ivar_buffer.nodata, lvl);
    renderer.renderSmooth(cam[lvl], &ivar_buffer, lvl, initialIvar(), initialIvar());
    kscene.init(cam[lvl], Sophus::SE3f(), idepth_buffer, ivar_buffer, lvl);
    kimage = frame.getRawImage();
}

template <typename sceneType, typename jmapType, typename idsType>
void meshOptimizerCPU<sceneType, jmapType, idsType>::initKeyframe(frameCPU &frame, dataCPU<float> &idepth, dataCPU<float> &ivar, int lvl)
{
    kscene.init(cam[lvl], Sophus::SE3f(), idepth, ivar, lvl);

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
    // kframe.setAffine(vec2<float>(0.0, 0.0));
}

template <typename sceneType, typename jmapType, typename idsType>
void meshOptimizerCPU<sceneType, jmapType, idsType>::initKeyframe(frameCPU &frame, std::vector<vec2<float>> &texcoords, std::vector<float> &idepths, int lvl)
{
    kscene.init(cam[lvl], Sophus::SE3f(), texcoords, idepths);
    kimage = frame.getRawImage();
}

template <typename sceneType, typename jmapType, typename idsType>
void meshOptimizerCPU<sceneType, jmapType, idsType>::normalizeDepth()
{
    vec2<float> affine = kscene.meanStdDepthParam();
    affine(1) = 0.0;
    kscene.scaleDepthParam(affine);
}

template <typename sceneType, typename jmapType, typename idsType>
void meshOptimizerCPU<sceneType, jmapType, idsType>::optLightAffine(frameCPU &frame)
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
            DenseLinearProblem hg = computeHGLightAffine(&frame, lvl, false);

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
                e = computeError(&frame, lvl, false);
                float error = e.getError();
                // std::cout << "new error " << error << " time " << t.toc() << std::endl;

                std::cout << "new error " << error << " " << lambda << " " << it << " " << lvl << std::endl;

                if (error < last_error)
                {
                    // accept update, decrease lambda

                    best_affine = new_affine;
                    float p = error / last_error;

                    //if (lambda < MIN_LAMBDA)
                    //    lambda = 0.0f;
                    //else
                    //    lambda *= 0.5;
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
void meshOptimizerCPU<sceneType, jmapType, idsType>::optPose(frameCPU &frame)
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
            DenseLinearProblem hg = computeHGPose(&frame, lvl, false);

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

                Sophus::SE3f new_pose = best_pose * Sophus::SE3f::exp(inc.segment(0, 6)).inverse();
                vec2<float> new_affine = best_affine; // - vec2<float>(inc(6), inc(7));
                frame.setPose(new_pose);
                frame.setAffine(new_affine);

                e = computeError(&frame, lvl, false);
                float error = e.getError();

                std::cout << "new error " << error << " " << lambda << " " << it << " " << lvl << std::endl;

                if (error < last_error)
                {
                    // accept update, decrease lambda

                    best_pose = new_pose;
                    best_affine = new_affine;
                    float p = error / last_error;

                    //if (lambda < MIN_LAMBDA)
                    //    lambda = 0.0f;
                    //else
                    //    lambda *= 0.5;
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
void meshOptimizerCPU<sceneType, jmapType, idsType>::optMap(std::vector<frameCPU> &frames)
{
    int numMapParams = kscene.getParamIds().size();
    Error e;
    Error e_regu;
    DenseLinearProblem hg(0, numMapParams);
    DenseLinearProblem hg_regu(0, numMapParams);

    for (int lvl = 4; lvl >= 1; lvl--)
    {
        e.setZero();
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            e += computeError(&frames[i], lvl);
        }

        e *= 1.0 / frames.size();

        if (meshRegularization > 0.0)
        {
            e_regu = kscene.errorRegu();
            e_regu *= meshRegularization;
            e += e_regu;
        }

        float last_error = e.getError();

        std::cout << "optMap initial error " << last_error << " " << lvl << std::endl;

        int maxIterations = 1000;
        float lambda = 0.0;
        for (int it = 0; it < maxIterations; it++)
        {
            hg.setZero();
            for (std::size_t i = 0; i < frames.size(); i++)
            {
                hg += computeHGMap2(&frames[i], lvl);
            }

            hg *= 1.0 / frames.size();

            if (meshRegularization > 0.0)
            {
                hg_regu = kscene.HGRegu(0);
                hg_regu *= meshRegularization;
                hg += hg_regu;
            }
            // saveH(hg, "H.png");

            // std::vector<int> paramIds = hg.removeUnobservedParams();
            std::vector<int> paramIds = hg.getParamIds();

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

                std::vector<float> best_params;
                float map_inc_mag = 0.0;
                int map_inc_mag_count = 0;
                for (size_t index = 0; index < paramIds.size(); index++)
                {
                    int paramId = paramIds[index];

                    float best_param = kscene.getParam(paramId);
                    float inc_param = inc(index);
                    float new_param = best_param - inc_param;

                    float weight = 0.1; // H.coeffRef(index, index);
                    best_params.push_back(best_param);
                    // the derivative is with respecto to the keyframe pose
                    // the update should take this into account
                    kscene.setParam(new_param, paramId);
                    kscene.setParamWeight(weight, paramId);
                    map_inc_mag += inc_param * inc_param;
                    map_inc_mag_count += 1;
                }
                map_inc_mag /= map_inc_mag_count;

                e.setZero();
                for (std::size_t i = 0; i < frames.size(); i++)
                {
                    e += computeError(&frames[i], lvl);
                }

                e *= 1.0 / frames.size();

                if (meshRegularization > 0.0)
                {
                    e_regu = kscene.errorRegu();
                    e_regu *= meshRegularization;
                    e += e_regu;
                }

                float error = e.getError();

                std::cout << "new error " << error << " " << lambda << " " << it << " " << lvl << std::endl;

                if (error < last_error)
                {
                    // accept update, decrease lambda
                    float p = error / last_error;

                    //if (lambda < MIN_LAMBDA)
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
                    for (size_t index = 0; index < paramIds.size(); index++)
                    {
                        kscene.setParam(best_params[index], paramIds[index]);
                    }

                    n_try++;

                    if (lambda == 0.0f)
                        lambda = MIN_LAMBDA;
                    else
                        lambda *= std::pow(2.0, n_try);

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
void meshOptimizerCPU<sceneType, jmapType, idsType>::optPoseMap(std::vector<frameCPU> &frames)
{
    assert(frames.size() > 0);

    int numMapParams = kscene.getParamIds().size();
    int numFrameParams = frames.size() * 8;

    Error e;
    Error e_regu;
    DenseLinearProblem hg(numFrameParams, numMapParams);
    DenseLinearProblem hg_regu(numFrameParams, numMapParams);

    for (int lvl = 0; lvl >= 0; lvl--)
    {
        e.setZero();
        for (std::size_t i = 0; i < frames.size(); i++)
        {
            e += computeError(&frames[i], lvl);
        }

        e *= 1.0 / frames.size();

        if (meshRegularization > 0.0)
        {
            e_regu = kscene.errorRegu();
            e_regu *= meshRegularization;
            e += e_regu;
        }

        float last_error = e.getError();

        std::cout << "optPoseMap initial error " << last_error << " lvl: " << lvl << std::endl;

        int maxIterations = 1000;
        float lambda = 0.0;
        for (int it = 0; it < maxIterations; it++)
        {
            hg.setZero();
            for (std::size_t i = 0; i < frames.size(); i++)
            {
                hg += computeHGPoseMap2(&frames[i], i, frames.size(), lvl);
            }

            hg *= 1.0 / frames.size();

            if (meshRegularization > 0.0)
            {
                hg_regu = kscene.HGRegu(frames.size());
                hg_regu *= meshRegularization;
                hg += hg_regu;
            }

            // std::vector<int> paramIds = hg.removeUnobservedParams();
            std::vector<int> paramIds = hg.getParamIds();
            // saveH(hg, "H.png");

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
                    if (i == frames.size() - 1)
                        new_pose.translation() = new_pose.translation().normalized() * frames[i].getPose().translation().norm();
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
                for (size_t index = 0; index < paramIds.size(); index++)
                {
                    int paramId = paramIds[index];
                    // negative ids are for the poses
                    if (paramId < numFrameParams)
                        continue;

                    int mapParamId = paramId - numFrameParams;

                    float best_param = kscene.getParam(mapParamId);
                    float inc_param = inc(index);
                    float new_param = best_param - inc_param;
                    float weight = 0.1; // H.coeffRef(id.second, id.second);
                    // if(std::fabs(inc_param/best_param) > 0.4)
                    //     solverSucceded = false;
                    best_params[mapParamId] = best_param;
                    kscene.setParam(new_param, mapParamId);
                    kscene.setParamWeight(weight, mapParamId);
                    map_inc_mag += inc_param * inc_param;
                    map_inc_mag_count += 1;
                }
                map_inc_mag /= map_inc_mag_count;

                e.setZero();
                for (std::size_t i = 0; i < frames.size(); i++)
                {
                    e += computeError(&frames[i], lvl);
                }

                e *= 1.0 / frames.size();

                if (meshRegularization > 0.0)
                {
                    e_regu = kscene.errorRegu();
                    e_regu *= meshRegularization;
                    e += e_regu;
                }

                float error = e.getError();

                std::cout << "new error " << error << " " << it << " " << lambda << " lvl: " << lvl << std::endl;

                if (error < last_error)
                {
                    // accept update, decrease lambda
                    float p = error / last_error;

                    //if (lambda < MIN_LAMBDA)
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
                    for (size_t index = 0; index < frames.size(); index++)
                    {
                        frames[index].setPose(best_poses[index]);
                        frames[index].setAffine(best_affines[index]);
                    }

                    for (int index = 0; index < paramIds.size(); index++)
                    {
                        int paramId = paramIds[index];
                        if (paramId < numFrameParams)
                            continue;

                        int mapParamId = paramId - numFrameParams;

                        kscene.setParam(best_params[mapParamId], mapParamId);
                    }

                    n_try++;

                    if (lambda == 0.0f)
                        lambda = MIN_LAMBDA;
                    else
                        lambda *= std::pow(2.0, n_try);

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
