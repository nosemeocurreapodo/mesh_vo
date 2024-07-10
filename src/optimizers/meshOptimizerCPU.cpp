#include "optimizers/meshOptimizerCPU.h"
#include <math.h>
#include "utils/tictoc.h"

meshOptimizerCPU::meshOptimizerCPU(camera &_cam)
    : keyframe(_cam.width, _cam.height),
      image_buffer(_cam.width, _cam.height, -1.0),
      idepth_buffer(_cam.width, _cam.height, -1.0),
      error_buffer(_cam.width, _cam.height, -1.0),
      j1_buffer(_cam.width, _cam.height, Eigen::Vector3f(0.0, 0.0, 0.0)),
      j2_buffer(_cam.width, _cam.height, Eigen::Vector3f(0.0, 0.0, 0.0)),
      j3_buffer(_cam.width, _cam.height, Eigen::Vector3f(0.0, 0.0, 0.0)),
      id_buffer(_cam.width, _cam.height, Eigen::Vector3i(-1, -1, -1)),
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
    jacMethod = MapJacobianMethod::idepth;
}

void meshOptimizerCPU::initKeyframe(frameCPU &frame, dataCPU<float> &idepth, dataCPU<float> &idepthVar, int lvl)
{
    keyframeMesh.clear();
    invVar.clear();

    for (float y = 0.0; y < MESH_HEIGHT; y++)
    {
        for (float x = 0.0; x < MESH_WIDTH; x++)
        {
            Eigen::Vector2f pix;
            pix[0] = (cam[lvl].width - 1) * x / (MESH_WIDTH - 1);
            pix[1] = (cam[lvl].height - 1) * y / (MESH_HEIGHT - 1);
            Eigen::Vector3f ray = cam[lvl].pixToRay(pix);
            float idph = idepth.get(pix[1], pix[0], lvl);
            float idVar = idepthVar.get(pix[1], pix[0], lvl);
            if (idph == idepth.nodata)
                continue;

            if (idph <= 0.0)
                continue;

            Eigen::Vector3f vertice = ray / idph;

            unsigned int id = keyframeMesh.addVertice(vertice);

            invVar[id][id] = idVar;
        }
    }

    keyframe = frame;
    keyframeMesh.setPose(frame.pose);
    keyframeMesh.buildTriangles(cam[lvl]);
}

MeshTexCoordsCPU meshOptimizerCPU::buildFrameMesh(frameCPU &frame, int lvl)
{
    MeshTexCoordsCPU frameMesh = keyframeMesh;
    frameMesh.transform(frame.pose);
    frameMesh.removeOcluded(cam[lvl]);
    // frameMesh.devideBigTriangles(cam[lvl]);

    dataCPU<float> frameIdepth(cam[0].width, cam[0].height, -1);
    renderer.renderIdepth(keyframeMesh, cam[lvl], frame.pose, frameIdepth, lvl);

    for (float y = 0.0; y < MESH_HEIGHT; y++)
    {
        for (float x = 0.0; x < MESH_WIDTH; x++)
        {
            Eigen::Vector2f pix;
            pix[0] = (cam[lvl].width - 1) * x / (MESH_WIDTH - 1);
            pix[1] = (cam[lvl].height - 1) * y / (MESH_HEIGHT - 1);

            float idph = frameIdepth.get(pix[1], pix[0], lvl);
            if (idph != frameIdepth.nodata)
                continue;

            Eigen::Vector3f ray = cam[lvl].pixToRay(pix);

            float idepth = (1.0 - 0.5) * float(rand() % 1000) / 1000.0 + 0.5;

            Eigen::Vector3f vertice = ray / idepth;

            unsigned int id = frameMesh.addVertice(vertice);

            invVar[id][id] = 1.0 / INITIAL_VAR;
        }
    }

    frameMesh.buildTriangles(cam[lvl]);

    return frameMesh;
}

Error meshOptimizerCPU::computeError(frameCPU &frame, int lvl)
{
    Error e;

    image_buffer.set(image_buffer.nodata, lvl);
    renderer.renderImage(keyframeMesh, cam[lvl], keyframe.image, frame.pose, image_buffer, lvl);

    std::array<int, 2> size = frame.image.getSize(lvl);
    for (int y = 0; y < size[1]; y++)
        for (int x = 0; x < size[0]; x++)
        {
            float p1 = frame.image.get(y, x, lvl);
            float p2 = image_buffer.get(y, x, lvl);
            if (p1 == frame.image.nodata || p2 == image_buffer.nodata)
                continue;
            e.error += std::pow(p1 - p2, 2);
            e.count++;
        }

    return e;
}

HGMapped meshOptimizerCPU::computeHGPose(frameCPU &frame, int lvl)
{
    HGMapped hg;

    // renderer.renderJPose(keyframeMesh, cam[lvl], keyframe, frame, j1_buffer, j2_buffer, error_buffer, lvl);
    idepth_buffer.set(idepth_buffer.nodata, lvl);
    renderer.renderIdepth(keyframeMesh, cam[lvl], frame.pose, idepth_buffer, lvl);
    renderer.renderJPose(idepth_buffer, cam[lvl], keyframe, frame, j1_buffer, j2_buffer, error_buffer, lvl);

    for (int y = 0; y < cam[lvl].height; y++)
    {
        for (int x = 0; x < cam[lvl].width; x++)
        {
            Eigen::Vector3f j_tra = j1_buffer.get(y, x, lvl);
            Eigen::Vector3f j_rot = j2_buffer.get(y, x, lvl);
            float err = error_buffer.get(y, x, lvl);
            if (j_tra == j1_buffer.nodata || j_rot == j2_buffer.nodata || err == error_buffer.nodata)
                continue;
            Eigen::Matrix<float, 6, 1> J;
            J << j_tra(0), j_tra(1), j_tra(2), j_rot(0), j_rot(1), j_rot(2);
            hg.count++;
            for (int i = 0; i < 6; i++)
            {
                hg.G.add(J[i] * err, i - 6);
                // hg->G[i - 6] = J[i] * residual;
                for (int j = i; j < 6; j++)
                {
                    float jj = J[i] * J[j];
                    hg.H.add(jj, i - 6, j - 6);
                    hg.H.add(jj, j - 6, i - 6);
                }
            }
        }
    }

    return hg;
}

HGMapped meshOptimizerCPU::computeHGMapDepth(frameCPU &frame, int lvl)
{
    HGMapped hg;

    error_buffer.set(error_buffer.nodata, lvl);
    j1_buffer.set(j1_buffer.nodata, lvl);
    id_buffer.set(id_buffer.nodata, lvl);

    renderer.renderJMapDepth(keyframeMesh, cam[lvl], keyframe, frame, j1_buffer, error_buffer, id_buffer, jacMethod, lvl);

    for (int y = 0; y < cam[lvl].height; y++)
    {
        for (int x = 0; x < cam[lvl].width; x++)
        {
            Eigen::Vector3f jac = j1_buffer.get(y, x, lvl);
            float err = error_buffer.get(y, x, lvl);
            Eigen::Vector3i ids = id_buffer.get(y, x, lvl);

            if (jac == j1_buffer.nodata || err == error_buffer.nodata || ids == id_buffer.nodata)
                continue;

            hg.count += 1;
            for (int i = 0; i < 3; i++)
            {
                // if the jacobian is 0
                // we really cannot say anything about the depth
                // can make the hessian non-singular
                if (jac(i) == 0)
                    continue;

                hg.G.add(jac(i) * err, ids(i));
                //(*hg).G[v_ids[i]] += J[i] * error;

                for (int j = i; j < 3; j++)
                {
                    float jj = jac(i) * jac(j);
                    hg.H.add(jj, ids(i), ids(j));
                    hg.H.add(jj, ids(j), ids(i));
                    //(*hg).H[v_ids[i]][v_ids[j]] += jj;
                    //(*hg).H[v_ids[j]][v_ids[i]] += jj;
                }
            }
        }
    }

    return hg;
}

HGMapped meshOptimizerCPU::computeHGMapNormalDepth(frameCPU &frame, int lvl)
{
    HGMapped hg;

    error_buffer.set(error_buffer.nodata, lvl);
    j1_buffer.set(j1_buffer.nodata, lvl);
    j2_buffer.set(j2_buffer.nodata, lvl);
    j3_buffer.set(j3_buffer.nodata, lvl);
    id_buffer.set(id_buffer.nodata, lvl);

    renderer.renderJMapNormalDepth(keyframeMesh, cam[lvl], keyframe, frame, j1_buffer, j2_buffer, j3_buffer, error_buffer, id_buffer, jacMethod, lvl);

    for (int y = 0; y < cam[lvl].height; y++)
    {
        for (int x = 0; x < cam[lvl].width; x++)
        {
            Eigen::Vector3f jacDepth = j1_buffer.get(y, x, lvl);
            Eigen::Vector2f jacNormal[3];
            jacNormal[0](0) = j2_buffer.get(y, x, lvl)(0);
            jacNormal[0](1) = j2_buffer.get(y, x, lvl)(1);
            jacNormal[1](0) = j2_buffer.get(y, x, lvl)(2);
            jacNormal[1](1) = j3_buffer.get(y, x, lvl)(0);
            jacNormal[2](0) = j3_buffer.get(y, x, lvl)(1);
            jacNormal[2](1) = j3_buffer.get(y, x, lvl)(2);

            float err = error_buffer.get(y, x, lvl);
            Eigen::Vector3i ids = id_buffer.get(y, x, lvl);

            if (jacDepth == j1_buffer.nodata || err == error_buffer.nodata || ids == id_buffer.nodata)
                continue;

            hg.count += 1;
            for (int i = 0; i < 3; i++)
            {
                // if the jacobian is 0
                // we really cannot say anything about the depth
                // can make the hessian non-singular
                if (jacDepth(i) == 0)
                    continue;

                hg.G.add(jacDepth(i) * err, ids(i));
                //(*hg).G[v_ids[i]] += J[i] * error;

                for (int j = i; j < 3; j++)
                {
                    float jj = jacDepth(i) * jacDepth(j);
                    hg.H.add(jj, ids(i), ids(j));
                    hg.H.add(jj, ids(j), ids(i));
                    //(*hg).H[v_ids[i]][v_ids[j]] += jj;
                    //(*hg).H[v_ids[j]][v_ids[i]] += jj;
                }
            }
        }
    }

    return hg;
}

HGMapped meshOptimizerCPU::computeHGPoseMap(frameCPU &frame, int frame_index, int lvl)
{
    HGMapped hg;

    j1_buffer.set(j1_buffer.nodata, lvl);
    j2_buffer.set(j2_buffer.nodata, lvl);
    j3_buffer.set(j3_buffer.nodata, lvl);
    error_buffer.set(error_buffer.nodata, lvl);
    id_buffer.set(id_buffer.nodata, lvl);

    renderer.renderJPoseMap(keyframeMesh, cam[lvl], keyframe, frame, j1_buffer, j2_buffer, j3_buffer, error_buffer, id_buffer, jacMethod, lvl);

    for (int y = 0; y < cam[lvl].height; y++)
    {
        for (int x = 0; x < cam[lvl].width; x++)
        {
            Eigen::Vector3f j_tra = j1_buffer.get(y, x, lvl);
            Eigen::Vector3f j_rot = j2_buffer.get(y, x, lvl);
            Eigen::Vector3f j_map = j3_buffer.get(y, x, lvl);
            float error = error_buffer.get(y, x, lvl);
            Eigen::Vector3i ids = id_buffer.get(y, x, lvl);

            if (j_tra == j1_buffer.nodata || j_rot == j2_buffer.nodata || j_map == j3_buffer.nodata || error == error_buffer.nodata || ids == id_buffer.nodata)
                continue;

            hg.count += 1;

            Eigen::Matrix<float, 6, 1> J_pose;
            J_pose << j_tra(0), j_tra(1), j_tra(2), j_rot(0), j_rot(1), j_rot(2);

            for (int i = 0; i < 6; i++)
            {
                hg.G.add(J_pose[i] * error, i - (frame.id + 1) * 6);

                for (int j = i; j < 6; j++)
                {
                    float jj = J_pose[i] * J_pose[j];
                    hg.H.add(jj, i - (frame.id + 1) * 6, j - (frame.id + 1) * 6);
                    hg.H.add(jj, j - (frame.id + 1) * 6, i - (frame.id + 1) * 6);
                }
            }

            for (int i = 0; i < 3; i++)
            {
                // if the jacobian is 0
                // we really cannot say anything about the depth
                // can make the hessian non-singular
                if (j_map(i) == 0)
                    continue;
                hg.G.add(j_map(i) * error, ids(i));
                //(*hg).G[v_ids[i]] += J[i] * error;

                for (int j = i; j < 3; j++)
                {
                    float jj = j_map(i) * j_map(j);
                    hg.H.add(jj, ids(i), ids(j));
                    hg.H.add(jj, ids(j), ids(i));
                    //(*hg).H[v_ids[i]][v_ids[j]] += jj;
                    //(*hg).H[v_ids[j]][v_ids[i]] += jj;
                }
            }
        }
    }

    return hg;
}

Error meshOptimizerCPU::errorRegu()
{
    Error error;

    std::vector<unsigned int> triIds = keyframeMesh.getTrianglesIds();
    // for each triangle
    for (size_t index = 0; index < triIds.size(); index++)
    {
        unsigned int id = triIds[index];
        std::array<unsigned int, 3> tri = keyframeMesh.getTriangleIndices(id);

        float theta[3];
        for (int j = 0; j < 3; j++)
        {
            if (jacMethod == MapJacobianMethod::depth)
                theta[j] = keyframeMesh.getVerticeDepth(tri[j]);
            if (jacMethod == MapJacobianMethod::idepth)
                theta[j] = 1.0 / keyframeMesh.getVerticeDepth(tri[j]);
            if (jacMethod == MapJacobianMethod::log_depth)
                theta[j] = std::log(keyframeMesh.getVerticeDepth(tri[j]));
            if (jacMethod == MapJacobianMethod::log_idepth)
                theta[j] = std::log(1.0 / keyframeMesh.getVerticeDepth(tri[j]));
        }

        if (theta[0] <= 0.0 || theta[1] <= 0.0 || theta[2] <= 0.0)
            continue;

        float diff1 = theta[0] - theta[1];
        float diff2 = theta[0] - theta[2];
        float diff3 = theta[1] - theta[2];

        error.error += diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
    }
    // divided by the number of triangles
    // we don't want to have less error if we have less triangles
    error.count = triIds.size();
    return error;
}

HGMapped meshOptimizerCPU::HGRegu()
{
    HGMapped hg;

    std::vector<unsigned int> triIds = keyframeMesh.getTrianglesIds();

    for (size_t i = 0; i < triIds.size(); i++)
    {
        unsigned int t_id = triIds[i];

        std::array<unsigned int, 3> v_ids = keyframeMesh.getTriangleIndices(t_id);

        float theta[3];
        for (int j = 0; j < 3; j++)
        {
            if (jacMethod == MapJacobianMethod::depth)
                theta[j] = keyframeMesh.getVerticeDepth(v_ids[j]);
            if (jacMethod == MapJacobianMethod::idepth)
                theta[j] = 1.0 / keyframeMesh.getVerticeDepth(v_ids[j]);
            if (jacMethod == MapJacobianMethod::log_depth)
                theta[j] = std::log(keyframeMesh.getVerticeDepth(v_ids[j]));
            if (jacMethod == MapJacobianMethod::log_idepth)
                theta[j] = std::log(1.0 / keyframeMesh.getVerticeDepth(v_ids[j]));
        }

        if (theta[0] <= 0.0 || theta[1] <= 0.0 || theta[2] <= 0.0)
            continue;

        float diff1 = theta[0] - theta[1];
        float diff2 = theta[0] - theta[2];
        float diff3 = theta[1] - theta[2];

        float J1[3] = {1.0, -1.0, 0.0};
        float J2[3] = {1.0, 0.0, -1.0};
        float J3[3] = {0.0, 1.0, -1.0};

        for (int j = 0; j < 3; j++)
        {
            // if (hg.G(NUM_FRAMES*6 + vertexIndex[j]) == 0)
            //     continue;
            hg.G[v_ids[j]] += (diff1 * J1[j] + diff2 * J2[j] + diff3 * J3[j]);
            for (int k = 0; k < 3; k++)
            {
                hg.H[v_ids[j]][v_ids[k]] += (J1[j] * J1[k] + J2[j] * J2[k] + J3[j] * J3[k]);
            }
        }
    }

    hg.count = triIds.size();

    return hg;
}

Error meshOptimizerCPU::errorInitial(MeshTexCoordsCPU &initMesh, MatrixMapped &initThetaVar)
{
    Error error;

    std::vector<unsigned int> vertsIds = keyframeMesh.getVerticesIds();

    for (size_t index = 0; index < vertsIds.size(); index++)
    {
        unsigned int id = vertsIds[index];

        float initVar = initThetaVar[id][id];
        float theta, initTheta;
        if (jacMethod == MapJacobianMethod::depth)
        {
            theta = keyframeMesh.getVerticeDepth(id);
            initTheta = initMesh.getVerticeDepth(id);
        }
        if (jacMethod == MapJacobianMethod::idepth)
        {
            theta = 1.0 / keyframeMesh.getVerticeDepth(id);
            initTheta = 1.0 / initMesh.getVerticeDepth(id);
        }
        if (jacMethod == MapJacobianMethod::log_depth)
        {
            theta = std::log(keyframeMesh.getVerticeDepth(id));
            initTheta = std::log(initMesh.getVerticeDepth(id));
        }
        if (jacMethod == MapJacobianMethod::log_idepth)
        {
            theta = std::log(1.0 / keyframeMesh.getVerticeDepth(id));
            initTheta = std::log(1.0 / initMesh.getVerticeDepth(id));
        }

        float diff = theta - initTheta;

        error.error += initVar * diff * diff;
    }
    // divided by the number of triangles
    // we don't want to have less error if we have less triangles
    error.count = vertsIds.size();
    return error;
}

HGMapped meshOptimizerCPU::HGInitial(MeshTexCoordsCPU &initMesh, MatrixMapped &initThetaVar)
{
    HGMapped hg;

    std::vector<unsigned int> vertsIds = keyframeMesh.getVerticesIds();

    for (size_t i = 0; i < vertsIds.size(); i++)
    {
        unsigned int v_id = vertsIds[i];

        float initVar = initThetaVar[v_id][v_id];
        float theta, initTheta;
        if (jacMethod == MapJacobianMethod::depth)
        {
            theta = keyframeMesh.getVerticeDepth(v_id);
            initTheta = initMesh.getVerticeDepth(v_id);
        }
        if (jacMethod == MapJacobianMethod::idepth)
        {
            theta = 1.0 / keyframeMesh.getVerticeDepth(v_id);
            initTheta = 1.0 / initMesh.getVerticeDepth(v_id);
        }
        if (jacMethod == MapJacobianMethod::log_depth)
        {
            theta = std::log(keyframeMesh.getVerticeDepth(v_id));
            initTheta = std::log(initMesh.getVerticeDepth(v_id));
        }
        if (jacMethod == MapJacobianMethod::log_idepth)
        {
            theta = std::log(1.0 / keyframeMesh.getVerticeDepth(v_id));
            initTheta = std::log(1.0 / initMesh.getVerticeDepth(v_id));
        }

        hg.G[v_id] += initVar * (theta - initTheta);
        hg.H[v_id][v_id] += initVar;
    }

    hg.count = vertsIds.size();

    return hg;
}

void meshOptimizerCPU::optPose(frameCPU &frame)
{
    int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

    tic_toc t;
    Error e;
    HGMapped hg;

    for (int lvl = 2; lvl >= 1; lvl--)
    {
        std::cout << "*************************lvl " << lvl << std::endl;
        t.tic();
        Sophus::SE3f best_pose = frame.pose;
        e.setZero();
        e = computeError(frame, lvl);
        float last_error = e.error / e.count;

        std::cout << "init error " << last_error << " time " << t.toc() << std::endl;

        for (int it = 0; it < maxIterations[lvl]; it++)
        {
            t.tic();
            hg.setZero();
            hg += computeHGPose(frame, lvl);

            std::vector<int> ids = hg.G.getIds();

            Eigen::VectorXf G = hg.G.toEigen(ids);
            Eigen::SparseMatrix<float> H = hg.H.toEigen(ids);

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

                Eigen::Matrix<float, 6, 1> inc = H_lambda.ldlt().solve(G);

                // Sophus::SE3f new_pose = frame.pose * Sophus::SE3f::exp(inc_pose);
                frame.pose = best_pose * Sophus::SE3f::exp(inc).inverse();
                // Sophus::SE3f new_pose = Sophus::SE3f::exp(inc_pose).inverse() * frame.pose;

                t.tic();
                e.setZero();
                e = computeError(frame, lvl);
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

void meshOptimizerCPU::optMapDepth(std::vector<frameCPU> &frames)
{
    tic_toc t;

    Error e;
    Error e_regu;
    Error e_init;

    HGMapped hg;
    HGMapped hg_regu;
    HGMapped hg_init;

    for (int lvl = 1; lvl >= 1; lvl--)
    {
        t.tic();
        // keyframeMesh.toRayIdepth();
        // std::vector<float> best_idepths = keyframeMesh.getVerticesIdepths();

        MeshTexCoordsCPU initialMesh = keyframeMesh;
        MatrixMapped initialInvVar = invVar;

        e.setZero();
        for (std::size_t i = 0; i < frames.size(); i++)
            e += computeError(frames[i], lvl);
        e.error /= e.count;

        e_regu = errorRegu();
        e_regu.error /= e_regu.count;

        e_init = errorInitial(initialMesh, initialInvVar);
        e_init.error /= e_init.count;

        float last_error = e.error + meshRegularization * e_regu.error + meshInitial * e_init.error;

        std::cout << "--------lvl " << lvl << " initial error " << last_error << " " << t.toc() << std::endl;

        int maxIterations = 100;
        float lambda = 0.0;
        for (int it = 0; it < maxIterations; it++)
        {
            t.tic();

            hg.setZero();
            for (std::size_t i = 0; i < frames.size(); i++)
                hg += computeHGMapDepth(frames[i], lvl);

            std::vector<int> ids = hg.G.getIds();

            Eigen::VectorXf G = hg.G.toEigen(ids);
            Eigen::SparseMatrix<float> H = hg.H.toEigen(ids);

            H /= hg.count;
            G /= hg.count;

            hg_regu = HGRegu();

            Eigen::VectorXf G_regu = hg_regu.G.toEigen(ids);
            Eigen::SparseMatrix<float> H_regu = hg_regu.H.toEigen(ids);

            H_regu /= hg_regu.count;
            G_regu /= hg_regu.count;

            hg_init = HGInitial(initialMesh, initialInvVar);

            Eigen::VectorXf G_init = hg_init.G.toEigen(ids);
            Eigen::SparseMatrix<float> H_init = hg_init.H.toEigen(ids);

            H_init /= hg_init.count;
            G_init /= hg_init.count;

            H += meshRegularization * H_regu + meshInitial * H_init;
            G += meshRegularization * G_regu + meshInitial * G_init;

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

                std::vector<float> best_depths;
                for (int index = 0; index < (int)ids.size(); index++)
                {
                    // float best_idepth = keyframeMesh.getVerticeIdepth(ids[index]);
                    // float new_idepth = best_idepth + inc(index);
                    // if (new_idepth < 0.001 || new_idepth > 100.0)
                    //     new_idepth = best_idepth;
                    // best_idepths.push_back(best_idepth);
                    // keyframeMesh.setVerticeIdepth(new_idepth, ids[index]);

                    float best_depth = keyframeMesh.getVerticeDepth(ids[index]);
                    float new_depth;
                    if (jacMethod == MapJacobianMethod::depth)
                        new_depth = best_depth + inc(index);
                    if (jacMethod == MapJacobianMethod::idepth)
                        new_depth = 1.0 / (1.0 / best_depth + inc(index));
                    if (jacMethod == MapJacobianMethod::log_depth)
                        new_depth = std::exp(std::log(best_depth) + inc(index));
                    if (jacMethod == MapJacobianMethod::log_idepth)
                        new_depth = 1.0 / std::exp(std::log(1.0 / best_depth) + inc(index));
                    if (new_depth < 0.0001 || new_depth > 1000.0)
                        new_depth = best_depth;
                    best_depths.push_back(best_depth);
                    keyframeMesh.setVerticeDepth(new_depth, ids[index]);
                }

                t.tic();

                e.setZero();
                for (std::size_t i = 0; i < frames.size(); i++)
                    e += computeError(frames[i], lvl);
                e.error /= e.count;

                e_regu = errorRegu();
                e_regu.error /= e_regu.count;

                e_init = errorInitial(initialMesh, initialInvVar);
                e_init.error /= e_init.count;

                float error = e.error + meshRegularization * e_regu.error + meshInitial * e_init.error;

                std::cout << "new error " << error << " " << t.toc() << std::endl;

                if (error < last_error)
                {
                    invVar = hg.H;

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
                    for (int index = 0; index < ids.size(); index++)
                        keyframeMesh.setVerticeDepth(best_depths[index], ids[index]);

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

void meshOptimizerCPU::optPoseMap(std::vector<frameCPU> &frames)
{
    tic_toc t;

    Error e;
    Error e_regu;
    HGMapped hg;
    HGMapped hg_regu;

    for (int lvl = 1; lvl >= 1; lvl--)
    {
        t.tic();

        e.setZero();
        for (std::size_t i = 0; i < frames.size(); i++)
            e += computeError(frames[i], lvl);
        e.error /= e.count;

        e_regu = errorRegu();
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
                hg += computeHGPoseMap(frames[i], i, lvl);

            std::vector<int> ids = hg.G.getIds();

            Eigen::VectorXf G = hg.G.toEigen(ids);
            Eigen::SparseMatrix<float> H = hg.H.toEigen(ids);

            H /= hg.count;
            G /= hg.count;

            hg_regu = HGRegu();

            Eigen::VectorXf G_regu = hg_regu.G.toEigen(ids);
            Eigen::SparseMatrix<float> H_regu = hg_regu.H.toEigen(ids);

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
                        int index = j - (i + 1) * 6 + frames.size() * 6;
                        pose_inc(j) = inc(index);
                    }
                    best_poses.push_back(frames[i].pose);
                    frames[i].pose = frames[i].pose * Sophus::SE3f::exp(pose_inc).inverse();
                }

                // update map
                std::map<unsigned int, float> best_depths;
                for (int index = 0; index < (int)ids.size(); index++)
                {
                    // negative ids are for the poses
                    if (ids[index] < 0)
                        continue;

                    float best_depth = keyframeMesh.getVerticeDepth(ids[index]);
                    float new_depth;
                    if (jacMethod == MapJacobianMethod::depth)
                        new_depth = best_depth + inc(index);
                    if (jacMethod == MapJacobianMethod::idepth)
                        new_depth = 1.0 / (1.0 / best_depth + inc(index));
                    if (jacMethod == MapJacobianMethod::log_depth)
                        new_depth = std::exp(std::log(best_depth) + inc(index));
                    if (jacMethod == MapJacobianMethod::log_idepth)
                        new_depth = 1.0 / std::exp(std::log(1.0 / best_depth) + inc(index));
                    if (new_depth < 0.001 || new_depth > 100.0)
                        new_depth = best_depth;
                    best_depths[ids[index]] = best_depth;
                    keyframeMesh.setVerticeDepth(new_depth, ids[index]);
                }
                // keyframeMesh.setVerticesIdepths(new_idepths);

                t.tic();

                e.setZero();
                for (std::size_t i = 0; i < frames.size(); i++)
                    e += computeError(frames[i], lvl);
                e.error /= e.count;

                e_regu = errorRegu();
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
                    for (int index = 0; index < frames.size(); index++)
                        frames[index].pose = best_poses[index];

                    for (int index = 0; index < ids.size(); index++)
                    {
                        // negative ids are for the poses
                        if (ids[index] < 0)
                            continue;
                        keyframeMesh.setVerticeDepth(best_depths[ids[index]], ids[index]);
                    }

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