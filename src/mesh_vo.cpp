#include "mesh_vo.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/OrderingMethods>
#include <Eigen/IterativeLinearSolvers>

#include <map>

meshVO::meshVO(float _fx, float _fy, float _cx, float _cy, int _width, int _height)
    : cam(_fx, _fy, _cx, _cy, _width, _height)
{
    scene.initWithRandomIdepth(cam);
}

void meshVO::visualOdometry(cv::Mat _frame)
{
    for (int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(_frame, lastframe.image.texture[lvl], cv::Size(cam.width[lvl], cam.height[lvl]), cv::INTER_LANCZOS4);
        cpu.computeFrameDerivative(lastframe, cam, lvl);
    }

    if(!keyframe.init)
    {
        lastframe.copyTo(keyframe);
        return;
    }

    tic_toc t;
    t.tic();
    // frameData.pose = _globalPose*keyframeData.pose.inverse();
    optPose(lastframe); //*Sophus::SE3f::exp(inc_pose).inverse());

    std::cout << "estimated pose " << std::endl;
    std::cout << lastframe.pose.matrix() << std::endl;
    std::cout << "clacPose time " << t.toc() << std::endl;

    {
        //optPoseMapJoint();
    }
}

void meshVO::localization(cv::Mat frame)
{
    for (int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(frame, lastframe.image.texture[lvl], cv::Size(cam.width[lvl], cam.height[lvl]), cv::INTER_LANCZOS4);
        cpu.computeFrameDerivative(lastframe, cam, lvl);
    }

    cpu.computeFrameIdepth(lastframe, cam, scene, 1);
    lastframe.image.show("lastframe image", 1);
    lastframe.idepth.show("lastframe idepth", 1);

    // frameData.pose = _globalPose*keyframeData.pose.inverse();
    //optPose(lastframe); //*Sophus::SE3f::exp(inc_pose).inverse());
}

void meshVO::mapping(cv::Mat _frame, Sophus::SE3f _globalPose)
{
    tic_toc t;

    t.tic();

    for (int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(_frame, lastframe.image.texture[lvl], cv::Size(cam.width[lvl], cam.height[lvl]), cv::INTER_LANCZOS4);
        cpu.computeFrameDerivative(lastframe, cam, lvl);
    }

    std::cout << "save frame time " << t.toc() << std::endl;

    lastframe.pose = _globalPose; //*keyframeData.pose.inverse();

    t.tic();
    // optMapVertex();
    // optMapJoint();
    optPoseMap();
    std::cout << "update map time " << t.toc() << std::endl;
}

void meshVO::optPose(frameCpu &frame)
{
    int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

    Sophus::SE3f bestPose = frame.pose;

    tic_toc t;

    for (int lvl = 4; lvl >= 1; lvl--)
    {
        std::cout << "*************************lvl " << lvl << std::endl;
        t.tic();
        float last_error = cpu.computeError(frame, keyframe, cam, lvl);

        std::cout << "init error " << last_error << " time " << t.toc() << std::endl;

        for (int it = 0; it < maxIterations[lvl]; it++)
        {
            t.tic();
            HGPose hgpose = cpu.computeHGPose(frame, keyframe, cam, lvl);
            std::cout << "HGPose time " << t.toc() << std::endl;

            float lambda = 0.0;
            int n_try = 0;
            while (true)
            {
                Eigen::Matrix<float, 6, 6> acc_H_pose_lambda;
                acc_H_pose_lambda = hgpose.H_pose;

                for (int j = 0; j < 6; j++)
                    acc_H_pose_lambda(j, j) *= 1.0 + lambda;

                Eigen::Matrix<float, 6, 1> inc_pose = acc_H_pose_lambda.ldlt().solve(hgpose.G_pose);

                Sophus::SE3f new_pose = bestPose * Sophus::SE3f::exp(inc_pose).inverse();

                t.tic();
                float error = cpu.computeError(frame, keyframe, cam, lvl);
                std::cout << "new error time " << t.toc() << std::endl;

                if (error < last_error)
                {
                    // accept update, decrease lambda
                    bestPose = new_pose;

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
                    n_try++;

                    if (lambda < 0.2f)
                        lambda = 0.2f;
                    else
                        lambda *= std::pow(2.0, n_try);

                    // reject update, increase lambda, use un-updated data
                    // std::cout << "update rejected " << std::endl;

                    if (!(inc_pose.dot(inc_pose) > 1e-8))
                    // if(!(inc.dot(inc) > 1e-6))
                    {
                        // std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations and " << t_try << " total tries, with lambda " << lambda << std::endl;
                        // if too small, do next level!
                        it = maxIterations[lvl];
                        break;
                    }
                }
            }

            frame.pose = bestPose;
        }
    }
}


void meshVO::optMap()
{
    tic_toc t;
    for (int lvl = 1; lvl >= 1; lvl--)
    {
        t.tic();
        std::vector<float> best_vertices;
        best_vertices = scene.scene_vertices;

        float last_error = cpu.computeError(lastframe, keyframe, cam, lvl); // + errorMesh();

        std::cout << "init error time " << t.toc() << std::endl;
        std::cout << "lvl " << lvl << " initial error " << last_error << std::endl;

        int maxIterations = 100;
        float lambda = 0.0;
        for (int it = 0; it < maxIterations; it++)
        {
            t.tic();

            //cpu.computeHGMap();
            //HJMapStackGPU(lvl);
            // HJMesh();

            std::cout << "HJ time " << t.toc() << std::endl;

            int n_try = 0;
            while (true)
            {
                Eigen::SparseMatrix<float> H_depth_lambda = H_depth;

                for (int j = 0; j < H_depth_lambda.rows(); j++)
                {
                    H_depth_lambda.coeffRef(j, j) *= (1.0 + lambda);
                }

                t.tic();

                H_depth_lambda.makeCompressed();
                // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > solver;
                Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
                // Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int> > solver;
                // Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > solver;
                solver.analyzePattern(H_depth_lambda);
                // std::cout << solver.info() << std::endl;
                solver.factorize(H_depth_lambda);
                // std::cout << solver.lastErrorMessage() << std::endl;
                inc_depth = -solver.solve(J_depth);
                // inc_depth = -acc_H_depth_lambda.llt().solve(acc_J_depth);
                // inc_depth = - acc_H_depth_lambda.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(acc_J_depth);
                // inc_depth = -acc_H_depth_lambda.colPivHouseholderQr().solve(acc_J_depth);

                /*
                for(int j = 0; j < int(J_depth.size()); j++)
                {
                    float h = acc_H_depth_lambda.coeffRef(j,j);
                    if(h > 0.0)
                    //if(abs(J_depth(j)) > 0.0)
                    {
                        inc_depth(j) = -J_depth(j)/h;
                        //inc_depth(j) = -(1.0/(1.0+lambda))*J_depth(j)/fabs(J_depth(j));
                        //std::cout << "update" << std::endl;
                    }
                }
*/
                std::cout << "solve time " << t.toc() << std::endl;

                t.tic();

                /*
                for (int index = 0; index < VERTEX_HEIGH * VERTEX_WIDTH; index++)
                {
                    if (inc_depth(index) != inc_depth(index))
                    {
                        std::cout << "some nand in inc_depth " << std::endl;
                        continue;
                    }
                    scene.scene_vertices[index * 3 + 2] = best_vertices[index * 3 + 2] + inc_depth(index);
                    if (scene.scene_vertices[index * 3 + 2] < min_idepth || scene.scene_vertices[index * 3 + 2] > max_idepth)
                        scene.scene_vertices[index * 3 + 2] = best_vertices[index * 3 + 2];
                }
                */

                std::cout << "set data time " << t.toc() << std::endl;

                t.tic();

                float error = cpu.computeError(lastframe, keyframe, cam, lvl); // + errorMesh();

                std::cout << "new error time " << t.toc() << std::endl;

                std::cout << "lvl " << lvl << " new error " << error << std::endl;

                if (error < last_error)
                {
                    // accept update, decrease lambda
                    std::cout << "update accepted " << std::endl;

                    best_vertices = scene.scene_vertices;

                    float p = error / last_error;

                    if (lambda < 0.2f)
                        lambda = 0.0f;
                    else
                        lambda *= 0.5;

                    last_error = error;

                    if (p > 0.999f)
                    {
                        std::cout << "lvl " << lvl << " converged after " << it << " itarations with lambda " << lambda << std::endl;
                        // if converged, do next level
                        it = maxIterations;
                    }

                    // if update accepted, do next iteration
                    break;
                }
                else
                {
                    n_try++;

                    if (lambda < 0.2f)
                        lambda = 0.2f;
                    else
                        lambda *= 2.0; // std::pow(2.0, n_try);

                    scene.scene_vertices = best_vertices;

                    // reject update, increase lambda, use un-updated data
                    std::cout << "update rejected " << std::endl;

                    if (inc_depth.dot(inc_depth) < 1e-8)
                    {
                        std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations with lambda " << lambda << std::endl;
                        // if too small, do next level!
                        it = maxIterations;
                        break;
                    }
                }
            }
        }
    }
}


void meshVO::optPoseMap()
{
    tic_toc t;

    for (int lvl = 0; lvl >= 0; lvl--)
    {
        t.tic();

        std::vector<float> best_vertices;

        best_vertices = scene.scene_vertices;

        float last_error = cpu.computeError(lastframe, keyframe, cam, lvl); // + errorMesh();
        std::cout << "initial error time " << t.toc() << std::endl;
        std::cout << "lvl " << lvl << " initial error " << last_error << std::endl;

        int maxIterations = 100;
        float lambda = 0.0;

        for (int it = 0; it < maxIterations; it++)
        {
            t.tic();

            //HJPoseMapStackGPU(lvl);
            //HJMesh();

            std::cout << "HJ time " << t.toc() << std::endl;

            int n_try = 0;
            while (true)
            {
                t.tic();

                Eigen::MatrixXf H_joint_lambda = H_joint;

                for (int j = 0; j < H_joint_lambda.rows(); j++)
                {
                    H_joint_lambda(j, j) *= (1.0 + lambda);
                }

                // inc_joint = H_joint_lambda.ldlt().solve(J_joint);
                // inc_joint = H_joint_lambda.colPivHouseholderQr().solve(J_joint);

                for (int j = 0; j < int(J_joint.size()); j++)
                {
                    float h = H_joint_lambda(j, j);
                    if (h > 0.0 && abs(J_joint(j)) > 0.0)
                    // if(J_joint(j) > 0.0)
                    {
                        inc_joint(j) = J_joint(j) / h;
                        // inc_joint(j) = (1.0/(1.0+lambda))*J_joint(j)/fabs(J_joint(j));
                        // std::cout << "update" << std::endl;
                    }
                }

                std::cout << "solve time " << t.toc() << std::endl;

                t.tic();

                /*
                for (int i = 0; i < MAX_FRAMES; i++)
                {
                    if (frameDataStack[i].init == true)
                    {
                        Eigen::VectorXf inc_pose(6);
                        bool good = true;
                        for (int j = 0; j < 6; j++)
                        {
                            if (std::isnan(inc_joint(i * 6 + j)) || std::isinf(inc_joint(i * 6 + j)))
                            {
                                std::cout << "nand in inc_joint pose part" << std::endl;
                                good = false;
                            }
                            inc_pose(j) = inc_joint(i * 6 + j);
                        }
                        if (good == false)
                        {
                            continue;
                        }
                        frameDataStack[i].pose = ((bestPoses[i] * keyframeData.pose.inverse()) * Sophus::SE3f::exp(inc_pose).inverse()) * keyframeData.pose;
                        // frameDataStack[i].pose = bestPoses[i]*Sophus::SE3f::exp(inc_pose).inverse();
                    }
                }
                */

                /*
                for (int index = 0; index < VERTEX_HEIGH * VERTEX_WIDTH; index++)
                {
                    if (std::isnan(inc_joint(MAX_FRAMES * 6 + index)))
                    {
                        std::cout << "nand in inc_joint depth part " << std::endl;
                        continue;
                    }
                    scene_vertices[index * 3 + 2] = best_vertices[index * 3 + 2] - inc_joint(MAX_FRAMES * 6 + index);
                    if (scene_vertices[index * 3 + 2] < min_idepth || scene_vertices[index * 3 + 2] > max_idepth)
                        scene_vertices[index * 3 + 2] = best_vertices[index * 3 + 2];
                }
                */

                std::cout << "set data time " << t.toc() << std::endl;

                t.tic();

                //float error = errorStackGPU(lvl);// + errorMesh();

                std::cout << "new error time " << t.toc() << std::endl;
                //std::cout << "lvl " << lvl << " new error " << error << std::endl;

                //if (error < last_error)
                if(true)
                {
                    // accept update, decrease lambda
                    std::cout << "update accepted " << std::endl;

                    /*
                    for (int i = 0; i < MAX_FRAMES; i++)
                        bestPoses[i] = frameDataStack[i].pose;
                    best_vertices = scene_vertices;
                    */

                    float p = 0.0;//error / last_error;

                    if (lambda < 0.2f)
                        lambda = 0.0f;
                    else
                        lambda *= 0.5;

                    last_error = 0.0;//error;

                    if (p > 0.999f)
                    {
                        std::cout << "lvl " << lvl << " converged after " << it << " itarations with lambda " << lambda << std::endl;
                        // if converged, do next level
                        it = maxIterations;
                    }

                    // if update accepted, do next iteration
                    break;
                }
                else
                {
                    n_try++;

                    if (lambda < 0.2f)
                        lambda = 0.2f;
                    else
                        lambda *= 2.0; // std::pow(2.0, n_try);

                    /*
                    for (int i = 0; i < MAX_FRAMES; i++)
                        frameDataStack[i].pose = bestPoses[i];
                    scene_vertices = best_vertices;
                    */
                    // reject update, increase lambda, use un-updated data
                    std::cout << "update rejected " << lambda << " " << inc_joint.dot(inc_joint) << std::endl;

                    if (inc_joint.dot(inc_joint) < 1e-16)
                    {
                        std::cout << "lvl " << lvl << " inc size too small, after " << it << " itarations with lambda " << lambda << std::endl;
                        // if too small, do next level!
                        it = maxIterations;
                        break;
                    }
                }
            }
        }
    }
}


float meshVO::errorMesh()
{
    float error = 0;
    for (int i = 0; i < int(scene.scene_indices.size()); i += 3)
    {
        // std::cout << "triangle" << std::endl;
        float idepth[3];
        for (int j = 0; j < 3; j++)
        {
            int vertexIndex = scene.scene_indices.at(i + j);
            idepth[j] = scene.scene_vertices.at(vertexIndex * 3 + 2);
        }
        float diff1 = idepth[0] - idepth[1];
        float diff2 = idepth[0] - idepth[2];
        float diff3 = idepth[1] - idepth[2];

        error += diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
    }
    return MESH_REGU * error / (VERTEX_HEIGH * VERTEX_WIDTH);
    /*
    float error = 0;
    for(int i = 0; i < int(scene_indices.size()); i+=3)
    {
        //std::cout << "triangle" << std::endl;
        Eigen::Vector3f vertex[3];
        for(int j = 0; j < 3; j++)
        {
            int vertexIndex = scene_indices.at(i+j);
            float rx = scene_vertices.at(vertexIndex*3+0);
            float ry = scene_vertices.at(vertexIndex*3+1);
            float rz = scene_vertices.at(vertexIndex*3+2);
            vertex[j] = Eigen::Vector3f(rx,ry,1.0)/rz;
            //std::cout << "vertex " << j << " " << vertex[j] << std::endl;
            if(vertex[j](0) != vertex[j](0) || vertex[j](1) != vertex[j](1) || vertex[j](2) != vertex[j](2))
            {
                std::cout << "some nand " << std::endl;
                std::cout << i << std::endl;
                std::cout << j << std::endl;
                std::cout << vertexIndex << std::endl;
                std::cout << rx << " " << ry << " " << rz << std::endl;
                std::cout << vertex[j] << std::endl;
            }
        }
        Eigen::Vector3f diff1 = vertex[0]-vertex[1];
        Eigen::Vector3f diff2 = vertex[0]-vertex[2];
        Eigen::Vector3f diff3 = vertex[1]-vertex[2];

        error += diff1.dot(diff1) + diff2.dot(diff2) + diff3.dot(diff3);
    }
    return 10.0*error/(VERTEX_HEIGH*VERTEX_WIDTH);
    */
}

void meshVO::addHGMesh()
{
    for (int i = 0; i < int(scene.scene_indices.size()); i += 3)
    {
        int vertexIndex[3];
        float idepth[3];
        for (int j = 0; j < 3; j++)
        {
            vertexIndex[j] = scene.scene_indices.at(i + j);
            idepth[j] = scene.scene_vertices.at(vertexIndex[j] * 3 + 2);
        }
        float diff1 = idepth[0] - idepth[1];
        float diff2 = idepth[0] - idepth[2];
        float diff3 = idepth[1] - idepth[2];

        float J1[3] = {1.0, -1.0, 0.0};
        float J2[3] = {1.0, 0.0, -1.0};
        float J3[3] = {0.0, 1.0, -1.0};

        for (int j = 0; j < 3; j++)
        {
            J_depth(vertexIndex[j]) += (MESH_REGU / (VERTEX_HEIGH * VERTEX_WIDTH)) * (diff1 * J1[j] + diff2 * J2[j] + diff3 * J3[j]);
            J_joint(MAX_FRAMES * 6 + vertexIndex[j]) += (MESH_REGU / (VERTEX_HEIGH * VERTEX_WIDTH)) * (diff1 * J1[j] + diff2 * J2[j] + diff3 * J3[j]);

            for (int k = 0; k < 3; k++)
            {
                H_depth.coeffRef(vertexIndex[j], vertexIndex[k]) += (MESH_REGU / (VERTEX_WIDTH * VERTEX_HEIGH)) * (J1[j] * J1[k] + J2[j] * J2[k] + J3[j] * J3[k]);
                H_joint(MAX_FRAMES * 6 + vertexIndex[j], MAX_FRAMES * 6 + vertexIndex[k]) += (MESH_REGU / (VERTEX_WIDTH * VERTEX_HEIGH)) * (J1[j] * J1[k] + J2[j] * J2[k] + J3[j] * J3[k]);
            }
        }
    }
    /*
    for(int i = 0; i < int(scene_indices.size()); i+=3)
    {
        int vertexIndex[3];
        Eigen::Vector3f vertex[3];
        Eigen::Vector3f J[3];
        for(int j = 0; j < 3; j++)
        {
            vertexIndex[j] = scene_indices.at(i+j);
            float rx = scene_vertices.at(vertexIndex[j]*3+0);
            float ry = scene_vertices.at(vertexIndex[j]*3+1);
            float rz = scene_vertices.at(vertexIndex[j]*3+2);
            vertex[j] = Eigen::Vector3f(rx,ry,1.0)/rz;
            J[j] = Eigen::Vector3f(-rx/(rz*rz),-ry*(rz*rz),-1.0/(rz*rz));
        }
        for(int j = 0; j < 3; j++)
        {
            Eigen::Vector3f diff1;
            Eigen::Vector3f diff2;
            if(j == 0)
            {
                diff1 = vertex[0] - vertex[1];
                diff2 = vertex[0] - vertex[2];
            }
            if(j == 1)
            {
                diff1 = vertex[1] - vertex[0];
                diff2 = vertex[1] - vertex[2];
            }
            if(j == 2)
            {
                diff1 = vertex[2] - vertex[0];
                diff2 = vertex[2] - vertex[1];
            }
            J_depth(vertexIndex[j]) += 0.0*(diff1+diff2).dot(J[j]);
            for(int k = 0; k < 3; k++)
                H_depth.coeffRef(vertexIndex[j],vertexIndex[k]) = 0.0*1.0*J[j].dot(J[k]);
        }
    }
    */
}
