#include "scene/sceneCPU.h"

sceneCPU::sceneCPU(float fx, float fy, float cx, float cy, int width, int height)
    : cam(fx, fy, cx, cy, width, height)
{

}

void sceneCPU::init(frameCPU &f)
{
    f.copyTo(keyframe);
}

float sceneCPU::computeError(frameCPU &frame, int lvl)
{

    HGPose hgpose = errorPerIndexCPU(frame, lvl, 0, cam.height[lvl]);
    //  float error = treadReducer.reduce(std::bind(&mesh_vo::errorCPUPerIndex, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), _frame, lvl, 0, height[lvl]);
    
    if(hgpose.count > 0)
        hgpose.error /= hgpose.count;

    return hgpose.error;
}

HGPose errorPerIndex(frameCPU &frame, int lvl, int ymin, int ymax)
{
    HGPose hgpose;

    Sophus::SE3f relativePose = frame.pose; // * keyframe.pose.inverse();

    for (int y = ymin; y < ymax; y++)
        for (int x = 0; x < cam.width[lvl]; x++)
        {
            float vkf = float(keyframe.image.get(y, x, lvl));
            float keyframeId = keyframe.idepth.get(y, x, lvl);

            if (vkf == keyframe.image.nodata || keyframeId == keyframe.idepth.nodata)
                continue;

            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(cam.fxinv[lvl] * (x + 0.5) + cam.cxinv[lvl], cam.fyinv[lvl] * (y + 0.5) + cam.cyinv[lvl], 1.0) / keyframeId;
            Eigen::Vector3f pointFrame = relativePose * poinKeyframe;

            if (pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector2f pixelFrame = Eigen::Vector2f(cam.fx[lvl] * pointFrame(0) / pointFrame(2) + cam.cx[lvl], cam.fy[lvl] * pointFrame(1) / pointFrame(2) + cam.cy[lvl]);

            if (pixelFrame(0) < 0.0 || pixelFrame(0) >= cam.width[lvl] || pixelFrame(1) < 0 || pixelFrame(1) >= cam.height[lvl])
                continue;

            float vf = float(frame.image.get(pixelFrame(1), pixelFrame(0), lvl));

            if (vf == frame.image.nodata)
                continue;

            float residual = vf - vkf;
            float error = residual * residual;

            hgpose.error += error;
            hgpose.count++;
        }

    return hgpose;
}

dataCPU<float> computeIdepth(frameCPU &frame, int lvl)
{
    dataCPU<float> frameIdepth;
    idepthPerIndex(frame, frameIdepth, lvl, 0, cam.height[lvl]);
    return frameIdepth;
}

void idepthPerIndex(frameCPU &frame, frameCPU &frameIdepth, int lvl, int ymin, int ymax)
{
    Sophus::SE3f relativePose = frame.pose * keyframe.pose.inverse();

    for (int y = ymin; y < ymax; y++)
        for (int x = 0; x < cam.width[lvl]; x++)
        {
            float vkf = float(keyframe.image.get(y, x, lvl));
            float keyframeId = keyframeIdepth.get(y, x, lvl);

            if (vkf == keyframe.image.nodata || keyframeId == keyframe.idepth.nodata)
                continue;

            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(cam.fxinv[lvl] * (x + 0.5) + cam.cxinv[lvl], cam.fyinv[lvl] * (y + 0.5) + cam.cyinv[lvl], 1.0) / keyframeId;
            Eigen::Vector3f pointFrame = relativePose * poinKeyframe;

            if (pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector2f pixelFrame = Eigen::Vector2f(cam.fx[lvl] * pointFrame(0) / pointFrame(2) + cam.cx[lvl], cam.fy[lvl] * pointFrame(1) / pointFrame(2) + cam.cy[lvl]);

            if (pixelFrame(0) < 0.0 || pixelFrame(0) >= cam.width[lvl] || pixelFrame(1) < 0 || pixelFrame(1) >= cam.height[lvl])
                continue;

            frameIdpeht.set(1.0/pointFrame(2), pixelFrame(1), pixelFrame(0), lvl);
        }
}

HGPose computeHGPose(frameCPU &frame, int lvl)
{
    HGPose hgpose = HGPosePerIndex(frame, keyframe, cam, lvl, 0, cam.height[lvl]);
    //  HJPose _hjpose = treadReducer.reduce(std::bind(&mesh_vo::HJPoseCPUPerIndex, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), _frame, lvl, 0, height[lvl]);
    if(hgpose.count > 0)
    {
        hgpose.H_pose /= hgpose.count;
        hgpose.G_pose /= hgpose.count;
        hgpose.error /= hgpose.count;
    }
    return hgpose;
}

HGPose HGPosePerIndex(frameCPU &frame, int lvl, int ymin, int ymax)
{
    HGPose hgpose;

    Sophus::SE3f relativePose = frame.pose; // * keyframe.pose.inverse();

    for (int y = ymin; y < ymax; y++)
        for (int x = 0; x < cam.width[lvl]; x++)
        {
            float vkf = float(keyframe.image.get(y, x, lvl));
            float keyframeId = keyframe.idepth.get(y, x, lvl);
            if (vkf == keyframe.image.nodata || keyframeId == keyframe.idepth.nodata)
                continue;

            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(cam.fxinv[lvl] * x + cam.cxinv[lvl], cam.fyinv[lvl] * y + cam.cyinv[lvl], 1.0) / keyframeId;
            Eigen::Vector3f pointFrame = relativePose * poinKeyframe;

            if (pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector2f pixelFrame = Eigen::Vector2f(cam.fx[lvl] * pointFrame(0) / pointFrame(2) + cam.cx[lvl], cam.fy[lvl] * pointFrame(1) / pointFrame(2) + cam.cy[lvl]);

            if (pixelFrame(0) < 0.0 || pixelFrame(0) >= cam.width[lvl] || pixelFrame(1) < 0.0 || pixelFrame(1) >= cam.height[lvl])
                continue;

            float vf = float(frame.image.get(pixelFrame(1), pixelFrame(0), lvl));
            float dx = frame.dx.get(pixelFrame(1), pixelFrame(0), lvl);
            float dy = frame.dy.get(pixelFrame(1), pixelFrame(0), lvl);

            if (vf == frame.image.nodata || dx == frame.dx.nodata || dy == frame.dy.nodata)
                continue;

            Eigen::Vector2f d_f_d_uf(dx, dy);

            float id = 1.0 / pointFrame(2);

            // z-buffer
            float l_idepth = frame.idepth.get(pixelFrame(1), pixelFrame(0), lvl);
            if (l_idepth > id && l_idepth != frame.idepth.nodata)
                continue;

            float v0 = d_f_d_uf(0) * cam.fx[lvl] * id;
            float v1 = d_f_d_uf(1) * cam.fy[lvl] * id;
            float v2 = -(v0 * pointFrame(0) + v1 * pointFrame(1)) * id;

            Eigen::Vector3f d_I_d_tra = Eigen::Vector3f(v0, v1, v2);
            Eigen::Vector3f d_I_d_rot = Eigen::Vector3f(-pointFrame(2) * v1 + pointFrame(1) * v2, pointFrame(2) * v0 - pointFrame(0) * v2, -pointFrame(1) * v0 + pointFrame(0) * v1);

            float residual = vf - vkf;
            float residual_2 = residual * residual;

            hgpose.error += residual_2;

            Eigen::Matrix<float, 6, 1> J;
            J << d_I_d_tra(0), d_I_d_tra(1), d_I_d_tra(2), d_I_d_rot(0), d_I_d_rot(1), d_I_d_rot(2);

            hgpose.count++;
            for (int i = 0; i < 6; i++)
            {
                hgpose.G_pose(i) += J[i] * residual;
                for (int j = i; j < 6; j++)
                {
                    float jj = J[i] * J[j];
                    hgpose.H_pose(i, j) += jj;
                    hgpose.H_pose(j, i) += jj;
                }
            }
        }

    return hgpose;
}

void sceneCPU::optMap(frameCPU &frame)
{
    tic_toc t;
    for (int lvl = 1; lvl >= 1; lvl--)
    {
        t.tic();
        std::vector<float> best_vertices = scene_vertices;

        float last_error = computeError(frame, lvl);
        //last_error += scene.errorRegu();

        std::cout << "--------lvl " << lvl << " initial error " << last_error << " " << t.toc() << std::endl;

        int maxIterations = 5;
        float lambda = 0.0;
        for (int it = 0; it < maxIterations; it++)
        {
            t.tic();

            HGMap hgmap = computeHGMap(frame, lvl);
            //scene.HGRegu(hgmap);
            // HJMapStackGPU(lvl);
            //  HJMesh();

            //check that the hessian is nonsingular
            //if it is "fix" it
            for (int i = 0; i < hgmap.G_depth.size(); i++)
            {
                int gcount = hgmap.G_count(i);
                if (gcount == 0)
                    hgmap.H_depth.coeffRef(i, i) = 1.0;
                // else
                //     hgmap.G_depth(i) /= float(gcount);
            }

            std::cout << "HG time " << t.toc() << std::endl;

            int n_try = 0;
            while (true)
            {
                Eigen::SparseMatrix<float> H_depth_lambda = hgmap.H_depth;

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
                if (solver.info() != Eigen::Success)
                {
                    // some problem i have still to debug
                    it = maxIterations;
                    break;
                }
                // std::cout << solver.lastErrorMessage() << std::endl;
                Eigen::VectorXf inc_depth = -solver.solve(hgmap.G_depth);
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

                std::vector<float> new_vertices = best_vertices;

                for (int index = 0; index < VERTEX_HEIGH * VERTEX_WIDTH; index++)
                {
                    if (inc_depth(index) != inc_depth(index))
                    {
                        std::cout << "some nand in inc_depth " << std::endl;
                        continue;
                    }
                    new_vertices[index * 3 + 2] = best_vertices[index * 3 + 2] + inc_depth(index);
                    if (new_vertices[index * 3 + 2] < 0.01 || new_vertices[index * 3 + 2] > 10.0)
                        new_vertices[index * 3 + 2] = best_vertices[index * 3 + 2];
                }

                scene_vertices = new_vertices;

                t.tic();

                float error = computeError(frame, lvl);
                //error += scene.errorRegu();

                std::cout << "new error " << error << " " << t.toc() << std::endl;

                if (error < last_error)
                {
                    // accept update, decrease lambda
                    best_vertices = new_vertices;

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
                    n_try++;

                    if (lambda < 0.2f)
                        lambda = 0.2f;
                    else
                        lambda *= 2.0; // std::pow(2.0, n_try);

                    scene_vertices = best_vertices;

                    // reject update, increase lambda, use un-updated data

                    if (inc_depth.dot(inc_depth) < 1e-8)
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

