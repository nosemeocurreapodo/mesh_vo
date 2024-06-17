#include "scene/keyframeIdepthSceneCPU.h"
#include "utils/tictoc.h"

keyframeIdepthSceneCPU::keyframeIdepthSceneCPU(float fx, float fy, float cx, float cy, int width, int height)
    : cam(fx, fy, cx, cy, width, height),
      keyframeIdepth(-1.0)
{
    multiThreading = true;
}
void keyframeIdepthSceneCPU::init(frameCPU &frame, dataCPU<float> &idepth)
{
    frame.copyTo(keyframe);
    idepth.copyTo(keyframeIdepth);
}

dataCPU<float> keyframeIdepthSceneCPU::computeFrameIdepth(frameCPU &frame, int lvl)
{
    dataCPU<float> frameIdepth(-1.0);

    if (multiThreading)
    {
        errorTreadReduce.reduce(boost::bind(&keyframeIdepthSceneCPU::idepthPerIndex, this, frame, frameIdepth, lvl, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3, boost::placeholders::_4), 0, cam.height[lvl], 0);
    }
    else
    {
        Error dummy;
        idepthPerIndex(frame, frameIdepth, lvl, 0, cam.height[lvl], &dummy, 0);
    }

    return frameIdepth;
}

void keyframeIdepthSceneCPU::idepthPerIndex(frameCPU &frame, dataCPU<float> &frameIdepth, int lvl, int ymin, int ymax, Error *dummy, int tid)
{
    Sophus::SE3f relativePose = frame.pose * keyframe.pose.inverse();

    for (int y = ymin; y < ymax; y++)
        for (int x = 0; x < cam.width[lvl]; x++)
        {
            float keyframeId = keyframeIdepth.get(y, x, lvl);

            if (keyframeId == keyframeIdepth.nodata)
                continue;

            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(cam.fxinv[lvl] * (x + 0.5) + cam.cxinv[lvl], cam.fyinv[lvl] * (y + 0.5) + cam.cyinv[lvl], 1.0) / keyframeId;
            Eigen::Vector3f pointFrame = relativePose * poinKeyframe;

            if (pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector2f pixelFrame = Eigen::Vector2f(cam.fx[lvl] * pointFrame(0) / pointFrame(2) + cam.cx[lvl], cam.fy[lvl] * pointFrame(1) / pointFrame(2) + cam.cy[lvl]);

            if (pixelFrame(0) < 0.0 || pixelFrame(0) >= cam.width[lvl] || pixelFrame(1) < 0 || pixelFrame(1) >= cam.height[lvl])
                continue;

            frameIdepth.set(1.0 / pointFrame(2), pixelFrame(1), pixelFrame(0), lvl);
        }
}

float keyframeIdepthSceneCPU::computeError(frameCPU &frame, int lvl)
{
    Error error;

    if (multiThreading)
    {
        errorTreadReduce.reduce(boost::bind(&keyframeIdepthSceneCPU::errorPerIndex, this, frame, lvl, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3, boost::placeholders::_4), 0, cam.height[lvl], 0);
        error = errorTreadReduce.stats;
    }
    else
    {
        errorPerIndex(frame, lvl, 0, cam.height[lvl], &error, 0);
    }

    if (error.count > 0)
        error.error /= error.count;

    return error.error;
}

void keyframeIdepthSceneCPU::errorPerIndex(frameCPU &frame, int lvl, int ymin, int ymax, Error *e, int tid)
{
    Sophus::SE3f relativePose = frame.pose * keyframe.pose.inverse();

    for (int y = ymin; y < ymax; y++)
        for (int x = 0; x < cam.width[lvl]; x++)
        {
            float vkf = float(keyframe.image.get(y, x, lvl));
            float keyframeId = keyframeIdepth.get(y, x, lvl);

            if (vkf == keyframe.image.nodata || keyframeId == keyframeIdepth.nodata)
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

            (*e).error += error;
            (*e).count++;
        }
}

dataCPU<float> keyframeIdepthSceneCPU::computeErrorImage(frameCPU &frame, int lvl)
{
    // HGPose hgpose = errorPerIndex(frame, lvl, 0, cam.height[lvl]);
    //   float error = treadReducer.reduce(std::bind(&mesh_vo::errorCPUPerIndex, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), _frame, lvl, 0, height[lvl]);

    dataCPU<float> errorImage(-1.0);

    if (multiThreading)
    {
        errorTreadReduce.reduce(boost::bind(&keyframeIdepthSceneCPU::errorImagePerIndex, this, frame, errorImage, lvl, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3, boost::placeholders::_4), 0, cam.height[lvl], 0);
    }
    else
    {
        Error dummy;
        errorImagePerIndex(frame, errorImage, lvl, 0, cam.height[lvl], &dummy, 0);
    }

    return errorImage;
}

void keyframeIdepthSceneCPU::errorImagePerIndex(frameCPU &frame, dataCPU<float> &errorImage, int lvl, int ymin, int ymax, Error *e, int tid)
{
    Sophus::SE3f relativePose = frame.pose * keyframe.pose.inverse();

    for (int y = ymin; y < ymax; y++)
        for (int x = 0; x < cam.width[lvl]; x++)
        {
            float vkf = float(keyframe.image.get(y, x, lvl));
            float keyframeId = keyframeIdepth.get(y, x, lvl);

            if (vkf == keyframe.image.nodata || keyframeId == keyframeIdepth.nodata)
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

            //error relative to keyframe
            errorImage.set(error, y, x, lvl);
        }
}


HGPose keyframeIdepthSceneCPU::computeHGPose(frameCPU &frame, int lvl)
{
    HGPose hg;

    if (multiThreading)
    {
        hgPoseTreadReduce.reduce(boost::bind(&keyframeIdepthSceneCPU::HGPosePerIndex, this, frame, lvl, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3, boost::placeholders::_4), 0, cam.height[lvl], 0);
        hg = hgPoseTreadReduce.stats;
    }
    else
    {
        HGPosePerIndex(frame, lvl, 0, cam.height[lvl], &hg, 0);
    }
    /*
    if (hg.count > 0)
    {
        hg.H /= hg.count;
        hg.G /= hg.count;
    }
    */
   return hg;
}

void keyframeIdepthSceneCPU::HGPosePerIndex(frameCPU &frame, int lvl, int ymin, int ymax, HGPose *hg, int tid)
{
    Sophus::SE3f relativePose = frame.pose * keyframe.pose.inverse();

    for (int y = ymin; y < ymax; y++)
        for (int x = 0; x < cam.width[lvl]; x++)
        {
            float vkf = float(keyframe.image.get(y, x, lvl));
            float keyframeId = keyframeIdepth.get(y, x, lvl);
            if (vkf == keyframe.image.nodata || keyframeId == keyframeIdepth.nodata)
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

            float v0 = d_f_d_uf(0) * cam.fx[lvl] * id;
            float v1 = d_f_d_uf(1) * cam.fy[lvl] * id;
            float v2 = -(v0 * pointFrame(0) + v1 * pointFrame(1)) * id;

            Eigen::Vector3f d_I_d_tra = Eigen::Vector3f(v0, v1, v2);
            Eigen::Vector3f d_I_d_rot = Eigen::Vector3f(-pointFrame(2) * v1 + pointFrame(1) * v2, pointFrame(2) * v0 - pointFrame(0) * v2, -pointFrame(1) * v0 + pointFrame(0) * v1);

            float residual = vf - vkf;
            float residual_2 = residual * residual;

            Eigen::Matrix<float, 6, 1> J;
            J << d_I_d_tra(0), d_I_d_tra(1), d_I_d_tra(2), d_I_d_rot(0), d_I_d_rot(1), d_I_d_rot(2);

            for (int i = 0; i < 6; i++)
            {
                (*hg).G(i) += J[i] * residual;
                for (int j = i; j < 6; j++)
                {
                    float jj = J[i] * J[j];
                    (*hg).H(i, j) += jj;
                    (*hg).H(j, i) += jj;
                }
            }
        }
}

void keyframeIdepthSceneCPU::optPose(frameCPU &frame)
{
    int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

    tic_toc t;

    for (int lvl = 2; lvl >= 1; lvl--)
    {
        std::cout << "*************************lvl " << lvl << std::endl;
        t.tic();
        Sophus::SE3f best_pose = frame.pose;
        float last_error = computeError(frame, lvl);

        std::cout << "init error " << last_error << " time " << t.toc() << std::endl;

        for (int it = 0; it < maxIterations[lvl]; it++)
        {
            t.tic();
            HGPose hg = computeHGPose(frame, lvl);
            std::cout << "HGPose time " << t.toc() << std::endl;

            float lambda = 0.0;
            int n_try = 0;
            while (true)
            {
                Eigen::Matrix<float, 6, 6> H_lambda;
                H_lambda = hg.H;

                for (int j = 0; j < 6; j++)
                    H_lambda(j, j) *= 1.0 + lambda;

                Eigen::Matrix<float, 6, 1> inc = H_lambda.ldlt().solve(hg.G);

                // Sophus::SE3f new_pose = frame.pose * Sophus::SE3f::exp(inc_pose);
                frame.pose = best_pose * Sophus::SE3f::exp(inc).inverse();
                // Sophus::SE3f new_pose = Sophus::SE3f::exp(inc_pose).inverse() * frame.pose;

                t.tic();
                float error = computeError(frame, lvl);
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

void keyframeIdepthSceneCPU::optMap(std::vector<frameCPU> &frame)
{
    std::cout << "not implemented" << std::endl;
}
