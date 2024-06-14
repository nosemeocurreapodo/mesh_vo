#include "platform_cpu.h"

void platformCpu::computeFrameDerivative(frameCpu &frame, camera &cam, int lvl)
{
    frame.dx.set(frame.dx.nodata, lvl);
    frame.dy.set(frame.dy.nodata, lvl);

    for (int y = 1; y < cam.height[lvl] - 1; y++)
        for (int x = 1; x < cam.width[lvl] - 1; x++)
        {
            float dx = (frame.image.get(y, x + 1, lvl) - frame.image.get(y, x - 1, lvl)) / 2.0;
            float dy = (frame.image.get(y + 1, x, lvl) - frame.image.get(y - 1, x, lvl)) / 2.0;

            frame.dx.set(dx, y, x, lvl);
            frame.dy.set(dy, y, x, lvl);
        }
}

void platformCpu::computeFrameIdepth(frameCpu &frame, camera &cam, sceneMesh &scene, int lvl)
{
    frame.idepth.set(frame.idepth.nodata, lvl);

    // for each triangle
    for (std::size_t index = 0; index < scene.scene_indices.size(); index += 3)
    {
        Eigen::Vector3f kf_tri_ver[3];
        Eigen::Vector3f kf_tri_ray[3];
        Eigen::Vector2f kf_tri_pix[3];
        float kf_tri_idepth[3];

        for (int vertex = 0; vertex < 3; vertex++)
        {
            int vertex_i = scene.scene_indices[index + vertex];

            // scene vertices are ray + idepth in keyframe coordinates
            kf_tri_ray[vertex](0) = scene.scene_vertices[vertex_i * 3];
            kf_tri_ray[vertex](1) = scene.scene_vertices[vertex_i * 3 + 1];
            kf_tri_ray[vertex](2) = 1.0;
            kf_tri_idepth[vertex] = scene.scene_vertices[vertex_i * 3 + 2];

            kf_tri_ver[vertex] = kf_tri_ray[vertex] / kf_tri_idepth[vertex];

            kf_tri_pix[vertex](0) = cam.fx[lvl] * kf_tri_ray[vertex](0) + cam.cx[lvl];
            kf_tri_pix[vertex](1) = cam.fy[lvl] * kf_tri_ray[vertex](1) + cam.cy[lvl];
        }

        int min_x = std::min(std::min(kf_tri_pix[0](0), kf_tri_pix[1](0)), kf_tri_pix[2](0));
        int max_x = std::max(std::max(kf_tri_pix[0](0), kf_tri_pix[1](0)), kf_tri_pix[2](0));
        int min_y = std::min(std::min(kf_tri_pix[0](1), kf_tri_pix[1](1)), kf_tri_pix[2](1));
        int max_y = std::max(std::max(kf_tri_pix[0](1), kf_tri_pix[1](1)), kf_tri_pix[2](1));

        // triangle outside of keyframe
        if (min_x >= cam.width[lvl] || max_x < 0.0)
            continue;
        if (min_y >= cam.height[lvl] || max_y < 0.0)
            continue;

        Eigen::Vector3f f_tri_ver[3];
        Eigen::Vector3f f_tri_ray[3];
        Eigen::Vector2f f_tri_pix[3];

        for (int vertex = 0; vertex < 3; vertex++)
        {
            // vertex from world reference to camera reference system
            f_tri_ver[vertex] = frame.pose * kf_tri_ver[vertex];
            f_tri_ray[vertex] = f_tri_ver[vertex] / f_tri_ver[vertex](2);

            f_tri_pix[vertex](0) = cam.fx[lvl] * f_tri_ray[vertex](0) + cam.cx[lvl];
            f_tri_pix[vertex](1) = cam.fy[lvl] * f_tri_ray[vertex](1) + cam.cy[lvl];
        }

        Eigen::Vector3f f_tri_nor = (f_tri_ver[0] - f_tri_ver[2]).cross(f_tri_ver[0] - f_tri_ver[1]);

        // back-face culling
        float point_dot_normal = f_tri_ver[0].dot(f_tri_nor);
        if (point_dot_normal <= 0.0)
            continue;

        min_x = std::min(std::min(f_tri_pix[0](0), f_tri_pix[1](0)), f_tri_pix[2](0));
        max_x = std::max(std::max(f_tri_pix[0](0), f_tri_pix[1](0)), f_tri_pix[2](0));
        min_y = std::min(std::min(f_tri_pix[0](1), f_tri_pix[1](1)), f_tri_pix[2](1));
        max_y = std::max(std::max(f_tri_pix[0](1), f_tri_pix[1](1)), f_tri_pix[2](1));

        // triangle outside of frame
        if (min_x >= cam.width[lvl] || max_x < 0.0)
            continue;
        if (min_y >= cam.height[lvl] || max_y < 0.0)
            continue;

        Eigen::Matrix2f T;
        T(0, 0) = f_tri_pix[0](0) - f_tri_pix[2](0);
        T(0, 1) = f_tri_pix[1](0) - f_tri_pix[2](0);
        T(1, 0) = f_tri_pix[0](1) - f_tri_pix[2](1);
        T(1, 1) = f_tri_pix[1](1) - f_tri_pix[2](1);
        Eigen::Matrix2f T_inv;
        T_inv = T.inverse();

        for (int y = min_y; y <= max_y; y++)
        {
            for (int x = min_x; x <= max_x; x++)
            {
                /*
                Eigen::Vector3f ray = Eigen::Vector3f(cam.fxinv[lvl] * (x + 0.5) + cam.cxinv[lvl], cam.fyinv[lvl] * (y + 0.5) + cam.cyinv[lvl], 1.0);
                float ray_dot_normal = ray.dot(frame_normal);
                if (ray_dot_normal <= 0.0) // osea, este punto no se ve desde la camara...
                    continue;
                float depth = point_dot_normal / ray_dot_normal; // ya estoy seguro que es positivo

                frame.idepth.texture[lvl].at<float>(y, x) = 1.0 / depth;
                */

                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);

                Eigen::Vector2f lambda = T_inv * (f_pix - f_tri_pix[2]);

                if (lambda(0) < 0.0 || lambda(1) < 0.0 || (1.0 - lambda(0) - lambda(1)) < 0.0)
                    continue;

                // Eigen::Vector2f kf_pix = lambda(0) * kf_tri_pix[0] + lambda(1) * kf_tri_pix[1] + (1 - lambda(0) - lambda(1)) * kf_tri_pix[2];
                // Eigen::Vector3f kf_ray = lambda(0) * kf_tri_ray[0] + lambda(1) * kf_tri_ray[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ray[2];
                // Eigen::Vector3f kf_ver = lambda(0) * kf_tri_ver[0] + lambda(1) * kf_tri_ver[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ver[2];

                // Eigen::Vector3f f_ray = lambda(0) * f_tri_ray[0] + lambda(1) * f_tri_ray[1] + (1 - lambda(0) - lambda(1)) * f_tri_ray[2];
                Eigen::Vector3f f_ver = lambda(0) * f_tri_ver[0] + lambda(1) * f_tri_ver[1] + (1 - lambda(0) - lambda(1)) * f_tri_ver[2];

                float idepth = 1.0 / f_ver(2);

                float l_idepth = frame.idepth.get(f_pix(1), f_pix(0), lvl);
                if (l_idepth > idepth && l_idepth != frame.idepth.nodata)
                    continue;

                frame.idepth.set(idepth, f_pix(1), f_pix(0), lvl);
            }
        }
    }
}

float platformCpu::computeError(frameCpu &frame, frameCpu &keyframe, camera &cam, sceneMesh &scene, int lvl)
{
    frame.error.set(frame.error.nodata, lvl);

    HGPose hgpose = errorPerIndex(frame, keyframe, cam, scene, lvl, 0, cam.height[lvl]);
    //HGPose hgpose = errorPerIndex2(frame, keyframe, cam, scene, lvl);
    // float error = treadReducer.reduce(std::bind(&mesh_vo::errorCPUPerIndex, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), _frame, lvl, 0, height[lvl]);
    return hgpose.error / hgpose.count;
}

HGPose platformCpu::errorPerIndex(frameCpu &frame, frameCpu &keyframe, camera &cam, sceneMesh &scene, int lvl, int ymin, int ymax)
{
    HGPose hgpose;

    Sophus::SE3f relativePose = frame.pose; // * keyframe.pose.inverse();

    for (int y = ymin; y < ymax; y++)
        for (int x = 0; x < cam.width[lvl]; x++)
        {
            uchar vkf = keyframe.image.get(y, x, lvl);
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

            uchar vf = frame.image.get(pixelFrame(1), pixelFrame(0), lvl);

            if (vf == frame.image.nodata)
                continue;

            float residual = float(vf) - float(vkf);
            float error = residual * residual;

            frame.error.set(error, pixelFrame(1), pixelFrame(0), lvl);

            hgpose.error += error;
            hgpose.count++;
        }

    return hgpose;
}

HGPose platformCpu::errorPerIndex2(frameCpu &frame, frameCpu &keyframe, camera &cam, sceneMesh &scene, int lvl)
{
    HGPose hg_pose;

    frame.idepth.set(frame.idepth.nodata, lvl);
    frame.error.set(frame.error.nodata, lvl);

    // for each triangle
    for (std::size_t index = 0; index < scene.scene_indices.size(); index += 3)
    {
        // get its vertices
        Eigen::Vector3f kf_tri_ver[3];
        Eigen::Vector3f kf_tri_ray[3];
        Eigen::Vector2f kf_tri_pix[3];
        float kf_tri_idepth[3];

        for (int vertex = 0; vertex < 3; vertex++)
        {
            int vertex_i = scene.scene_indices[index + vertex];

            // scene vertices are ray + idepth in keyframe coordinates
            kf_tri_ray[vertex](0) = scene.scene_vertices[vertex_i * 3];
            kf_tri_ray[vertex](1) = scene.scene_vertices[vertex_i * 3 + 1];
            kf_tri_ray[vertex](2) = 1.0;
            kf_tri_idepth[vertex] = scene.scene_vertices[vertex_i * 3 + 2];

            kf_tri_ver[vertex] = kf_tri_ray[vertex] / kf_tri_idepth[vertex];

            kf_tri_pix[vertex](0) = cam.fx[lvl] * kf_tri_ray[vertex](0) + cam.cx[lvl];
            kf_tri_pix[vertex](1) = cam.fy[lvl] * kf_tri_ray[vertex](1) + cam.cy[lvl];
        }

        int min_x = std::min(std::min(kf_tri_pix[0](0), kf_tri_pix[1](0)), kf_tri_pix[2](0));
        int max_x = std::max(std::max(kf_tri_pix[0](0), kf_tri_pix[1](0)), kf_tri_pix[2](0));
        int min_y = std::min(std::min(kf_tri_pix[0](1), kf_tri_pix[1](1)), kf_tri_pix[2](1));
        int max_y = std::max(std::max(kf_tri_pix[0](1), kf_tri_pix[1](1)), kf_tri_pix[2](1));

        // triangle outside of frame
        if (min_x >= cam.width[lvl] || max_x < 0.0)
            continue;
        if (min_y >= cam.height[lvl] || max_y < 0.0)
            continue;

        Eigen::Vector3f f_tri_ver[3];
        Eigen::Vector3f f_tri_ray[3];
        Eigen::Vector2f f_tri_pix[3];

        for (int vertex = 0; vertex < 3; vertex++)
        {
            // vertex from world reference to camera reference system
            f_tri_ver[vertex] = frame.pose * kf_tri_ver[vertex];
            f_tri_ray[vertex] = f_tri_ver[vertex] / f_tri_ver[vertex](2);

            f_tri_pix[vertex](0) = cam.fx[lvl] * f_tri_ray[vertex](0) + cam.cx[lvl];
            f_tri_pix[vertex](1) = cam.fy[lvl] * f_tri_ray[vertex](1) + cam.cy[lvl];
        }

        Eigen::Vector3f f_tri_nor = (f_tri_ver[0] - f_tri_ver[2]).cross(f_tri_ver[0] - f_tri_ver[1]);

        // back-face culling
        float point_dot_normal = f_tri_ver[0].dot(f_tri_nor);
        if (point_dot_normal <= 0.0)
            continue;

        min_x = std::min(std::min(f_tri_pix[0](0), f_tri_pix[1](0)), f_tri_pix[2](0));
        max_x = std::max(std::max(f_tri_pix[0](0), f_tri_pix[1](0)), f_tri_pix[2](0));
        min_y = std::min(std::min(f_tri_pix[0](1), f_tri_pix[1](1)), f_tri_pix[2](1));
        max_y = std::max(std::max(f_tri_pix[0](1), f_tri_pix[1](1)), f_tri_pix[2](1));

        // triangle outside of frame
        if (min_x >= cam.width[lvl] || max_x < 0.0)
            continue;
        if (min_y >= cam.height[lvl] || max_y < 0.0)
            continue;

        Eigen::Matrix2f T;
        T(0, 0) = f_tri_pix[0](0) - f_tri_pix[2](0);
        T(0, 1) = f_tri_pix[1](0) - f_tri_pix[2](0);
        T(1, 0) = f_tri_pix[0](1) - f_tri_pix[2](1);
        T(1, 1) = f_tri_pix[1](1) - f_tri_pix[2](1);
        Eigen::Matrix2f T_inv;
        T_inv = T.inverse();

        for (int y = min_y; y <= max_y; y++)
        {
            for (int x = min_x; x <= max_x; x++)
            {
                /*
                Eigen::Vector3f ray = Eigen::Vector3f(cam.fxinv[lvl] * (x + 0.5) + cam.cxinv[lvl], cam.fyinv[lvl] * (y + 0.5) + cam.cyinv[lvl], 1.0);
                float ray_dot_normal = ray.dot(frame_normal);
                if (ray_dot_normal <= 0.0) // osea, este punto no se ve desde la camara...
                    continue;
                float depth = point_dot_normal / ray_dot_normal; // ya estoy seguro que es positivo

                frame.idepth.texture[lvl].at<float>(y, x) = 1.0 / depth;
                */

                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);

                Eigen::Vector2f lambda = T_inv * (f_pix - f_tri_pix[2]);

                if (lambda(0) < 0.0 || lambda(1) < 0.0 || (1.0 - lambda(0) - lambda(1)) < 0.0)
                    continue;

                Eigen::Vector2f kf_pix = lambda(0) * kf_tri_pix[0] + lambda(1) * kf_tri_pix[1] + (1 - lambda(0) - lambda(1)) * kf_tri_pix[2];
                // Eigen::Vector3f kf_ray = lambda(0) * kf_tri_ray[0] + lambda(1) * kf_tri_ray[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ray[2];
                // Eigen::Vector3f kf_ver = lambda(0) * kf_tri_ver[0] + lambda(1) * kf_tri_ver[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ver[2];

                // Eigen::Vector3f f_ray = lambda(0) * f_tri_ray[0] + lambda(1) * f_tri_ray[1] + (1 - lambda(0) - lambda(1)) * f_tri_ray[2];
                Eigen::Vector3f f_ver = lambda(0) * f_tri_ver[0] + lambda(1) * f_tri_ver[1] + (1 - lambda(0) - lambda(1)) * f_tri_ver[2];

                uchar kf_i = keyframe.image.get(kf_pix(1), kf_pix(0), lvl);
                uchar f_i = frame.image.get(f_pix(1), f_pix(0), lvl);
                if (kf_i == keyframe.image.nodata || f_i == frame.image.nodata)
                    continue;

                float f_idepth = 1.0 / f_ver(2);

                float residual = (f_i - kf_i);
                float residual_2 = residual * residual;

                // z buffer
                float l_idepth = frame.idepth.get(f_pix(1), f_pix(0), lvl);
                if (l_idepth > f_idepth && l_idepth != frame.idepth.nodata)
                    continue;
                
                hg_pose.error += residual_2;

                frame.error.set(residual_2, f_pix(1), f_pix(0), lvl);
                frame.idepth.set(f_idepth, f_pix(1), f_pix(0), lvl);

                hg_pose.count++;
            }
        }
    }

    return hg_pose;
}

HGPose platformCpu::computeHGPose(frameCpu &frame, frameCpu &keyframe, camera &cam, sceneMesh &scene, int lvl)
{
    //HGPose hgpose = HGPosePerIndex(frame, keyframe, cam, scene, lvl, 0, cam.height[lvl]);
    HGPose hgpose = HGPosePerIndex2(frame, keyframe, cam, scene, lvl);
    //  HJPose _hjpose = treadReducer.reduce(std::bind(&mesh_vo::HJPoseCPUPerIndex, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), _frame, lvl, 0, height[lvl]);
    hgpose.H_pose /= hgpose.count;
    hgpose.G_pose /= hgpose.count;
    hgpose.error /= hgpose.count;

    return hgpose;
}

HGPose platformCpu::HGPosePerIndex(frameCpu &frame, frameCpu &keyframe, camera &cam, sceneMesh &scene, int lvl, int ymin, int ymax)
{
    HGPose hgpose;

    Sophus::SE3f relativePose = frame.pose; // * keyframe.pose.inverse();

    for (int y = ymin; y < ymax; y++)
        for (int x = 0; x < cam.width[lvl]; x++)
        {
            uchar vkf = keyframe.image.get(y, x, lvl);
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

            uchar vf = frame.image.get(pixelFrame(1), pixelFrame(0), lvl);
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

            float residual = (vf - vkf);
            float residual_2 = residual * residual;

            // z-buffer
            float l_idepth = frame.idepth.get(pixelFrame(1), pixelFrame(0), lvl);
            if (l_idepth > id && l_idepth != frame.idepth.nodata)
                continue;

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

HGPose platformCpu::HGPosePerIndex2(frameCpu &frame, frameCpu &keyframe, camera &cam, sceneMesh &scene, int lvl)
{
    HGPose hg_pose;

    frame.idepth.set(frame.idepth.nodata, lvl);
    frame.error.set(frame.error.nodata, lvl);

    // for each triangle
    for (std::size_t index = 0; index < scene.scene_indices.size(); index += 3)
    {
        // get its vertices
        Eigen::Vector3f kf_tri_ver[3];
        Eigen::Vector3f kf_tri_ray[3];
        Eigen::Vector2f kf_tri_pix[3];
        float kf_tri_idepth[3];

        for (int vertex = 0; vertex < 3; vertex++)
        {
            int vertex_i = scene.scene_indices[index + vertex];

            // scene vertices are ray + idepth in keyframe coordinates
            kf_tri_ray[vertex](0) = scene.scene_vertices[vertex_i * 3];
            kf_tri_ray[vertex](1) = scene.scene_vertices[vertex_i * 3 + 1];
            kf_tri_ray[vertex](2) = 1.0;
            kf_tri_idepth[vertex] = scene.scene_vertices[vertex_i * 3 + 2];

            kf_tri_ver[vertex] = kf_tri_ray[vertex] / kf_tri_idepth[vertex];

            kf_tri_pix[vertex](0) = cam.fx[lvl] * kf_tri_ray[vertex](0) + cam.cx[lvl];
            kf_tri_pix[vertex](1) = cam.fy[lvl] * kf_tri_ray[vertex](1) + cam.cy[lvl];
        }

        Eigen::Vector3f f_tri_ver[3];
        Eigen::Vector3f f_tri_ray[3];
        Eigen::Vector2f f_tri_pix[3];

        for (int vertex = 0; vertex < 3; vertex++)
        {
            // vertex from world reference to camera reference system
            f_tri_ver[vertex] = frame.pose * kf_tri_ver[vertex];
            f_tri_ray[vertex] = f_tri_ver[vertex] / f_tri_ver[vertex](2);

            f_tri_pix[vertex](0) = cam.fx[lvl] * f_tri_ray[vertex](0) + cam.cx[lvl];
            f_tri_pix[vertex](1) = cam.fy[lvl] * f_tri_ray[vertex](1) + cam.cy[lvl];
        }

        Eigen::Vector3f f_tri_nor = (f_tri_ver[0] - f_tri_ver[2]).cross(f_tri_ver[0] - f_tri_ver[1]);

        // back-face culling
        float point_dot_normal = f_tri_ver[0].dot(f_tri_nor);
        if (point_dot_normal <= 0.0)
            continue;

        int min_x = std::min(std::min(f_tri_pix[0](0), f_tri_pix[1](0)), f_tri_pix[2](0));
        int max_x = std::max(std::max(f_tri_pix[0](0), f_tri_pix[1](0)), f_tri_pix[2](0));
        int min_y = std::min(std::min(f_tri_pix[0](1), f_tri_pix[1](1)), f_tri_pix[2](1));
        int max_y = std::max(std::max(f_tri_pix[0](1), f_tri_pix[1](1)), f_tri_pix[2](1));

        // triangle outside of frame
        if (min_x >= cam.width[lvl] || max_x < 0.0)
            continue;
        if (min_y >= cam.height[lvl] || max_y < 0.0)
            continue;

        Eigen::Matrix2f T;
        T(0, 0) = f_tri_pix[0](0) - f_tri_pix[2](0);
        T(0, 1) = f_tri_pix[1](0) - f_tri_pix[2](0);
        T(1, 0) = f_tri_pix[0](1) - f_tri_pix[2](1);
        T(1, 1) = f_tri_pix[1](1) - f_tri_pix[2](1);
        Eigen::Matrix2f T_inv;
        T_inv = T.inverse();

        for (int y = min_y; y <= max_y; y++)
        {
            for (int x = min_x; x <= max_x; x++)
            {
                /*
                Eigen::Vector3f ray = Eigen::Vector3f(cam.fxinv[lvl] * (x + 0.5) + cam.cxinv[lvl], cam.fyinv[lvl] * (y + 0.5) + cam.cyinv[lvl], 1.0);
                float ray_dot_normal = ray.dot(frame_normal);
                if (ray_dot_normal <= 0.0) // osea, este punto no se ve desde la camara...
                    continue;
                float depth = point_dot_normal / ray_dot_normal; // ya estoy seguro que es positivo

                frame.idepth.texture[lvl].at<float>(y, x) = 1.0 / depth;
                */

                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);

                Eigen::Vector2f lambda = T_inv * (f_pix - f_tri_pix[2]);

                if (lambda(0) < 0.0 || lambda(1) < 0.0 || (1.0 - lambda(0) - lambda(1)) < 0.0)
                    continue;

                Eigen::Vector2f kf_pix = lambda(0) * kf_tri_pix[0] + lambda(1) * kf_tri_pix[1] + (1 - lambda(0) - lambda(1)) * kf_tri_pix[2];
                // Eigen::Vector3f kf_ray = lambda(0) * kf_tri_ray[0] + lambda(1) * kf_tri_ray[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ray[2];
                // Eigen::Vector3f kf_ver = lambda(0) * kf_tri_ver[0] + lambda(1) * kf_tri_ver[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ver[2];

                // Eigen::Vector3f f_ray = lambda(0) * f_tri_ray[0] + lambda(1) * f_tri_ray[1] + (1 - lambda(0) - lambda(1)) * f_tri_ray[2];
                Eigen::Vector3f f_ver = lambda(0) * f_tri_ver[0] + lambda(1) * f_tri_ver[1] + (1 - lambda(0) - lambda(1)) * f_tri_ver[2];

                uchar kf_i = keyframe.image.get(kf_pix(1), kf_pix(0), lvl);
                uchar f_i = frame.image.get(f_pix(1), f_pix(0), lvl);
                float dx = frame.dx.get(f_pix(1), f_pix(0), lvl);
                float dy = frame.dy.get(f_pix(1), f_pix(0), lvl);

                if (kf_i == keyframe.image.nodata || f_i == frame.image.nodata || dx == frame.dx.nodata || dy == frame.dy.nodata)
                    continue;

                Eigen::Vector2f d_f_i_d_pix(dx, dy);
                float f_idepth = 1.0 / f_ver(2);

                float l_idepth = frame.idepth.get(f_pix(1), f_pix(0), lvl);
                if( l_idepth > f_idepth && l_idepth != frame.idepth.nodata)
                    continue;

                float v0 = d_f_i_d_pix(0) * cam.fx[lvl] * f_idepth;
                float v1 = d_f_i_d_pix(1) * cam.fy[lvl] * f_idepth;
                float v2 = -(v0 * f_ver(0) + v1 * f_ver(1)) * f_idepth;

                Eigen::Vector3f d_f_i_d_tra = Eigen::Vector3f(v0, v1, v2);
                Eigen::Vector3f d_f_i_d_rot = Eigen::Vector3f(-f_ver(2) * v1 + f_ver(1) * v2, f_ver(2) * v0 - f_ver(0) * v2, -f_ver(1) * v0 + f_ver(0) * v1);

                float residual = (f_i - kf_i);
                float residual_2 = residual * residual;

                hg_pose.error += residual_2;

                frame.error.set(residual_2, f_pix(1), f_pix(0), lvl);
                frame.idepth.set(f_idepth, f_pix(1), f_pix(0), lvl);

                Eigen::Matrix<float, 6, 1> J;
                J << d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2);

                hg_pose.count++;
                for (int i = 0; i < 6; i++)
                {
                    hg_pose.G_pose(i) += J[i] * residual;
                    for (int j = i; j < 6; j++)
                    {
                        float jj = J[i] * J[j];
                        hg_pose.H_pose(i, j) += jj;
                        hg_pose.H_pose(j, i) += jj;
                    }
                }
            }
        }
    }

    return hg_pose;
}

HGMap platformCpu::HGMapPerIndex(frameCpu &frame, frameCpu &keyframe, camera &cam, sceneMesh &scene, int lvl)
{
    HGMap hg_map;

    // for each triangle
    for (std::size_t index = 0; index < scene.scene_indices.size(); index += 3)
    {
        // get its vertices
        // Eigen::Vector3f world_vertex[3];

        int vertex_id[3];

        Eigen::Vector3f kf_tri_ver[3];
        Eigen::Vector3f kf_tri_ray[3];
        Eigen::Vector2f kf_tri_pix[3];
        float kf_tri_idepth[3];

        for (int vertex = 0; vertex < 3; vertex++)
        {
            int vertex_i = scene.scene_indices[index + vertex];

            vertex_id[vertex] = vertex_i;

            // scene vertices are ray + idepth in keyframe coordinates
            kf_tri_ray[vertex](0) = scene.scene_vertices[vertex_i * 3];
            kf_tri_ray[vertex](1) = scene.scene_vertices[vertex_i * 3 + 1];
            kf_tri_ray[vertex](2) = 1.0;
            kf_tri_idepth[vertex] = scene.scene_vertices[vertex_i * 3 + 2];

            kf_tri_ver[vertex] = kf_tri_ray[vertex] / kf_tri_idepth[vertex];

            kf_tri_pix[vertex](0) = cam.fx[lvl] * kf_tri_ray[vertex](0) + cam.cx[lvl];
            kf_tri_pix[vertex](1) = cam.fy[lvl] * kf_tri_ray[vertex](1) + cam.cy[lvl];
        }

        Eigen::Vector3f f_tri_ver[3];
        Eigen::Vector3f f_tri_ray[3];
        Eigen::Vector2f f_tri_pix[3];

        for (int vertex = 0; vertex < 3; vertex++)
        {
            // vertex from world reference to camera reference system
            f_tri_ver[vertex] = frame.pose * kf_tri_ver[vertex];
            f_tri_ray[vertex] = f_tri_ver[vertex] / f_tri_ver[vertex](2);

            f_tri_pix[vertex](0) = cam.fx[lvl] * f_tri_ray[vertex](0) + cam.cx[lvl];
            f_tri_pix[vertex](1) = cam.fy[lvl] * f_tri_ray[vertex](1) + cam.cy[lvl];
        }

        Eigen::Vector3f f_tri_nor = (f_tri_ver[0] - f_tri_ver[2]).cross(f_tri_ver[0] - f_tri_ver[1]);

        // back-face culling
        float point_dot_normal = f_tri_ver[0].dot(f_tri_nor);
        if (point_dot_normal <= 0.0)
            continue;

        int min_x = std::min(std::min(f_tri_pix[0](0), f_tri_pix[1](0)), f_tri_pix[2](0));
        int max_x = std::max(std::max(f_tri_pix[0](0), f_tri_pix[1](0)), f_tri_pix[2](0));
        int min_y = std::min(std::min(f_tri_pix[0](1), f_tri_pix[1](1)), f_tri_pix[2](1));
        int max_y = std::max(std::max(f_tri_pix[0](1), f_tri_pix[1](1)), f_tri_pix[2](1));

        // triangle outside of frame
        if (min_x >= cam.width[lvl] || max_x < 0.0)
            continue;
        if (min_y >= cam.height[lvl] || max_y < 0.0)
            continue;

        Eigen::Vector3f n_p0 = (kf_tri_ver[0] - kf_tri_ver[1]).cross(kf_tri_ver[2] - kf_tri_ver[1]);
        Eigen::Vector3f pw2mpw1_0 = (kf_tri_ver[2] - kf_tri_ver[1]);
        Eigen::Vector3f d_n_d_z0 = kf_tri_ver[0].cross(pw2mpw1_0);
        float n_p0_dot_point = n_p0.dot(kf_tri_ver[1]);
        Eigen::Vector3f pr_p0 = kf_tri_ver[1];

        Eigen::Vector3f n_p1 = (kf_tri_ver[1] - kf_tri_ver[0]).cross(kf_tri_ver[2] - kf_tri_ver[0]);
        Eigen::Vector3f pw2mpw1_1 = (kf_tri_ver[2] - kf_tri_ver[0]);
        Eigen::Vector3f d_n_d_z1 = kf_tri_ver[1].cross(pw2mpw1_1);
        float n_p1_dot_point = n_p1.dot(kf_tri_ver[0]);
        Eigen::Vector3f pr_p1 = kf_tri_ver[0];

        Eigen::Vector3f n_p2 = (kf_tri_ver[2] - kf_tri_ver[1]).cross(kf_tri_ver[0] - kf_tri_ver[1]);
        Eigen::Vector3f pw2mpw1_2 = (kf_tri_ver[0] - kf_tri_ver[1]);
        Eigen::Vector3f d_n_d_z2 = kf_tri_ver[2].cross(pw2mpw1_2);
        float n_p2_dot_point = n_p2.dot(kf_tri_ver[1]);
        Eigen::Vector3f pr_p2 = kf_tri_ver[1];

        Eigen::Matrix2f T;
        T(0, 0) = f_tri_pix[0](0) - f_tri_pix[2](0);
        T(0, 1) = f_tri_pix[1](0) - f_tri_pix[2](0);
        T(1, 0) = f_tri_pix[0](1) - f_tri_pix[2](1);
        T(1, 1) = f_tri_pix[1](1) - f_tri_pix[2](1);
        Eigen::Matrix2f T_inv;
        T_inv = T.inverse();

        for (int y = min_y; y <= max_y; y++)
        {
            for (int x = min_x; x <= max_x; x++)
            {
                /*
                Eigen::Vector3f ray = Eigen::Vector3f(cam.fxinv[lvl] * (x + 0.5) + cam.cxinv[lvl], cam.fyinv[lvl] * (y + 0.5) + cam.cyinv[lvl], 1.0);
                float ray_dot_normal = ray.dot(frame_normal);
                if (ray_dot_normal <= 0.0) // osea, este punto no se ve desde la camara...
                    continue;
                float depth = point_dot_normal / ray_dot_normal; // ya estoy seguro que es positivo

                frame.idepth.texture[lvl].at<float>(y, x) = 1.0 / depth;
                */

                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);

                Eigen::Vector2f lambda = T_inv * (f_pix - f_tri_pix[2]);

                if (lambda(0) < 0.0 || lambda(1) < 0.0 || (1.0 - lambda(0) - lambda(1)) < 0.0)
                    continue;

                Eigen::Vector2f kf_pix = lambda(0) * kf_tri_pix[0] + lambda(1) * kf_tri_pix[1] + (1 - lambda(0) - lambda(1)) * kf_tri_pix[2];
                Eigen::Vector3f kf_ray = lambda(0) * kf_tri_ray[0] + lambda(1) * kf_tri_ray[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ray[2];
                // Eigen::Vector3f kf_ver = lambda(0) * kf_tri_ver[0] + lambda(1) * kf_tri_ver[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ver[2];

                // Eigen::Vector3f f_ray = lambda(0) * f_tri_ray[0] + lambda(1) * f_tri_ray[1] + (1 - lambda(0) - lambda(1)) * f_tri_ray[2];
                Eigen::Vector3f f_ver = lambda(0) * f_tri_ver[0] + lambda(1) * f_tri_ver[1] + (1 - lambda(0) - lambda(1)) * f_tri_ver[2];

                uchar kf_i = keyframe.image.get(kf_pix(1), kf_pix(0), lvl);
                uchar f_i = frame.image.get(f_pix(1), f_pix(0), lvl);
                float dx = frame.dx.get(f_pix(1), f_pix(0), lvl);
                float dy = frame.dy.get(f_pix(1), f_pix(0), lvl);
                Eigen::Vector2f d_f_i_d_pix(dx, dy);

                // frame.idepth.texture[lvl].at<float>(y, x) = 1.0 / frame_depth;

                Eigen::Vector3f d_f_i_d_f_ver;
                d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx[lvl] / f_ver(2);
                d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy[lvl] / f_ver(2);
                d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) / f_ver(2) + d_f_i_d_f_ver(1) * f_ver(1) / f_ver(2));

                Eigen::Vector3f d_f_ver_d_kf_depth = frame.pose.rotationMatrix() * kf_ray;

                float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                float n_p0_dot_ray = n_p0.dot(kf_ray);
                float n_p1_dot_ray = n_p1.dot(kf_ray);
                float n_p2_dot_ray = n_p2.dot(kf_ray);

                float d_kf_depth_d_z0 = d_n_d_z0.dot(pr_p0) / n_p0_dot_ray - n_p0_dot_point * d_n_d_z0.dot(kf_ray) / (n_p0_dot_ray * n_p0_dot_ray);
                float d_kf_depth_d_z1 = d_n_d_z1.dot(pr_p1) / n_p1_dot_ray - n_p1_dot_point * d_n_d_z1.dot(kf_ray) / (n_p1_dot_ray * n_p1_dot_ray);
                float d_kf_depth_d_z2 = d_n_d_z2.dot(pr_p2) / n_p2_dot_ray - n_p2_dot_point * d_n_d_z2.dot(kf_ray) / (n_p2_dot_ray * n_p2_dot_ray);

                float d_z0_d_iz0 = -1.0 / (kf_tri_idepth[0] * kf_tri_idepth[0]);
                float d_z1_d_iz1 = -1.0 / (kf_tri_idepth[1] * kf_tri_idepth[1]);
                float d_z2_d_iz2 = -1.0 / (kf_tri_idepth[2] * kf_tri_idepth[2]);

                float d_f_i_d_z0 = d_f_i_d_kf_depth * d_kf_depth_d_z0 * d_z0_d_iz0;
                float d_f_i_d_z1 = d_f_i_d_kf_depth * d_kf_depth_d_z1 * d_z1_d_iz1;
                float d_f_i_d_z2 = d_f_i_d_kf_depth * d_kf_depth_d_z2 * d_z2_d_iz2;

                float error = f_i - kf_i;

                float J[3];
                J[0] = d_f_i_d_z0;
                J[1] = d_f_i_d_z1;
                J[2] = d_f_i_d_z2;

                for (int i = 0; i < 3; i++)
                {
                    hg_map.G_depth(vertex_id[i]) += J[i] * error;

                    for (int j = i; j < 3; j++)
                    {
                        // acc_H_depth(vertexID[i],vertexID[j]) += J[i]*J[j];
                        float jj = J[i] * J[j];
                        hg_map.H_depth.coeffRef(vertex_id[i], vertex_id[j]) += jj;
                        hg_map.H_depth.coeffRef(vertex_id[j], vertex_id[i]) += jj;
                    }
                }
            }
        }
    }
}