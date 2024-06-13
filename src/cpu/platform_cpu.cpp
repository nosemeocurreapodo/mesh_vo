#include "platform_cpu.h"

void platformCpu::computeFrameDerivative(frameCpu &frame, camera &cam, int lvl)
{
    for (int y = 1; y < cam.height[lvl] - 1; y++)
        for (int x = 1; x < cam.width[lvl] - 1; x++)
        {
            cv::Vec2f d;
            d.val[0] = (frame.image.texture[lvl].at<uchar>(y, x + 1) - frame.image.texture[lvl].at<uchar>(y, x - 1)) / 2.0;
            d.val[1] = (frame.image.texture[lvl].at<uchar>(y + 1, x) - frame.image.texture[lvl].at<uchar>(y - 1, x)) / 2.0;

            frame.der.texture[lvl].at<cv::Vec2f>(y, x) = d;
        }
}

void platformCpu::computeFrameIdepth(frameCpu &frame, camera &cam, sceneMesh &scene, int lvl)
{
    // for each triangle
    for (std::size_t index = 0; index < scene.scene_indices.size(); index += 3)
    {
        // get its vertices
        //Eigen::Vector3f world_vertex[3];
        Eigen::Vector3f keyframe_vertex[3];
        for (int vertex = 0; vertex < 3; vertex++)
        {
            int vertex_index = scene.scene_indices[index + vertex];

            //scene vertices are vertex in keyframe coordinates
            //keyframe_vertex[vertex](0) = scene.scene_vertices[vertex_index * 3];
            //keyframe_vertex[vertex](1) = scene.scene_vertices[vertex_index * 3 + 1];
            //keyframe_vertex[vertex](2) = scene.scene_vertices[vertex_index * 3 + 2];

            //scene vertices are ray + idepth in keyframe coordinates
            Eigen::Vector3f keyframe_ray;
            keyframe_ray(0) = scene.scene_vertices[vertex_index * 3];
            keyframe_ray(1) = scene.scene_vertices[vertex_index * 3 + 1];
            keyframe_ray(2) = 1.0;
            float keyframe_idepth = scene.scene_vertices[vertex_index * 3 + 2];

            keyframe_vertex[vertex] = keyframe_ray/keyframe_idepth;
        }

        Eigen::Vector3f frame_vertex[3];

        // vertex from world reference to camera reference system
        frame_vertex[0] = frame.pose * keyframe_vertex[0];
        frame_vertex[1] = frame.pose * keyframe_vertex[1];
        frame_vertex[2] = frame.pose * keyframe_vertex[2];

        Eigen::Vector3f frame_normal = (frame_vertex[0] - frame_vertex[2]).cross(frame_vertex[0] - frame_vertex[1]);

        // back-face culling
        float point_dot_normal = frame_vertex[0].dot(frame_normal);
        if (point_dot_normal <= 0.0)
            continue;

        Eigen::Vector2f pixel[3];
        pixel[0] = Eigen::Vector2f(cam.fx[lvl] * frame_vertex[0](0) / frame_vertex[0](2) + cam.cx[lvl], cam.fy[lvl] * frame_vertex[0](1) / frame_vertex[0](2) + cam.cy[lvl]);
        pixel[1] = Eigen::Vector2f(cam.fx[lvl] * frame_vertex[1](0) / frame_vertex[1](2) + cam.cx[lvl], cam.fy[lvl] * frame_vertex[1](1) / frame_vertex[1](2) + cam.cy[lvl]);
        pixel[2] = Eigen::Vector2f(cam.fx[lvl] * frame_vertex[2](0) / frame_vertex[2](2) + cam.cx[lvl], cam.fy[lvl] * frame_vertex[2](1) / frame_vertex[2](2) + cam.cy[lvl]);

        int min_x = std::min(std::min(pixel[0](0), pixel[1](0)), pixel[2](0));
        int max_x = std::max(std::max(pixel[0](0), pixel[1](0)), pixel[2](0));
        int min_y = std::min(std::min(pixel[0](1), pixel[1](1)), pixel[2](1));
        int max_y = std::max(std::max(pixel[0](1), pixel[1](1)), pixel[2](1));

        // triangle outside of frame
        if(min_x >= cam.width[lvl] || max_x < 0.0)
            continue;
        if(min_y >= cam.height[lvl] || max_y < 0.0)
            continue;

        Eigen::Matrix2f T;
        T(0, 0) = pixel[0](0) - pixel[2](0);
        T(0, 1) = pixel[1](0) - pixel[2](0);
        T(1, 0) = pixel[0](1) - pixel[2](1);
        T(1, 1) = pixel[1](1) - pixel[2](1);
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

                Eigen::Vector2f lambda = T_inv * (Eigen::Vector2f(x, y) - pixel[2]);
                
                if (lambda(0) < 0.0 || lambda(1) < 0.0 || (1.0 - lambda(0) - lambda(1)) < 0.0)
                    continue;

                float depth = lambda(0) * frame_vertex[0](2) + lambda(1) * frame_vertex[1](2) + (1 - lambda(0) - lambda(1)) * frame_vertex[2](2);

                frame.idepth.texture[lvl].at<float>(y, x) = 1.0 / depth;
            }
        }
    }
}

float platformCpu::computeError(frameCpu &frame, frameCpu &keyframe, camera &cam, int lvl)
{
    HGPose _hjpose = errorPerIndex(frame, keyframe, cam, lvl, 0, cam.height[lvl]);
    // float error = treadReducer.reduce(std::bind(&mesh_vo::errorCPUPerIndex, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), _frame, lvl, 0, height[lvl]);
    return _hjpose.error / _hjpose.count;
}

HGPose platformCpu::errorPerIndex(frameCpu &frame, frameCpu &keyframe, camera &cam, int lvl, int ymin, int ymax)
{
    HGPose hgpose;

    Sophus::SE3f relativePose = frame.pose * keyframe.pose.inverse();

    for (int y = ymin; y < ymax; y++)
        for (int x = 0; x < cam.width[lvl]; x++)
        {
            uchar vkf = keyframe.image.texture[lvl].at<uchar>(y, x);
            float keyframeId = keyframe.idepth.texture[lvl].at<float>(y, x);

            if (keyframeId <= 0.0)
                continue;

            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(cam.fxinv[lvl] * (x + 0.5) + cam.cxinv[lvl], cam.fyinv[lvl] * (y + 0.5) + cam.cyinv[lvl], 1.0) / keyframeId;
            Eigen::Vector3f pointFrame = relativePose * poinKeyframe;

            // std::cout << "pointKeyframe " << poinKeyframe << std::endl;
            // std::cout << "pointFrame " << pointFrame << std::endl;

            if (pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector3f pixelFrame = Eigen::Vector3f(cam.fx[lvl] * pointFrame(0) / pointFrame(2) + cam.cx[lvl], cam.fy[lvl] * pointFrame(1) / pointFrame(2) + cam.cy[lvl], 1.0);

            if (pixelFrame(0) < 0.0 || pixelFrame(0) > cam.width[lvl] || pixelFrame(1) < 0 || pixelFrame(1) > cam.height[lvl])
                continue;

            // std::cout << "pixelFrame " << pixelFrame << std::endl;
            uchar vf = frame.image.texture[lvl].at<uchar>(pixelFrame(1), pixelFrame(0));

            float residual = float(vf) - float(vkf);
            float error = residual * residual;

            frame.error.texture[lvl].at<float>(y, x) = error;

            hgpose.error += error;
            hgpose.count++;
        }

    return hgpose;
}

HGPose platformCpu::computeHGPose(frameCpu &frame, frameCpu &keyframe, camera &cam, int lvl)
{
    HGPose hgpose = HGPosePerIndex(frame, keyframe, cam, lvl, 0, cam.height[lvl]);
    // HJPose _hjpose = treadReducer.reduce(std::bind(&mesh_vo::HJPoseCPUPerIndex, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), _frame, lvl, 0, height[lvl]);
    hgpose.H_pose /= hgpose.count;
    hgpose.G_pose /= hgpose.count;
    hgpose.error /= hgpose.count;

    return hgpose;
}

HGPose platformCpu::HGPosePerIndex(frameCpu &frame, frameCpu &keyframe, camera &cam, int lvl, int ymin, int ymax)
{
    HGPose hgpose;

    Sophus::SE3f relativePose = frame.pose * keyframe.pose.inverse();

    for (int y = ymin; y < ymax; y++)
        for (int x = 0; x < cam.width[lvl]; x++)
        {
            uchar vkf = keyframe.image.texture[lvl].at<uchar>(y, x);
            float keyframeId = keyframe.idepth.texture[lvl].at<float>(y, x);

            // std::cout << "keyframeId " << keyframeId << std::endl;

            if (keyframeId <= 0.0)
                continue;

            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(cam.fxinv[lvl] * x + cam.cxinv[lvl], cam.fyinv[lvl] * y + cam.cyinv[lvl], 1.0) / keyframeId;
            Eigen::Vector3f pointFrame = relativePose * poinKeyframe;

            // std::cout << "pointFrame " << pointFrame << std::endl;

            if (pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector3f pixelFrame = Eigen::Vector3f(cam.fx[lvl] * pointFrame(0) / pointFrame(2) + cam.cx[lvl], cam.fy[lvl] * pointFrame(1) / pointFrame(2) + cam.cy[lvl], 1.0);

            // std::cout << "pixelFrame " << std::endl;

            if (pixelFrame(0) < 0.0 || pixelFrame(0) >= cam.width[lvl] || pixelFrame(1) < 0.0 || pixelFrame(1) >= cam.height[lvl])
                continue;

            uchar vf = frame.image.texture[lvl].at<uchar>(pixelFrame(1), pixelFrame(0));
            cv::Vec2f der = frame.der.texture[lvl].at<cv::Vec2f>(pixelFrame(1), pixelFrame(0));

            Eigen::Vector2f d_f_d_uf(der.val[0], der.val[1]);

            // std::cout << "vf " << vf << " der " << der << std::endl;

            float id = 1.0 / pointFrame(2);

            float v0 = d_f_d_uf(0) * cam.fx[lvl] * id;
            float v1 = d_f_d_uf(1) * cam.fy[lvl] * id;
            float v2 = -(v0 * pointFrame(0) + v1 * pointFrame(1)) * id;

            Eigen::Vector3f d_I_d_tra = Eigen::Vector3f(v0, v1, v2);
            Eigen::Vector3f d_I_d_rot = Eigen::Vector3f(-pointFrame(2) * v1 + pointFrame(1) * v2, pointFrame(2) * v0 - pointFrame(0) * v2, -pointFrame(1) * v0 + pointFrame(0) * v1);

            float residual = (vf - vkf);
            hgpose.error += residual * residual;

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
