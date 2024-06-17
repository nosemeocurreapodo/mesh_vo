#include "scene/rayDepthMeshSceneCPU.h"

rayDepthMeshSceneCPU::rayDepthMeshSceneCPU(float fx, float fy, float cx, float cy, int width, int height)
    : cam(fx, fy, cx, cy, width, height)
{
    // preallocate scene vertices to zero
    for (int y = 0; y < VERTEX_HEIGH; y++)
    {
        for (int x = 0; x < VERTEX_WIDTH; x++)
        {
            scene_vertices.push_back(0.0);
            scene_vertices.push_back(0.0);
            scene_vertices.push_back(0.0);
        }
    }

    // init scene indices
    for (int y = 0; y < VERTEX_HEIGH; y++)
    {
        for (int x = 0; x < VERTEX_WIDTH; x++)
        {
            if (x > 0 && y > 0)
            {
                if (((x % 2 == 0)))
                // if(((x % 2 == 0) && (y % 2 == 0)) || ((x % 2 != 0) && (y % 2 != 0)))
                // if (rand() > 0.5 * RAND_MAX)
                {
                    scene_indices.push_back(x - 1 + y * (VERTEX_WIDTH));
                    scene_indices.push_back(x + (y - 1) * (VERTEX_WIDTH));
                    scene_indices.push_back(x - 1 + (y - 1) * (VERTEX_WIDTH));

                    scene_indices.push_back(x + y * (VERTEX_WIDTH));
                    scene_indices.push_back(x + (y - 1) * (VERTEX_WIDTH));
                    scene_indices.push_back(x - 1 + y * (VERTEX_WIDTH));
                }
                else
                {
                    scene_indices.push_back(x + y * (VERTEX_WIDTH));
                    scene_indices.push_back(x - 1 + (y - 1) * (VERTEX_WIDTH));
                    scene_indices.push_back(x - 1 + y * (VERTEX_WIDTH));

                    scene_indices.push_back(x + y * (VERTEX_WIDTH));
                    scene_indices.push_back(x + (y - 1) * (VERTEX_WIDTH));
                    scene_indices.push_back(x - 1 + (y - 1) * (VERTEX_WIDTH));
                }
            }
        }
    }
}

void rayDepthMeshSceneCPU::init(frameCPU &frame, dataCPU<float> idepth)
{
    frame.copyTo(keyframe);
    setFromIdepth(idepth);
}

void rayDepthMeshSceneCPU::setFromIdepth(data_cpu<float> id)
{
    scene_vertices.clear();

    int lvl = 0;

    for (int y = 0; y < VERTEX_HEIGH; y++) // los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
    {
        for (int x = 0; x < VERTEX_WIDTH; x++)
        {
            float xi = (float(x) / float(VERTEX_WIDTH - 1)) * cam.width[lvl];
            float yi = (float(y) / float(VERTEX_HEIGH - 1)) * cam.height[lvl];

            float idepth = id.get(yi, xi, lvl);
            /*
            if(idepth <= min_idepth)
                idepth = min_idepth;
            if(idepth > max_idepth)
                idepth = max_idepth;
                */
            if (idepth != idepth || idepth < 0.1 || idepth > 1.0)
                idepth = 0.1 + (1.0 - 0.1) * float(y) / VERTEX_HEIGH;

            Eigen::Vector3f u = Eigen::Vector3f(xi, yi, 1.0);
            Eigen::Vector3f r = Eigen::Vector3f(cam.fxinv[lvl] * u(0) + cam.cxinv[lvl], cam.fyinv[lvl] * u(1) + cam.cyinv[lvl], 1.0);
            Eigen::Vector3f p = r / idepth;

            scene_vertices.push_back(r(0));
            scene_vertices.push_back(r(1));
            scene_vertices.push_back(idepth);
        }
    }
}

dataCPU<float> rayDepthMeshSceneCPU::computeFrameIdepth(frameCPU &frame, int lvl)
{
    dataCPU<float> idepth(-1.0);

    // for each triangle
    for (std::size_t index = 0; index < scene_indices.size(); index += 3)
    {
        Eigen::Vector3f kf_tri_ver[3];
        Eigen::Vector3f kf_tri_ray[3];
        Eigen::Vector2f kf_tri_pix[3];
        float kf_tri_idepth[3];

        for (int vertex = 0; vertex < 3; vertex++)
        {
            int vertex_i = scene_indices[index + vertex];

            // scene vertices are ray + idepth in keyframe coordinates
            kf_tri_ray[vertex](0) = scene_vertices[vertex_i * 3];
            kf_tri_ray[vertex](1) = scene_vertices[vertex_i * 3 + 1];
            kf_tri_ray[vertex](2) = 1.0;
            kf_tri_idepth[vertex] = scene_vertices[vertex_i * 3 + 2];

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
                if (f_pix(0) < 0.0 || f_pix(0) >= cam.width[lvl] || f_pix(1) < 0.0 || f_pix(1) >= cam.height[lvl])
                    continue;

                Eigen::Vector2f lambda = T_inv * (f_pix - f_tri_pix[2]);

                if (lambda(0) < 0.0 || lambda(1) < 0.0 || (1.0 - lambda(0) - lambda(1)) < 0.0)
                    continue;

                // Eigen::Vector2f kf_pix = lambda(0) * kf_tri_pix[0] + lambda(1) * kf_tri_pix[1] + (1 - lambda(0) - lambda(1)) * kf_tri_pix[2];
                // Eigen::Vector3f kf_ray = lambda(0) * kf_tri_ray[0] + lambda(1) * kf_tri_ray[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ray[2];
                // Eigen::Vector3f kf_ver = lambda(0) * kf_tri_ver[0] + lambda(1) * kf_tri_ver[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ver[2];

                // Eigen::Vector3f f_ray = lambda(0) * f_tri_ray[0] + lambda(1) * f_tri_ray[1] + (1 - lambda(0) - lambda(1)) * f_tri_ray[2];
                Eigen::Vector3f f_ver = lambda(0) * f_tri_ver[0] + lambda(1) * f_tri_ver[1] + (1 - lambda(0) - lambda(1)) * f_tri_ver[2];

                float idepth = 1.0 / f_ver(2);

                float l_idepth = idepth.get(f_pix(1), f_pix(0), lvl);
                if (l_idepth > idepth && l_idepth != idepth.nodata)
                    continue;

                idepth.set(idepth, f_pix(1), f_pix(0), lvl);
            }
        }
    }
    return idepth;
}

float rayDepthMeshSceneCPU::computeError(frameCPU &frame, int lvl)
{
    float error = 0.0;
    int count = 0;

    frame.error.set(frame.error.nodata, lvl);
    frame.idepth.set(frame.idepth.nodata, lvl);

    // for each triangle
    for (std::size_t index = 0; index < scene_indices.size(); index += 3)
    {
        // get its vertices
        Eigen::Vector3f kf_tri_ver[3];
        Eigen::Vector3f kf_tri_ray[3];
        Eigen::Vector2f kf_tri_pix[3];
        float kf_tri_idepth[3];

        for (int vertex = 0; vertex < 3; vertex++)
        {
            int vertex_i = scene_indices[index + vertex];

            // scene vertices are ray + idepth in keyframe coordinates
            kf_tri_ray[vertex](0) = scene_vertices[vertex_i * 3];
            kf_tri_ray[vertex](1) = scene_vertices[vertex_i * 3 + 1];
            kf_tri_ray[vertex](2) = 1.0;
            kf_tri_idepth[vertex] = scene_vertices[vertex_i * 3 + 2];

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
                if (f_pix(0) < 0.0 || f_pix(0) >= cam.width[lvl] || f_pix(1) < 0.0 || f_pix(1) >= cam.height[lvl])
                    continue;

                Eigen::Vector2f lambda = T_inv * (f_pix - f_tri_pix[2]);

                if (lambda(0) < 0.0 || lambda(1) < 0.0 || (1.0 - lambda(0) - lambda(1)) < 0.0)
                    continue;

                Eigen::Vector2f kf_pix = lambda(0) * kf_tri_pix[0] + lambda(1) * kf_tri_pix[1] + (1 - lambda(0) - lambda(1)) * kf_tri_pix[2];
                if (kf_pix(0) < 0.0 || kf_pix(0) >= cam.width[lvl] || kf_pix(1) < 0.0 || kf_pix(1) >= cam.height[lvl])
                    continue;

                // Eigen::Vector3f kf_ray = lambda(0) * kf_tri_ray[0] + lambda(1) * kf_tri_ray[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ray[2];
                // Eigen::Vector3f kf_ver = lambda(0) * kf_tri_ver[0] + lambda(1) * kf_tri_ver[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ver[2];

                // Eigen::Vector3f f_ray = lambda(0) * f_tri_ray[0] + lambda(1) * f_tri_ray[1] + (1 - lambda(0) - lambda(1)) * f_tri_ray[2];
                Eigen::Vector3f f_ver = lambda(0) * f_tri_ver[0] + lambda(1) * f_tri_ver[1] + (1 - lambda(0) - lambda(1)) * f_tri_ver[2];

                float kf_i = float(keyframe.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame.image.get(f_pix(1), f_pix(0), lvl));
                if (kf_i == keyframe.image.nodata || f_i == frame.image.nodata)
                    continue;

                float f_idepth = 1.0 / f_ver(2);

                float residual = f_i - kf_i;
                float residual_2 = residual * residual;

                // z buffer
                float l_idepth = frame.idepth.get(f_pix(1), f_pix(0), lvl);
                if (l_idepth > f_idepth && l_idepth != frame.idepth.nodata)
                    continue;

                error += residual_2;
                count++;

                frame.error.set(residual_2, f_pix(1), f_pix(0), lvl);
                frame.idepth.set(f_idepth, f_pix(1), f_pix(0), lvl);
            }
        }
    }

    if (count > 0)
        error /= count;

    return error;
}

HGPose rayDepthMeshSceneCPU::computeHGPose(frameCPU &frame, int lvl)
{
    HGPose hg_pose;

    frame.idepth.set(frame.idepth.nodata, lvl);
    frame.error.set(frame.error.nodata, lvl);

    // for each triangle
    for (std::size_t index = 0; index < scene_indices.size(); index += 3)
    {
        // get its vertices
        Eigen::Vector3f kf_tri_ver[3];
        Eigen::Vector3f kf_tri_ray[3];
        Eigen::Vector2f kf_tri_pix[3];
        float kf_tri_idepth[3];

        for (int vertex = 0; vertex < 3; vertex++)
        {
            int vertex_i = scene_indices[index + vertex];

            // scene vertices are ray + idepth in keyframe coordinates
            kf_tri_ray[vertex](0) = scene_vertices[vertex_i * 3];
            kf_tri_ray[vertex](1) = scene_vertices[vertex_i * 3 + 1];
            kf_tri_ray[vertex](2) = 1.0;
            kf_tri_idepth[vertex] = scene_vertices[vertex_i * 3 + 2];

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
                if (f_pix(0) < 0.0 || f_pix(0) >= cam.width[lvl] || f_pix(1) < 0.0 || f_pix(1) >= cam.height[lvl])
                    continue;

                Eigen::Vector2f lambda = T_inv * (f_pix - f_tri_pix[2]);

                if (lambda(0) < 0.0 || lambda(1) < 0.0 || (1.0 - lambda(0) - lambda(1)) < 0.0)
                    continue;

                Eigen::Vector2f kf_pix = lambda(0) * kf_tri_pix[0] + lambda(1) * kf_tri_pix[1] + (1 - lambda(0) - lambda(1)) * kf_tri_pix[2];
                if (kf_pix(0) < 0.0 || kf_pix(0) >= cam.width[lvl] || kf_pix(1) < 0.0 || kf_pix(1) >= cam.height[lvl])
                    continue;

                // Eigen::Vector3f kf_ray = lambda(0) * kf_tri_ray[0] + lambda(1) * kf_tri_ray[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ray[2];
                // Eigen::Vector3f kf_ver = lambda(0) * kf_tri_ver[0] + lambda(1) * kf_tri_ver[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ver[2];

                // Eigen::Vector3f f_ray = lambda(0) * f_tri_ray[0] + lambda(1) * f_tri_ray[1] + (1 - lambda(0) - lambda(1)) * f_tri_ray[2];
                Eigen::Vector3f f_ver = lambda(0) * f_tri_ver[0] + lambda(1) * f_tri_ver[1] + (1 - lambda(0) - lambda(1)) * f_tri_ver[2];

                float f_idepth = 1.0 / f_ver(2);

                // z-buffer
                float l_idepth = frame.idepth.get(f_pix(1), f_pix(0), lvl);
                if (l_idepth > f_idepth && l_idepth != frame.idepth.nodata)
                    continue;

                float kf_i = float(keyframe.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame.image.get(f_pix(1), f_pix(0), lvl));
                float dx = frame.dx.get(f_pix(1), f_pix(0), lvl);
                float dy = frame.dy.get(f_pix(1), f_pix(0), lvl);

                if (kf_i == keyframe.image.nodata || f_i == frame.image.nodata || dx == frame.dx.nodata || dy == frame.dy.nodata)
                    continue;

                Eigen::Vector2f d_f_i_d_pix(dx, dy);

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

HGMap rayDepthMeshSceneCPU::computeHGMap(frameCPU &frame, int lvl)
{
    HGMap hg_map;

    // for each triangle
    for (std::size_t index = 0; index < scene_indices.size(); index += 3)
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
            int vertex_i = scene_indices[index + vertex];

            vertex_id[vertex] = vertex_i;

            // scene vertices are ray + idepth in keyframe coordinates
            kf_tri_ray[vertex](0) = scene_vertices[vertex_i * 3];
            kf_tri_ray[vertex](1) = scene_vertices[vertex_i * 3 + 1];
            kf_tri_ray[vertex](2) = 1.0;
            kf_tri_idepth[vertex] = scene_vertices[vertex_i * 3 + 2];

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

        Eigen::Vector3f n_p[3];
        n_p[0] = (kf_tri_ver[0] - kf_tri_ver[1]).cross(kf_tri_ver[2] - kf_tri_ver[1]);
        n_p[1] = (kf_tri_ver[1] - kf_tri_ver[0]).cross(kf_tri_ver[2] - kf_tri_ver[0]);
        n_p[2] = (kf_tri_ver[2] - kf_tri_ver[1]).cross(kf_tri_ver[0] - kf_tri_ver[1]);

        Eigen::Vector3f pw2mpw1[3];
        pw2mpw1[0] = (kf_tri_ver[2] - kf_tri_ver[1]);
        pw2mpw1[1] = (kf_tri_ver[2] - kf_tri_ver[0]);
        pw2mpw1[2] = (kf_tri_ver[0] - kf_tri_ver[1]);

        float n_p_dot_point[3];
        n_p_dot_point[0] = n_p[0].dot(kf_tri_ver[1]);
        n_p_dot_point[1] = n_p[1].dot(kf_tri_ver[0]);
        n_p_dot_point[2] = n_p[2].dot(kf_tri_ver[1]);

        Eigen::Vector3f pr_p[3];
        pr_p[0] = kf_tri_ver[1];
        pr_p[1] = kf_tri_ver[0];
        pr_p[2] = kf_tri_ver[1];

        Eigen::Vector3f d_n_d_z[3];
        float d_z_d_iz[3];
        for (int i = 0; i < 3; i++)
        {
            d_n_d_z[i] = kf_tri_ver[i].cross(pw2mpw1[i]);
            d_z_d_iz[i] = -1.0 / (kf_tri_idepth[i] * kf_tri_idepth[i]);
        }

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
                if (f_pix(0) < 0.0 || f_pix(0) >= cam.width[lvl] || f_pix(1) < 0.0 || f_pix(1) >= cam.height[lvl])
                    continue;

                Eigen::Vector2f lambda = T_inv * (f_pix - f_tri_pix[2]);

                if (lambda(0) < 0.0 || lambda(1) < 0.0 || (1.0 - lambda(0) - lambda(1)) < 0.0)
                    continue;

                Eigen::Vector2f kf_pix = lambda(0) * kf_tri_pix[0] + lambda(1) * kf_tri_pix[1] + (1 - lambda(0) - lambda(1)) * kf_tri_pix[2];
                if (kf_pix(0) < 0.0 || kf_pix(0) >= cam.width[lvl] || kf_pix(1) < 0.0 || kf_pix(1) >= cam.height[lvl])
                    continue;

                Eigen::Vector3f kf_ray = lambda(0) * kf_tri_ray[0] + lambda(1) * kf_tri_ray[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ray[2];
                // Eigen::Vector3f kf_ver = lambda(0) * kf_tri_ver[0] + lambda(1) * kf_tri_ver[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ver[2];

                // Eigen::Vector3f f_ray = lambda(0) * f_tri_ray[0] + lambda(1) * f_tri_ray[1] + (1 - lambda(0) - lambda(1)) * f_tri_ray[2];
                Eigen::Vector3f f_ver = lambda(0) * f_tri_ver[0] + lambda(1) * f_tri_ver[1] + (1 - lambda(0) - lambda(1)) * f_tri_ver[2];

                float kf_i = float(keyframe.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame.image.get(f_pix(1), f_pix(0), lvl));
                float dx = frame.dx.get(f_pix(1), f_pix(0), lvl);
                float dy = frame.dy.get(f_pix(1), f_pix(0), lvl);

                if (kf_i == keyframe.image.nodata || f_i == frame.image.nodata || dx == frame.dx.nodata || dy == frame.dy.nodata)
                    continue;

                Eigen::Vector2f d_f_i_d_pix(dx, dy);

                Eigen::Vector3f d_f_i_d_f_ver;
                d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx[lvl] / f_ver(2);
                d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy[lvl] / f_ver(2);
                // d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) / f_ver(2) + d_f_i_d_f_ver(1) * f_ver(1) / f_ver(2));
                d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                Eigen::Vector3f d_f_ver_d_kf_depth = frame.pose.rotationMatrix() * kf_ray;

                float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                float error = f_i - kf_i;

                hg_map.error += error;
                hg_map.count += 1;

                float J[3];
                for (int i = 0; i < 3; i++)
                {
                    float n_p_dot_ray = n_p[i].dot(kf_ray);
                    float d_kf_depth_d_z = d_n_d_z[i].dot(pr_p[i]) / n_p_dot_ray - n_p_dot_point[i] * d_n_d_z[i].dot(kf_ray) / (n_p_dot_ray * n_p_dot_ray);
                    float d_f_i_d_z = d_f_i_d_kf_depth * d_kf_depth_d_z * d_z_d_iz[i];
                    J[i] = d_f_i_d_z;
                }

                for (int i = 0; i < 3; i++)
                {
                    // if the jacobian is 0
                    // we really cannot say anything about the depth
                    // can make the hessian non-singular
                    if (J[i] == 0)
                        continue;
                    hg_map.G_depth(vertex_id[i]) += J[i] * error; // / (cam.width[lvl]*cam.height[lvl]);
                    hg_map.G_count(vertex_id[i]) += 1;
                    for (int j = i; j < 3; j++)
                    {
                        float jj = J[i] * J[j]; // / (cam.width[lvl]*cam.height[lvl]);
                        hg_map.H_depth.coeffRef(vertex_id[i], vertex_id[j]) += jj;
                        hg_map.H_depth.coeffRef(vertex_id[j], vertex_id[i]) += jj;
                    }
                }
            }
        }
    }

    return hg_map;
}

HGPoseMap rayDepthMeshSceneCPU::computeHGPoseMap(frameCPU &frame, int lvl)
{
    HGPoseMap hg_posemap;

    // for each triangle
    for (std::size_t index = 0; index < scene_indices.size(); index += 3)
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
            int vertex_i = scene_indices[index + vertex];

            vertex_id[vertex] = vertex_i;

            // scene vertices are ray + idepth in keyframe coordinates
            kf_tri_ray[vertex](0) = scene_vertices[vertex_i * 3];
            kf_tri_ray[vertex](1) = scene_vertices[vertex_i * 3 + 1];
            kf_tri_ray[vertex](2) = 1.0;
            kf_tri_idepth[vertex] = scene_vertices[vertex_i * 3 + 2];

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

        Eigen::Vector3f n_p[3];
        n_p[0] = (kf_tri_ver[0] - kf_tri_ver[1]).cross(kf_tri_ver[2] - kf_tri_ver[1]);
        n_p[1] = (kf_tri_ver[1] - kf_tri_ver[0]).cross(kf_tri_ver[2] - kf_tri_ver[0]);
        n_p[2] = (kf_tri_ver[2] - kf_tri_ver[1]).cross(kf_tri_ver[0] - kf_tri_ver[1]);

        Eigen::Vector3f pw2mpw1[3];
        pw2mpw1[0] = (kf_tri_ver[2] - kf_tri_ver[1]);
        pw2mpw1[1] = (kf_tri_ver[2] - kf_tri_ver[0]);
        pw2mpw1[2] = (kf_tri_ver[0] - kf_tri_ver[1]);

        float n_p_dot_point[3];
        n_p_dot_point[0] = n_p[0].dot(kf_tri_ver[1]);
        n_p_dot_point[1] = n_p[1].dot(kf_tri_ver[0]);
        n_p_dot_point[2] = n_p[2].dot(kf_tri_ver[1]);

        Eigen::Vector3f pr_p[3];
        pr_p[0] = kf_tri_ver[1];
        pr_p[1] = kf_tri_ver[0];
        pr_p[2] = kf_tri_ver[1];

        Eigen::Vector3f d_n_d_z[3];
        float d_z_d_iz[3];
        for (int i = 0; i < 3; i++)
        {
            d_n_d_z[i] = kf_tri_ver[i].cross(pw2mpw1[i]);
            d_z_d_iz[i] = -1.0 / (kf_tri_idepth[i] * kf_tri_idepth[i]);
        }

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
                if (f_pix(0) < 0.0 || f_pix(0) >= cam.width[lvl] || f_pix(1) < 0.0 || f_pix(1) >= cam.height[lvl])
                    continue;

                Eigen::Vector2f lambda = T_inv * (f_pix - f_tri_pix[2]);

                if (lambda(0) < 0.0 || lambda(1) < 0.0 || (1.0 - lambda(0) - lambda(1)) < 0.0)
                    continue;

                Eigen::Vector2f kf_pix = lambda(0) * kf_tri_pix[0] + lambda(1) * kf_tri_pix[1] + (1 - lambda(0) - lambda(1)) * kf_tri_pix[2];
                if (kf_pix(0) < 0.0 || kf_pix(0) >= cam.width[lvl] || kf_pix(1) < 0.0 || kf_pix(1) >= cam.height[lvl])
                    continue;

                Eigen::Vector3f kf_ray = lambda(0) * kf_tri_ray[0] + lambda(1) * kf_tri_ray[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ray[2];
                // Eigen::Vector3f kf_ver = lambda(0) * kf_tri_ver[0] + lambda(1) * kf_tri_ver[1] + (1 - lambda(0) - lambda(1)) * kf_tri_ver[2];

                // Eigen::Vector3f f_ray = lambda(0) * f_tri_ray[0] + lambda(1) * f_tri_ray[1] + (1 - lambda(0) - lambda(1)) * f_tri_ray[2];
                Eigen::Vector3f f_ver = lambda(0) * f_tri_ver[0] + lambda(1) * f_tri_ver[1] + (1 - lambda(0) - lambda(1)) * f_tri_ver[2];

                float kf_i = float(keyframe.image.get(kf_pix(1), kf_pix(0), lvl));
                float f_i = float(frame.image.get(f_pix(1), f_pix(0), lvl));
                float dx = frame.dx.get(f_pix(1), f_pix(0), lvl);
                float dy = frame.dy.get(f_pix(1), f_pix(0), lvl);

                if (kf_i == keyframe.image.nodata || f_i == frame.image.nodata || dx == frame.dx.nodata || dy == frame.dy.nodata)
                    continue;

                Eigen::Vector2f d_f_i_d_pix(dx, dy);

                Eigen::Vector3f d_f_i_d_f_ver;
                d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx[lvl] / f_ver(2);
                d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy[lvl] / f_ver(2);
                // d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) / f_ver(2) + d_f_i_d_f_ver(1) * f_ver(1) / f_ver(2));
                d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                Eigen::Vector3f d_f_i_d_tra = d_f_i_d_f_ver;
                Eigen::Vector3f d_f_i_d_rot = Eigen::Vector3f(-f_ver(2) * d_f_i_d_f_ver(1) + f_ver(1) * d_f_i_d_f_ver(2), f_ver(2) * d_f_i_d_f_ver(0) - f_ver(0) * d_f_i_d_f_ver(2), -f_ver(1) * d_f_i_d_f_ver(0) + f_ver(0) * d_f_i_d_f_ver(1));

                Eigen::Vector3f d_f_ver_d_kf_depth = frame.pose.rotationMatrix() * kf_ray;

                float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                float error = f_i - kf_i;

                hg_map.error += error;
                hg_map.count += 1;

                float J[3];
                for (int i = 0; i < 3; i++)
                {
                    float n_p_dot_ray = n_p[i].dot(kf_ray);
                    float d_kf_depth_d_z = d_n_d_z[i].dot(pr_p[i]) / n_p_dot_ray - n_p_dot_point[i] * d_n_d_z[i].dot(kf_ray) / (n_p_dot_ray * n_p_dot_ray);
                    float d_f_i_d_z = d_f_i_d_kf_depth * d_kf_depth_d_z * d_z_d_iz[i];
                    J[i] = d_f_i_d_z;
                }

                for (int i = 0; i < 3; i++)
                {
                    // if the jacobian is 0
                    // we really cannot say anything about the depth
                    // can make the hessian non-singular
                    if (J[i] == 0)
                        continue;
                    hg_posemap.G(vertex_id[i]) += J[i] * error; // / (cam.width[lvl]*cam.height[lvl]);
                    hg_posemap.count(vertex_id[i]) += 1;
                    for (int j = i; j < 3; j++)
                    {
                        float jj = J[i] * J[j]; // / (cam.width[lvl]*cam.height[lvl]);
                        hg_map.H.coeffRef(vertex_id[i], vertex_id[j]) += jj;
                        hg_map.H.coeffRef(vertex_id[j], vertex_id[i]) += jj;
                    }
                }
            }
        }
    }

    return hg_posemap;
}

float rayDepthMeshSceneCPU::errorRegu()
{
    float error = 0;
    for (int i = 0; i < int(scene_indices.size()); i += 3)
    {
        // std::cout << "triangle" << std::endl;
        float idepth[3];
        for (int j = 0; j < 3; j++)
        {
            int vertexIndex = scene_indices.at(i + j);
            idepth[j] = scene_vertices.at(vertexIndex * 3 + 2);
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

void rayDepthMeshSceneCPU::HGRegu(HGMap &hgmap)
{
    for (int i = 0; i < int(scene_indices.size()); i += 3)
    {
        int vertexIndex[3];
        float idepth[3];
        for (int j = 0; j < 3; j++)
        {
            vertexIndex[j] = scene_indices.at(i + j);
            idepth[j] = scene_vertices.at(vertexIndex[j] * 3 + 2);
        }
        float diff1 = idepth[0] - idepth[1];
        float diff2 = idepth[0] - idepth[2];
        float diff3 = idepth[1] - idepth[2];

        float J1[3] = {1.0, -1.0, 0.0};
        float J2[3] = {1.0, 0.0, -1.0};
        float J3[3] = {0.0, 1.0, -1.0};

        for (int j = 0; j < 3; j++)
        {
            if (hgmap.G_count(vertexIndex[j]) == 0)
                continue;
            hgmap.G_depth(vertexIndex[j]) += (MESH_REGU / (VERTEX_HEIGH * VERTEX_WIDTH)) * (diff1 * J1[j] + diff2 * J2[j] + diff3 * J3[j]);
            // J_joint(MAX_FRAMES * 6 + vertexIndex[j]) += (MESH_REGU / (VERTEX_HEIGH * VERTEX_WIDTH)) * (diff1 * J1[j] + diff2 * J2[j] + diff3 * J3[j]);
            for (int k = 0; k < 3; k++)
            {
                hgmap.H_depth.coeffRef(vertexIndex[j], vertexIndex[k]) += (MESH_REGU / (VERTEX_WIDTH * VERTEX_HEIGH)) * (J1[j] * J1[k] + J2[j] * J2[k] + J3[j] * J3[k]);
                // H_joint(MAX_FRAMES * 6 + vertexIndex[j], MAX_FRAMES * 6 + vertexIndex[k]) += (MESH_REGU / (VERTEX_WIDTH * VERTEX_HEIGH)) * (J1[j] * J1[k] + J2[j] * J2[k] + J3[j] * J3[k]);
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


void meshVO::optPose(frameCPU &frame)
{
    int maxIterations[10] = {5, 20, 50, 100, 100, 100, 100, 100, 100, 100};

    tic_toc t;

    for (int lvl = 4; lvl >= 1; lvl--)
    {
        std::cout << "*************************lvl " << lvl << std::endl;
        t.tic();
        float last_error = computeError(frame, lvl);

        std::cout << "init error " << last_error << " time " << t.toc() << std::endl;

        for (int it = 0; it < maxIterations[lvl]; it++)
        {
            t.tic();
            HGPose hgpose = computeHGPose(frame, lvl);
            std::cout << "HGPose time " << t.toc() << std::endl;

            float lambda = 0.0;
            int n_try = 0;
            while (true)
            {
                Eigen::Matrix<float, 6, 6> H_lambda;
                H_lambda = hgpose.H;

                for (int j = 0; j < 6; j++)
                    acc_H_pose_lambda(j, j) *= 1.0 + lambda;

                Eigen::Matrix<float, 6, 1> inc = H_lambda.ldlt().solve(hgpose.G);

                Sophus::SE3f old_pose = frame.pose;
                // Sophus::SE3f new_pose = frame.pose * Sophus::SE3f::exp(inc_pose);
                Sophus::SE3f new_pose = frame.pose * Sophus::SE3f::exp(inc).inverse();
                // Sophus::SE3f new_pose = Sophus::SE3f::exp(inc_pose).inverse() * frame.pose;

                frame.pose = new_pose;

                t.tic();
                float error = computeError(frame, lvl);
                std::cout << "new error " << error << " time " << t.toc() << std::endl;

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
                        // if converged, do next level
                        it = maxIterations[lvl];
                    }
                    // if update accepted, do next iteration
                    break;
                }
                else
                {
                    frame.pose = old_pose;

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


void rayDepthMeshSceneCPU::optMap(frameCPU &frame)
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
            for (int i = 0; i < hgmap.G.size(); i++)
            {
                int gcount = hgmap.count(i);
                if (gcount == 0)
                    hgmap.H.coeffRef(i, i) = 1.0;
                // else
                //     hgmap.G_depth(i) /= float(gcount);
            }

            std::cout << "HG time " << t.toc() << std::endl;

            int n_try = 0;
            while (true)
            {
                Eigen::SparseMatrix<float> H_lambda = hgmap.H;

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
                Eigen::VectorXf inc = -solver.solve(hgmap.G);
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
                    if (inc(index) != inc(index))
                    {
                        std::cout << "some nand in inc_depth " << std::endl;
                        continue;
                    }
                    new_vertices[index * 3 + 2] = best_vertices[index * 3 + 2] + inc(index);
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

void rayDepthMeshSceneCPU::optPoseMap(frameCPU &frame)
{
    tic_toc t;

    for (int lvl = 0; lvl >= 0; lvl--)
    {
        t.tic();

        std::vector<float> best_vertices;

        //best_vertices = scene.scene_vertices;

        float last_error = computeErrorCPU(frame, lvl); // + errorMesh();
        std::cout << "initial error time " << t.toc() << std::endl;
        std::cout << "lvl " << lvl << " initial error " << last_error << std::endl;

        int maxIterations = 100;
        float lambda = 0.0;

        for (int it = 0; it < maxIterations; it++)
        {
            t.tic();

            HGPoseMap hgposemap = HJPoseMap(frame);
            // HJPoseMapStackGPU(lvl);
            // HJMesh();

            std::cout << "HJ time " << t.toc() << std::endl;

            int n_try = 0;
            while (true)
            {
                t.tic();

                Eigen::MatrixXf H_lambda;// = H_joint;

                for (int j = 0; j < H_joint_lambda.rows(); j++)
                {
                    H_joint_lambda(j, j) *= (1.0 + lambda);
                }

                // inc_joint = H_joint_lambda.ldlt().solve(J_joint);
                // inc_joint = H_joint_lambda.colPivHouseholderQr().solve(J_joint);

                //for (int j = 0; j < int(J_joint.size()); j++)
                for(int j = 0; j < 10; j++)
                {
                    float h = H_joint_lambda(j, j);
                    if (h > 0.0)// && abs(J_joint(j)) > 0.0)
                    // if(J_joint(j) > 0.0)
                    {
                        //inc_joint(j) = J_joint(j) / h;
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

                // float error = errorStackGPU(lvl);// + errorMesh();

                std::cout << "new error time " << t.toc() << std::endl;
                // std::cout << "lvl " << lvl << " new error " << error << std::endl;

                // if (error < last_error)
                if (true)
                {
                    // accept update, decrease lambda
                    std::cout << "update accepted " << std::endl;

                    /*
                    for (int i = 0; i < MAX_FRAMES; i++)
                        bestPoses[i] = frameDataStack[i].pose;
                    best_vertices = scene_vertices;
                    */

                    float p = 0.0; // error / last_error;

                    if (lambda < 0.2f)
                        lambda = 0.0f;
                    else
                        lambda *= 0.5;

                    last_error = 0.0; // error;

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
                    //std::cout << "update rejected " << lambda << " " << inc_joint.dot(inc_joint) << std::endl;

                    //if (inc_joint.dot(inc_joint) < 1e-16)
                    if(true)
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
