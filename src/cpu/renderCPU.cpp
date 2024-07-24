#include "cpu/renderCPU.h"
#include <math.h>
#include "utils/tictoc.h"



void renderCPU::renderImage(SceneBase &scene, camera &cam, frameCPU &kframe, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
{
    z_buffer.set(z_buffer.nodata, lvl);

    // std::unique_ptr<SceneBase> kframeMesh = scene.clone();
    std::unique_ptr<SceneBase> frameMesh = scene.clone();
    // kframeMesh->transform(kframe.pose);
    frameMesh->transform(pose);
    Sophus::SE3f kfTofPose = pose * kframe.pose.inverse();
    Sophus::SE3f fTokfPose = kfTofPose.inverse();

    std::vector<unsigned int> ids = frameMesh->getShapesIds();

    // for each triangle
    for (auto t_id : ids)
    {
        // Polygon kf_pol = mesh.getPolygon(t_id);
        //  if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //      continue;
        // if (kf_tri.getArea() < 1.0)
        //     continue;

        auto f_pol = frameMesh->getShape(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (f_pol->getArea() < 0.0)
            continue;

        std::array<int, 4> minmax = f_pol->getScreenBounds(cam);

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
                    continue;
                Eigen::Vector3f f_ray = cam.pixToRay(f_pix);

                f_pol->prepareForRay(f_ray);
                if (!f_pol->rayHitsShape())
                    continue;

                float f_depth = f_pol->getRayDepth();
                if (f_depth <= 0.0)
                    continue;

                Eigen::Vector3f f_ver = f_ray * f_depth;

                Eigen::Vector3f kf_ver = fTokfPose * f_ver;
                if (kf_ver(2) <= 0.0)
                    continue;
                Eigen::Vector3f kf_ray = kf_ver / kf_ver(2);
                Eigen::Vector2f kf_pix = cam.rayToPix(kf_ray);

                if (!cam.isPixVisible(kf_pix))
                    continue;

                auto kf_i = kframe.image.get(kf_pix(1), kf_pix(0), lvl);
                if (kf_i == kframe.image.nodata)
                    continue;

                float z_depth = z_buffer.get(y, x, lvl);
                if (z_depth < f_depth && z_depth != z_buffer.nodata)
                    continue;

                buffer.set(kf_i, y, x, lvl);

                z_buffer.set(f_depth, y, x, lvl);
            }
        }
    }
}

void renderCPU::renderImage(dataCPU<float> &poseIdepth, camera &cam, frameCPU &kframe, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
{
    Sophus::SE3f kfTofPose = pose * kframe.pose.inverse();
    Sophus::SE3f fTokfPose = kfTofPose.inverse();

    for (int y = cam.window_min_y; y <= cam.window_max_y; y++)
    {
        for (int x = cam.window_min_x; x <= cam.window_max_x; x++)
        {
            Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
            if (!cam.isPixVisible(f_pix))
                continue;
            Eigen::Vector3f f_ray = cam.pixToRay(f_pix);

            float f_idepth = poseIdepth.get(y, x, lvl);
            if (f_idepth <= 0.0 || f_idepth == poseIdepth.nodata)
                continue;

            Eigen::Vector3f f_ver = f_ray / f_idepth;

            Eigen::Vector3f kf_ver = fTokfPose * f_ver;
            if (kf_ver(2) <= 0.0)
                continue;
            Eigen::Vector3f kf_ray = kf_ver / kf_ver(2);
            Eigen::Vector2f kf_pix = cam.rayToPix(kf_ray);

            if (!cam.isPixVisible(kf_pix))
                continue;

            auto kf_i = kframe.image.get(kf_pix(1), kf_pix(0), lvl);
            if (kf_i == kframe.image.nodata)
                continue;

            buffer.set(kf_i, y, x, lvl);
        }
    }
}

void renderCPU::renderDebug(SceneBase &scene, camera &cam, frameCPU &frame, dataCPU<float> &buffer, int lvl)
{
    std::unique_ptr<SceneBase> frameMesh = scene.clone();
    frameMesh->transform(frame.pose);

    std::vector<unsigned int> ids = frameMesh->getShapesIds();

    // for each triangle
    for (auto t_id : ids)
    {
        // Triangle kf_tri = keyframeMesh.triangles[index];
        //  if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
        //      continue;
        // if (kf_tri.isBackFace())
        //     continue;
        auto f_pol = frameMesh->getShape(t_id);
        // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
        //     continue;
        if (f_pol->getArea() < 0.0)
            continue;

        std::array<int, 4> minmax = f_pol->getScreenBounds(cam);

        for (int y = minmax[2]; y <= minmax[3]; y++)
        {
            for (int x = minmax[0]; x <= minmax[1]; x++)
            {
                Eigen::Vector2f f_pix = Eigen::Vector2f(x, y);
                if (!cam.isPixVisible(f_pix))
                    continue;

                Eigen::Vector3f f_ray = cam.pixToRay(f_pix);

                f_pol->prepareForRay(f_ray);
                if (!f_pol->rayHitsShape())
                    continue;

                bool isLine = f_pol->isEdge();

                float f_i = frame.image.get(y, x, lvl);
                f_i /= 255.0;

                // z buffer
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                if (isLine)
                    buffer.set(1.0, y, x, lvl);
                else
                    buffer.set(f_i, y, x, lvl);
            }
        }
    }
}

