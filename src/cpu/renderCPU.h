#pragma once

#include <Eigen/Core>
#include <thread>

#include "common/camera.h"
#include "common/HGPose.h"
#include "common/HGMapped.h"
#include "common/Error.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/SceneBase.h"
#include "cpu/frameCPU.h"
#include "threadpoolCPU.h"
#include "params.h"

class renderCPU
{
public:
    renderCPU(unsigned int width, unsigned int height)
        : z_buffer(width, height, -1),
          pool()
    {
    }

    void renderImage(SceneBase &scene, camera &cam, frameCPU &kframe, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderImage(dataCPU<float> &poseIdepth, camera &cam, frameCPU &kframe, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);

    void renderDebug(SceneBase &scene, camera &cam, frameCPU &pose, dataCPU<float> &buffer, int lvl);

    template <int DoF>
    void renderJMap(SceneBase &scene, camera &cam, frameCPU &kframe, frameCPU &frame, dataCPU<std::array<float, DoF>> &jmap_buffer, dataCPU<float> &e_buffer, dataCPU<std::array<int, DoF>> &pId_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        //float min_area = 0.0 * (float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) / 16;
        // float min_angle = M_PI / 64.0;

        std::unique_ptr<SceneBase> kframeMesh = scene.clone();
        std::unique_ptr<SceneBase> frameMesh = scene.clone();

        kframeMesh->transform(kframe.pose);
        frameMesh->transform(frame.pose);

        renderJMapWindow<DoF>(kframeMesh.get(), frameMesh.get(), cam, &kframe, &frame, &jmap_buffer, &e_buffer, &pId_buffer, lvl);
    }

    /*
    template <int DoF>
    void renderJMap(SceneBase &scene, camera &cam, frameCPU &kframe, frameCPU &frame, dataCPU<std::array<float, DoF>> &jmap_buffer, dataCPU<float> &e_buffer, dataCPU<std::array<int, DoF>> &pId_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        float min_area = 0.0 * (float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) / 16;
        // float min_angle = M_PI / 64.0;

        std::unique_ptr<SceneBase> kframeMesh = scene.clone();
        std::unique_ptr<SceneBase> frameMesh = scene.clone();

        kframeMesh->transform(kframe.pose);
        frameMesh->transform(frame.pose);

        Sophus::SE3f kfTofPose = frame.pose * kframe.pose.inverse();
        Sophus::SE3f fTokfPose = kfTofPose.inverse();

        // for each triangle
        std::vector<unsigned int> t_ids = kframeMesh->getShapesIds();
        for (auto t_id : t_ids)
        {
            std::vector<unsigned int> p_ids = kframeMesh->getShapeParamsIds(t_id);

            auto kf_pol = kframeMesh->getShape(t_id);

            // if (kf_tri_3d.vertices[0](2) <= 0.0 || kf_tri_3d.vertices[1](2) <= 0.0 || kf_tri_3d.vertices[2](2) <= 0.0)
            //     continue;
            if (kf_pol->getArea() < min_area)
                continue;
            // std::array<float, 3> kf_tri_angles = kf_tri_2d.getAngles();
            // if (fabs(kf_tri_angles[0]) < min_angle || fabs(kf_tri_angles[1]) < min_angle || fabs(kf_tri_angles[2]) < min_angle)
            //     continue;

            auto f_pol = frameMesh->getShape(t_id);

            // if (f_tri_3d.vertices[0](2) <= 0.0 || f_tri_3d.vertices[1](2) <= 0.0 || f_tri_3d.vertices[2](2) <= 0.0)
            //     continue;
            if (f_pol->getArea() < min_area)
                continue;
            // std::array<float, 3> f_tri_angles = f_tri_2d.getAngles();
            // if (fabs(f_tri_angles[0]) < min_angle || fabs(f_tri_angles[1]) < min_angle || fabs(f_tri_angles[2]) < min_angle)
            //     continue;

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

                    // z-buffer
                    float l_idepth = z_buffer.get(y, x, lvl);
                    if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                        continue;

                    Eigen::Vector3f f_ver = f_ray * f_depth;
                    Eigen::Vector3f kf_ver = fTokfPose * f_ver;
                    if (kf_ver(2) <= 0.0)
                        continue;

                    Eigen::Vector3f kf_ray = kf_ver / kf_ver(2);

                    kf_pol->prepareForRay(kf_ray);
                    if (!kf_pol->rayHitsShape())
                        continue;

                    Eigen::Vector2f kf_pix = cam.rayToPix(kf_ray);
                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = kframe.image.get(kf_pix(1), kf_pix(0), lvl);
                    auto f_i = frame.image.get(y, x, lvl);
                    float dx = frame.dx.get(y, x, lvl);
                    float dy = frame.dy.get(y, x, lvl);

                    if (kf_i == kframe.image.nodata || f_i == frame.image.nodata || dx == frame.dx.nodata || dy == frame.dy.nodata)
                        continue;

                    Eigen::Vector2f d_f_i_d_pix(dx, dy);

                    float error = f_i - kf_i;

                    Eigen::Vector3f d_f_i_d_f_ver = cam.d_f_i_d_f_ver(d_f_i_d_pix, f_ver);

                    // Eigen::Vector3f d_f_i_d_f_ver;
                    // d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    // d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    // d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                    Eigen::Vector3f d_f_ver_d_kf_depth = kfTofPose.rotationMatrix() * kf_ray;
                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    // this could be the jacobian of the depth of the 3 vertices in a triangle
                    // or the jacobian of the normal + depth of a surfel
                    std::vector<float> Jacobian = kf_pol->getJacobian(d_f_i_d_kf_depth);

                    std::array<float, DoF> jacs = jmap_buffer.nodata;
                    std::array<int, DoF> ids = pId_buffer.nodata;
                    for (size_t i = 0; i < p_ids.size(); i++)
                    {
                        if (i >= DoF)
                            break;
                        jacs[i] = Jacobian[i];
                        ids[i] = p_ids[i];
                    }

                    e_buffer.set(error, y, x, lvl);
                    jmap_buffer.set(jacs, y, x, lvl);
                    pId_buffer.set(ids, y, x, lvl);
                }
            }
        }
    }
    */

    template <int DoF>
    void renderJPoseMap(SceneBase &mesh, camera &cam, frameCPU &kframe, frameCPU &frame, dataCPU<std::array<float, 6>> &jpose_buffer, dataCPU<std::array<float, DoF>> &jmap_buffer, dataCPU<float> &e_buffer, dataCPU<std::array<int, DoF>> &pId_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        float min_area = 0.0 * (float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) / 16.0;
        // float min_angle = M_PI / 64.0;

        std::unique_ptr<SceneBase> kframeScene = mesh.clone();
        std::unique_ptr<SceneBase> frameScene = mesh.clone();

        kframeScene->transform(kframe.pose);
        frameScene->transform(frame.pose);

        Sophus::SE3f kfTofPose = frame.pose * kframe.pose.inverse();
        Sophus::SE3f fTokfPose = kfTofPose.inverse();

        // for each triangle
        std::vector<unsigned int> t_ids = kframeScene->getShapesIds();
        for (auto t_id : t_ids)
        {
            std::vector<unsigned int> p_ids = kframeScene->getShapeParamsIds(t_id);

            auto kf_pol = kframeScene->getShape(t_id);
            // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
            //     continue;
            if (kf_pol->getArea() < min_area)
                continue;
            // std::array<float, 3> kf_tri_angles = kf_tri_2d.getAngles();
            // if (fabs(kf_tri_angles[0]) < min_angle || fabs(kf_tri_angles[1]) < min_angle || fabs(kf_tri_angles[2]) < min_angle)
            //    continue;

            auto f_pol = frameScene->getShape(t_id);
            // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
            //     continue;
            if (f_pol->getArea() < min_area)
                continue;
            // std::array<float, 3> f_tri_angles = f_tri_2d.getAngles();
            // if (fabs(f_tri_angles[0]) < min_angle || fabs(f_tri_angles[1]) < min_angle || fabs(f_tri_angles[2]) < min_angle)
            //    continue;

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

                    // z-buffer
                    float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                    if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                        continue;

                    Eigen::Vector3f f_ver = f_ray * f_depth;
                    Eigen::Vector3f kf_ver = fTokfPose * f_ver;
                    if (kf_ver(2) <= 0.0)
                        continue;
                    Eigen::Vector3f kf_ray = kf_ver / kf_ver(2);

                    kf_pol->prepareForRay(kf_ray);
                    if (!kf_pol->rayHitsShape())
                        continue;

                    Eigen::Vector2f kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = kframe.image.get(kf_pix(1), kf_pix(0), lvl);
                    auto f_i = frame.image.get(y, x, lvl);
                    float dx = frame.dx.get(y, x, lvl);
                    float dy = frame.dy.get(y, x, lvl);

                    if (kf_i == kframe.image.nodata || f_i == frame.image.nodata || dx == frame.dx.nodata || dy == frame.dy.nodata)
                        continue;

                    Eigen::Vector2f d_f_i_d_pix(dx, dy);

                    Eigen::Vector3f d_f_i_d_f_ver;
                    d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                    std::array<float, 3> d_f_i_d_tra = {d_f_i_d_f_ver(0), d_f_i_d_f_ver(1), d_f_i_d_f_ver(2)};
                    std::array<float, 3> d_f_i_d_rot = {-f_ver(2) * d_f_i_d_f_ver(1) + f_ver(1) * d_f_i_d_f_ver(2), f_ver(2) * d_f_i_d_f_ver(0) - f_ver(0) * d_f_i_d_f_ver(2), -f_ver(1) * d_f_i_d_f_ver(0) + f_ver(0) * d_f_i_d_f_ver(1)};

                    std::array<float, 6> jpose = {d_f_i_d_tra[0], d_f_i_d_tra[1], d_f_i_d_tra[2], d_f_i_d_rot[0], d_f_i_d_rot[1], d_f_i_d_rot[2]};

                    jpose_buffer.set(jpose, y, x, lvl);

                    Eigen::Vector3f d_f_ver_d_kf_depth = kfTofPose.rotationMatrix() * kf_ray;
                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    std::vector<float> Jacobian = kf_pol->getJacobian(d_f_i_d_kf_depth);

                    float error = f_i - kf_i;

                    std::array<float, DoF> jacs = jmap_buffer.nodata;
                    std::array<int, DoF> ids = pId_buffer.nodata;
                    for (size_t i = 0; i < p_ids.size(); i++)
                    {
                        if (i >= DoF)
                            break;
                        jacs[i] = Jacobian[i];
                        ids[i] = p_ids[i];
                    }

                    e_buffer.set(error, y, x, lvl);
                    jmap_buffer.set(jacs, y, x, lvl);
                    pId_buffer.set(ids, y, x, lvl);
                }
            }
        }
    }

    void renderJPose(SceneBase &scene, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<std::array<float, 6>> &jpose_buffer, dataCPU<float> &e_buffer, int lvl)
    {
        /*
        std::unique_ptr<SceneBase> kframeMesh = scene.clone();
        std::unique_ptr<SceneBase> frameMesh = scene.clone();

        kframeMesh->transform(kframe.pose);
        frameMesh->transform(frame.pose);
        */
    }

    void renderJPose(dataCPU<float> &frameIdepth, camera &cam, frameCPU &kframe, frameCPU &frame, dataCPU<std::array<float, 6>> &jpose_buffer, dataCPU<float> &e_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);
        renderJPoseWindow(&frameIdepth, cam, &kframe, &frame, &jpose_buffer, &e_buffer, lvl);
    }

    void renderJPoseParallel(dataCPU<float> &frame2Idepth, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<std::array<float, 6>> &jpose_buffer, dataCPU<float> &e_buffer, int lvl)
    {
        int divi_y = 16;
        int divi_x = 1;

        std::array<int, 2> windowSize;
        windowSize[0] = cam.width / divi_x;
        windowSize[1] = cam.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * windowSize[0];
                int max_x = (tx + 1) * windowSize[0];
                int min_y = ty * windowSize[1];
                int max_y = (ty + 1) * windowSize[1];

                camera cam_window = cam;
                cam_window.setWindow(min_x, max_x, min_y, max_y);

                // renderJPoseWindow(&frame2Idepth, cam_window, &frame1, &frame2, &jtra_buffer, &jrot_buffer, &e_buffer, lvl);
                pool.enqueue(std::bind(&renderCPU::renderJPoseWindow, this, &frame2Idepth, cam_window, &frame1, &frame2, &jpose_buffer, &e_buffer, lvl));
            }
        }

        pool.waitUntilDone();
    }

    void renderIdepth(SceneBase &scene, camera &cam, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        frameMesh = scene.clone();
        frameMesh->transform(pose);

        renderIdepthWindow(cam, &buffer, lvl);
    }

    void renderIdepthParallel(SceneBase &scene, camera &cam, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        frameMesh = scene.clone();
        frameMesh->transform(pose);

        int divi_y = 16;
        int divi_x = 1;

        std::array<int, 2> windowSize;
        windowSize[0] = cam.width / divi_x;
        windowSize[1] = cam.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * windowSize[0];
                int max_x = (tx + 1) * windowSize[0];
                int min_y = ty * windowSize[1];
                int max_y = (ty + 1) * windowSize[1];

                camera cam_window = cam;
                cam_window.setWindow(min_x, max_x, min_y, max_y);

                // renderIdepthWindow(cam_window, &buffer, lvl);
                pool.enqueue(std::bind(&renderCPU::renderIdepthWindow, this, cam_window, &buffer, lvl));
            }
        }

        pool.waitUntilDone();
    }

private:
    /*
        void renderJPoseWindow( camera &cam, frameCPU &kframe, frameCPU &frame, dataCPU<std::array<float, 3>> &jtra_buffer, dataCPU<std::array<float, 3>> &jrot_buffer, dataCPU<float> &e_buffer, int lvl)
        {
            float min_area = 0.0 * (float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) / 16;
            // float min_angle = M_PI / 64.0;

            Sophus::SE3f kfTofPose = frame.pose * kframe.pose.inverse();
            Sophus::SE3f fTokfPose = kfTofPose.inverse();

            // for each triangle
            std::vector<unsigned int> t_ids = kframeMesh->getShapesIds();
            for (auto t_id : t_ids)
            {
                auto kf_pol = kframeMesh->getShape(t_id);
                // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
                //     continue;
                if (kf_pol->getArea() < min_area)
                    continue;
                // std::array<float, 3> kf_angle = kf_tri.getAngles();
                // if (fabs(kf_angle[0]) < min_angle || fabs(kf_angle[1]) < min_angle || fabs(kf_angle[2]) < min_angle)
                //    continue;
                auto f_pol = frameMesh->getShape(t_id);
                // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
                //     continue;
                if (f_pol->getArea() < min_area)
                    continue;
                // std::array<float, 3> f_angle = f_tri_2d.getAngles();
                // if (fabs(f_angle[0]) < min_angle || fabs(f_angle[1]) < min_angle || fabs(f_angle[2]) < min_angle)
                //    continue;

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

                        // z-buffer
                        float l_depth = z_buffer.get(y, x, lvl);
                        if (l_depth < f_depth && l_depth != z_buffer.nodata)
                            continue;

                        Eigen::Vector3f f_ver = f_ray * f_depth;
                        Eigen::Vector3f kf_ver = fTokfPose * f_ver;
                        if (kf_ver(2) <= 0.0)
                            continue;

                        Eigen::Vector3f kf_ray = kf_ver / kf_ver(2);

                        kf_pol->prepareForRay(kf_ray);
                        if (!kf_pol->rayHitsShape())
                            continue;

                        Eigen::Vector2f kf_pix = cam.rayToPix(kf_ray);

                        if (!cam.isPixVisible(kf_pix))
                            continue;

                        auto kf_i = kframe.image.get(kf_pix(1), kf_pix(0), lvl);
                        auto f_i = frame.image.get(y, x, lvl);
                        float dx = frame.dx.get(y, x, lvl);
                        float dy = frame.dy.get(y, x, lvl);

                        if (kf_i == kframe.image.nodata || f_i == frame.image.nodata || dx == frame.dx.nodata || dy == frame.dy.nodata)
                            continue;

                        Eigen::Vector2f d_f_i_d_pix(dx, dy);

                        // Eigen::MatrixXf d_pix_d_f_ver = cam.dPixdPoint(f_ver);

                        // Eigen::Vector3f d_f_i_d_f_ver = d_f_i_d_pix * d_pix_d_f_ver;
                        // Eigen::Vector3f d_f_i_d_tra = d_f_i_d_f_ver;
                        // Eigen::Vector3f d_f_i_d_rot = Eigen::Vector3f(-f_ver(2) * d_f_i_d_f_ver(1) + f_ver(1) * d_f_i_d_f_ver(2), f_ver(2) * d_f_i_d_f_ver(0) - f_ver(0) * d_f_i_d_f_ver(2), -f_ver(1) * d_f_i_d_f_ver(0) + f_ver(0) * d_f_i_d_f_ver(1));

                        float v0 = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                        float v1 = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                        float v2 = -(v0 * f_ver(0) + v1 * f_ver(1)) / f_ver(2);
                        std::array<float, 3> d_f_i_d_tra = {v0, v1, v2};
                        std::array<float, 3> d_f_i_d_rot = {-f_ver(2) * v1 + f_ver(1) * v2, f_ver(2) * v0 - f_ver(0) * v2, -f_ver(1) * v0 + f_ver(0) * v1};

                        float residual = (f_i - kf_i);

                        jtra_buffer.set(d_f_i_d_tra, y, x, lvl);
                        jrot_buffer.set(d_f_i_d_rot, y, x, lvl);
                        e_buffer.set(residual, y, x, lvl);
                        z_buffer.set(f_depth, y, x, lvl);
                    }
                }
            }
        }
    */
    void renderJPoseWindow(dataCPU<float> *frameIdepth, camera cam, frameCPU *kframe, frameCPU *frame, dataCPU<std::array<float, 6>> *jpose_buffer, dataCPU<float> *e_buffer, int lvl)
    {
        Sophus::SE3f kfTofPose = frame->pose * kframe->pose.inverse();
        Sophus::SE3f fTokfPose = kfTofPose.inverse();

        for (int y = cam.window_min_y; y < cam.window_max_y; y++)
        {
            for (int x = cam.window_min_x; x < cam.window_max_x; x++)
            {
                Eigen::Vector2f f_pix(x, y);
                Eigen::Vector3f f_ray = cam.pixToRay(f_pix);
                float f_idepth = frameIdepth->get(y, x, lvl);
                if (f_idepth <= 0.0 || f_idepth == frameIdepth->nodata)
                    continue;
                Eigen::Vector3f f_ver = f_ray / f_idepth;

                Eigen::Vector3f kf_ver = fTokfPose * f_ver;
                if (kf_ver(2) <= 0.0)
                    continue;

                Eigen::Vector3f kf_ray = kf_ver / kf_ver(2);
                Eigen::Vector2f kf_pix = cam.rayToPix(kf_ray);
                if (!cam.isPixVisible(kf_pix))
                    continue;

                // z-buffer
                // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                //    continue;

                auto kf_i = kframe->image.get(kf_pix(1), kf_pix(0), lvl);
                auto f_i = frame->image.get(y, x, lvl);
                float dx = frame->dx.get(y, x, lvl);
                float dy = frame->dy.get(y, x, lvl);
                // float dx = frame2.dx.get(f2_pix(1), f2_pix(0), lvl);
                // float dy = frame2.dy.get(f2_pix(1), f2_pix(0), lvl);
                // Eigen::Vector2f d_f_i_d_pix(dx, dy);

                if (kf_i == kframe->image.nodata || f_i == frame->image.nodata || dx == frame->dx.nodata || dy == frame->dy.nodata)
                    continue;

                Eigen::Vector2f d_f_i_d_pix(dx, dy);

                float v0 = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                float v1 = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                float v2 = -(v0 * f_ver(0) + v1 * f_ver(1)) / f_ver(2);

                std::array<float, 3> d_f_i_d_tra = {v0, v1, v2};
                std::array<float, 3> d_f_i_d_rot = {-f_ver(2) * v1 + f_ver(1) * v2, f_ver(2) * v0 - f_ver(0) * v2, -f_ver(1) * v0 + f_ver(0) * v1};

                std::array<float, 6> j_pose = {d_f_i_d_tra[0], d_f_i_d_tra[1], d_f_i_d_tra[2], d_f_i_d_rot[0], d_f_i_d_rot[1], d_f_i_d_rot[2]};

                float residual = (f_i - kf_i);

                jpose_buffer->set(j_pose, y, x, lvl);
                e_buffer->set(residual, y, x, lvl);
            }
        }
    }

    template <int DoF>
    void renderJMapWindow(SceneBase *kframeScene, SceneBase *frameScene, camera cam, frameCPU *kframe, frameCPU *frame, dataCPU<std::array<float, DoF>> *jmap_buffer, dataCPU<float> *e_buffer, dataCPU<std::array<int, DoF>> *pId_buffer, int lvl)
    {
        float min_area = 0.0 * (float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) / 16;
        // float min_angle = M_PI / 64.0;

        // std::unique_ptr<SceneBase> kframeMesh = scene.clone();
        // std::unique_ptr<SceneBase> frameMesh = scene.clone();

        // kframeMesh->transform(kframe.pose);
        // frameMesh->transform(frame.pose);

        Sophus::SE3f kfTofPose = frame->pose * kframe->pose.inverse();
        Sophus::SE3f fTokfPose = kfTofPose.inverse();

        // for each triangle
        std::vector<unsigned int> t_ids = kframeScene->getShapesIds();
        for (auto t_id : t_ids)
        {
            std::vector<unsigned int> p_ids = kframeScene->getShapeParamsIds(t_id);

            auto kf_pol = kframeScene->getShape(t_id);

            // if (kf_tri_3d.vertices[0](2) <= 0.0 || kf_tri_3d.vertices[1](2) <= 0.0 || kf_tri_3d.vertices[2](2) <= 0.0)
            //     continue;
            if (kf_pol->getArea() < min_area)
                continue;
            // std::array<float, 3> kf_tri_angles = kf_tri_2d.getAngles();
            // if (fabs(kf_tri_angles[0]) < min_angle || fabs(kf_tri_angles[1]) < min_angle || fabs(kf_tri_angles[2]) < min_angle)
            //     continue;

            auto f_pol = frameScene->getShape(t_id);

            // if (f_tri_3d.vertices[0](2) <= 0.0 || f_tri_3d.vertices[1](2) <= 0.0 || f_tri_3d.vertices[2](2) <= 0.0)
            //     continue;
            if (f_pol->getArea() < min_area)
                continue;
            // std::array<float, 3> f_tri_angles = f_tri_2d.getAngles();
            // if (fabs(f_tri_angles[0]) < min_angle || fabs(f_tri_angles[1]) < min_angle || fabs(f_tri_angles[2]) < min_angle)
            //     continue;

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

                    // z-buffer
                    float l_idepth = z_buffer.get(y, x, lvl);
                    if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                        continue;

                    Eigen::Vector3f f_ver = f_ray * f_depth;
                    Eigen::Vector3f kf_ver = fTokfPose * f_ver;
                    if (kf_ver(2) <= 0.0)
                        continue;

                    Eigen::Vector3f kf_ray = kf_ver / kf_ver(2);

                    kf_pol->prepareForRay(kf_ray);
                    if (!kf_pol->rayHitsShape())
                        continue;

                    Eigen::Vector2f kf_pix = cam.rayToPix(kf_ray);
                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = kframe->image.get(kf_pix(1), kf_pix(0), lvl);
                    auto f_i = frame->image.get(y, x, lvl);
                    float dx = frame->dx.get(y, x, lvl);
                    float dy = frame->dy.get(y, x, lvl);

                    if (kf_i == kframe->image.nodata || f_i == frame->image.nodata || dx == frame->dx.nodata || dy == frame->dy.nodata)
                        continue;

                    Eigen::Vector2f d_f_i_d_pix(dx, dy);

                    float error = f_i - kf_i;

                    Eigen::Vector3f d_f_i_d_f_ver = cam.d_f_i_d_f_ver(d_f_i_d_pix, f_ver);

                    // Eigen::Vector3f d_f_i_d_f_ver;
                    // d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    // d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    // d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                    Eigen::Vector3f d_f_ver_d_kf_depth = kfTofPose.rotationMatrix() * kf_ray;
                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    // this could be the jacobian of the depth of the 3 vertices in a triangle
                    // or the jacobian of the normal + depth of a surfel
                    std::vector<float> Jacobian = kf_pol->getJacobian(d_f_i_d_kf_depth);

                    std::array<float, DoF> jacs = jmap_buffer->nodata;
                    std::array<int, DoF> ids = pId_buffer->nodata;
                    for (size_t i = 0; i < p_ids.size(); i++)
                    {
                        if (i >= DoF)
                            break;
                        jacs[i] = Jacobian[i];
                        ids[i] = p_ids[i];
                    }

                    e_buffer->set(error, y, x, lvl);
                    jmap_buffer->set(jacs, y, x, lvl);
                    pId_buffer->set(ids, y, x, lvl);
                }
            }
        }
    }

    void renderIdepthWindow(camera cam, dataCPU<float> *buffer, int lvl)
    {
        std::vector<unsigned int> shapesIds = frameMesh->getShapesIds();

        // for each triangle
        for (auto t_id : shapesIds)
        {
            // Triangle kf_tri = keyframeMesh.triangles[t_id];
            // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
            //     continue;
            // if (kf_tri.isBackFace())
            //     continue;
            auto f_pol = frameMesh->getShape(t_id);
            // if (f_tri2d.vertices[0](2) <= 0.0 || f_tri2d.vertices[1](2) <= 0.0 || f_tri2d.vertices[2](2) <= 0.0)
            //      continue;
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

                    float z_depth = z_buffer.get(y, x, lvl);
                    if (z_depth <= f_depth && z_depth != z_buffer.nodata)
                        continue;

                    buffer->set(1.0 / f_depth, y, x, lvl);

                    z_buffer.set(f_depth, y, x, lvl);
                }
            }
        }
    }

    dataCPU<float> z_buffer;

    std::unique_ptr<SceneBase> frameMesh;

    ThreadPool pool;
};
