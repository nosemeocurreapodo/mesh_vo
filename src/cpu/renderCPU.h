#pragma once

#include <Eigen/Core>
#include <thread>

#include "common/camera.h"
#include "common/types.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/SceneBase.h"
#include "threadpoolCPU.h"
#include "params.h"

class renderCPU
{
public:
    renderCPU(unsigned int width, unsigned int height)
        : z_buffer(width, height, -1),
          pool(RENDERER_NTHREADS)
    {
    }

    dataCPU<float> getzbuffer()
    {
        return z_buffer;
    }

    void renderIdepthLineSearch(frameCPU *kframe, frameCPU *frame, camera &cam, dataCPU<float> *buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        window win(0, cam.width, 0, cam.height);

        renderIdepthLineSearchWindow(kframe, frame, cam, win, buffer, lvl);
    }

    void renderRandom(camera &cam, dataCPU<float> *buffer, int lvl)
    {
        window win(0, cam.width, 0, cam.height);

        renderRandomWindow(cam, win, buffer, lvl);
    }

    void renderSmooth(camera &cam, dataCPU<float> *buffer, int lvl, float start = 1.0, float end = 2.0)
    {
        window win(0, cam.width, 0, cam.height);

        renderSmoothWindow(cam, win, buffer, lvl, start, end);
    }

    void renderInterpolate(camera &cam, dataCPU<float> *buffer, int lvl)
    {
        window win(0, cam.width, 0, cam.height);

        renderInterpolateWindow(cam, win, buffer, lvl);
    }

    void renderImage(SceneBase *kscene, frameCPU *kframe, SceneBase *scene, camera &cam, dataCPU<float> *buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        window win(0, cam.width, 0, cam.height);

        renderImageWindow(kscene, kframe, scene, cam, win, buffer, lvl);
    }

    void renderImageParallel(SceneBase *kscene, frameCPU *kframe, SceneBase *scene, camera &cam, dataCPU<float> *buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        int divi_y = pool.getNumThreads();
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

                window win(min_x, max_x, min_y, max_y);

                // renderImageWindow(kscene, kframe, scene, cam, window, buffer, lvl);
                pool.enqueue(std::bind(&renderCPU::renderImageWindow, this, kscene, kframe, scene, cam, win, buffer, lvl));
            }
        }

        pool.waitUntilDone();
    }

    /*
    void renderImage(dataCPU<float> &poseIdepth, camera &cam, frameCPU &kframe, Sophus::SE3f pose, dataCPU<float> &buffer, int lvl)
    {
        renderImageWindow(&poseIdepth, cam, &kframe, pose, &buffer, lvl);
    }

    void renderImageParallel(dataCPU<float> &poseIdepth, camera &cam, frameCPU &kframe, Sophus::SE3f pose, dataCPU<float> &buffer, int lvl)
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

                // renderImageWindow(&poseIdepth, cam, &kframe, pose, &buffer, lvl);
                pool.enqueue(std::bind(&renderCPU::renderImageWindow, this, &poseIdepth, cam, &kframe, pose, &buffer, lvl));
            }
        }

        pool.waitUntilDone();
    }
    */

    void renderDebug(SceneBase *scene, frameCPU *frame, camera &cam, dataCPU<float> *buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        window win(0, cam.width, 0, cam.height);

        renderDebugWindow(scene, frame, cam, win, buffer, lvl);
    }

    void renderDebugParallel(SceneBase *scene, frameCPU *frame, camera &cam, dataCPU<float> *buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        int divi_y = pool.getNumThreads();
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

                window win(min_x, max_x, min_y, max_y);

                // renderDebugWindow(scene, frame, cam, win, buffer, lvl);
                pool.enqueue(std::bind(&renderCPU::renderDebugWindow, this, scene, frame, cam, win, buffer, lvl));
            }
        }

        pool.waitUntilDone();
    }

    template <typename Type1, typename Type2>
    void renderJMap(SceneBase *kscene, frameCPU *kframe, SceneBase *scene, frameCPU *frame, camera &cam, dataCPU<Type1> *jmap_buffer, dataCPU<float> *e_buffer, dataCPU<Type2> *pId_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        window win(0, cam.width, 0, cam.height);

        renderJMapWindow(kscene, kframe, scene, frame, cam, win, jmap_buffer, e_buffer, pId_buffer, lvl);
    }

    template <typename Type1, typename Type2>
    void renderJMapParallel(SceneBase *kscene, frameCPU *kframe, SceneBase *scene, frameCPU *frame, camera &cam, dataCPU<Type1> *jmap_buffer, dataCPU<float> *e_buffer, dataCPU<Type2> *pId_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        int divi_y = pool.getNumThreads();
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

                window win(min_x, max_x, min_y, max_y);

                // renderJMapWindow(kscene, kframe, scene, frame, cam, win, jmap_buffer, e_buffer, pId_buffer, lvl);
                pool.enqueue(std::bind(&renderCPU::renderJMapWindow<Type1, Type2>, this, kscene, kframe, scene, frame, cam, win, jmap_buffer, e_buffer, pId_buffer, lvl));
            }
        }

        pool.waitUntilDone();
    }

    template <typename Type1, typename Type2, typename Type3>
    void renderJPoseMap(SceneBase *kscene, frameCPU *kframe, SceneBase *scene, frameCPU *frame, camera cam, dataCPU<Type1> *jpose_buffer, dataCPU<Type2> *jmap_buffer, dataCPU<float> *e_buffer, dataCPU<Type3> *pId_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        window win(0, cam.width, 0, cam.height);

        renderJPoseMapWindow(kscene, kframe, scene, frame, cam, win, jpose_buffer, jmap_buffer, e_buffer, pId_buffer, lvl);
    }

    template <typename Type1, typename Type2, typename Type3>
    void renderJPoseMapParallel(SceneBase *kscene, frameCPU *kframe, SceneBase *scene, frameCPU *frame, camera cam, dataCPU<Type1> *jpose_buffer, dataCPU<Type2> *jmap_buffer, dataCPU<float> *e_buffer, dataCPU<Type3> *pId_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        int divi_y = pool.getNumThreads();
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

                window win(min_x, max_x, min_y, max_y);

                // renderJPoseMapWindow(kscene, kframe, scene, frame, cam, win, jpose_buffer, jmap_buffer, e_buffer, pId_buffer, lvl);
                pool.enqueue(std::bind(&renderCPU::renderJPoseMapWindow<Type1, Type2, Type3>, this, kscene, kframe, scene, frame, cam, win, jpose_buffer, jmap_buffer, e_buffer, pId_buffer, lvl));
            }
        }

        pool.waitUntilDone();
    }

    template <typename Type>
    void renderJPose(SceneBase *kscene, frameCPU *kframe, SceneBase *scene, frameCPU *frame, camera cam, dataCPU<Type> *jpose_buffer, dataCPU<float> *e_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        window win(0, cam.width, 0, cam.height);

        renderJPoseWindow(kscene, kframe, scene, frame, cam, win, jpose_buffer, e_buffer, lvl);
    }

    template <typename Type>
    void renderJPoseParallel(SceneBase *kscene, frameCPU *kframe, SceneBase *scene, frameCPU *frame, camera cam, dataCPU<Type> *jpose_buffer, dataCPU<float> *e_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        int divi_y = pool.getNumThreads();
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

                window win(min_x, max_x, min_y, max_y);

                // renderJPoseWindow(kscene, kframe, scene, frame, cam, win, jpose_buffer, e_buffer, lvl);
                pool.enqueue(std::bind(&renderCPU::renderJPoseWindow<Type>, this, kscene, kframe, scene, frame, cam, win, jpose_buffer, e_buffer, lvl));
            }
        }

        pool.waitUntilDone();
    }

    /*
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
    */

    void renderIdepth(SceneBase *scene, camera cam, dataCPU<float> *buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        window win(0, cam.width, 0, cam.height);

        renderIdepthWindow(scene, cam, win, buffer, lvl);
    }

    void renderIdepthParallel(SceneBase *scene, camera cam, dataCPU<float> *buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        int divi_y = pool.getNumThreads();
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

                window win(min_x, max_x, min_y, max_y);

                // renderIdepthWindow(scene, cam, win, buffer, lvl);
                pool.enqueue(std::bind(&renderCPU::renderIdepthWindow, this, scene, cam, win, buffer, lvl));
            }
        }

        pool.waitUntilDone();
    }

    void renderWeight(SceneBase *scene, camera cam, dataCPU<float> *buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        window win(0, cam.width, 0, cam.height);

        renderWeightWindow(scene, cam, win, buffer, lvl);
    }

    void renderWeightParallel(SceneBase *scene, camera cam, dataCPU<float> *buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        int divi_y = pool.getNumThreads();
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

                window win(min_x, max_x, min_y, max_y);

                // renderIdepthWindow(scene, cam, win, buffer, lvl);
                pool.enqueue(std::bind(&renderCPU::renderWeightWindow, this, scene, cam, win, buffer, lvl));
            }
        }

        pool.waitUntilDone();
    }

private:
    void renderSmoothWindow(camera cam, window win, dataCPU<float> *buffer, int lvl, float start = 1.0, float end = 2.0)
    {
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                float val = start + (end - start) * float(y) / (cam.width - 1.0);
                buffer->set(val, y, x, lvl);
            }
        }
    }

    void renderRandomWindow(camera cam, window win, dataCPU<float> *buffer, int lvl, float min = 1.0, float max = 2.0)
    {
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                if (buffer->get(y, x, lvl) == buffer->nodata)
                {
                    float val = (max - min) * float(rand() % 1000) / 1000.0 + min;
                    buffer->set(val, y, x, lvl);
                }
            }
        }
    }

    void renderInterpolateWindow(camera cam, window win, dataCPU<float> *buffer, int lvl)
    {
        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                if (buffer->get(y, x, lvl) == buffer->nodata)
                {
                    int size = 10;
                    float acc = 0.0;
                    int count = 0;
                    while (true)
                    {
                        for (int y_ = y - size; y_ <= y + size; y_ += size)
                        {
                            for (int x_ = x - size; x_ <= x + size; x_ += size)
                            {
                                if (!cam.isPixVisible(x_, y_))
                                    continue;

                                auto val = buffer->get(y_, x_, lvl);
                                if (val == buffer->nodata)
                                    continue;

                                acc += val;
                                count += 1;
                            }
                        }
                        if (count == 0)
                        {
                            size *= 2;
                        }
                        else
                        {
                            acc /= count;
                            break;
                        }
                    }
                    buffer->set(acc, y, x, lvl);
                }
            }
        }
    }

    void renderIdepthLineSearchWindow(frameCPU *kframe, frameCPU *frame, camera cam, window win, dataCPU<float> *buffer, int lvl)
    {
        Sophus::SE3f kfTofPose = frame->pose * kframe->pose.inverse();
        Sophus::SE3f fTokfPose = kfTofPose.inverse();

        int size = 5;

        for (int y = win.min_y; y <= win.max_y; y++)
        {
            for (int x = win.min_x; x <= win.max_x; x++)
            {
                auto f_i = frame->image.get(y, x, lvl);
                if (f_i == frame->image.nodata)
                    continue;

                vec2<float> f_pix(x, y);
                // if (!cam.isPixVisible(f_pix))
                //     continue;

                vec3<float> f_ray = cam.pixToRay(x, y);

                float depth_min = 0.1;
                float depth_max = 10.0;

                vec3<float> f_ver_min = f_ray * depth_min;
                Eigen::Vector3f kf_ver_min_e = fTokfPose * Eigen::Vector3f(f_ver_min(0), f_ver_min(1), f_ver_min(2));
                vec3<float> kf_ver_min(kf_ver_min_e(0), kf_ver_min_e(1), kf_ver_min_e(2));
                vec3<float> kf_ray_min = kf_ver_min / kf_ver_min(2);
                vec2<float> kf_pix_min = cam.rayToPix(kf_ray_min);

                vec3<float> f_ver_max = f_ray * depth_max;
                Eigen::Vector3f kf_ver_max_e = fTokfPose * Eigen::Vector3f(f_ver_max(0), f_ver_max(1), f_ver_max(2));
                vec3<float> kf_ver_max(kf_ver_max_e(0), kf_ver_max_e(1), kf_ver_max_e(2));
                vec3<float> kf_ray_max = kf_ver_max / kf_ver_max(2);
                vec2<float> kf_pix_max = cam.rayToPix(kf_ray_max);

                vec2<float> kf_pix_diff = kf_pix_max - kf_pix_min;
                float kf_pix_diff_norm = kf_pix_diff.norm();
                kf_pix_diff = kf_pix_diff / kf_pix_diff_norm;

                for (int i = 0; i < kf_pix_diff_norm; i++)
                {
                    vec2<float> kf_pix = kf_pix_min + kf_pix_diff * i;

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    // float z_depth = z_buffer.get(y, x, lvl);
                    // if (z_depth < f_depth && z_depth != z_buffer.nodata)
                    //     continue;

                    auto kf_i = kframe->image.get(kf_pix(1), kf_pix(0), lvl);
                    if (kf_i == kframe->image.nodata)
                        continue;

                    float error = (f_i - kf_i) * (f_i - kf_i);

                    float last_error = buffer->get(y, x, lvl);

                    if (error < last_error || last_error == buffer->nodata)
                    {
                        buffer->set(error, y, x, lvl);

                        vec3<float> kf_ray = cam.pixToRay(kf_pix);
                        Eigen::Vector3f Y_e = fTokfPose * Eigen::Vector3f(f_ray(0), f_ray(1), f_ray(2));
                        vec3<float> Y(Y_e(0), Y_e(1), Y_e(2));
                        Eigen::Vector3f C_e = fTokfPose.translation();
                        vec3<float> C(C_e(0), C_e(1), C_e(2));

                        float f_depth = (C(1) / kf_ray(1) - C(2)) / (Y(2) - Y(1) / kf_ray(1));
                        // float f_depth = depth_min + i * (depth_max - depth_min) / kf_pix_diff_norm;
                        // float f_idepth = 1.0 / f_depth;

                        z_buffer.set(f_depth, y, x, lvl);
                    }
                }
            }
        }
    }

    void renderImageWindow(SceneBase *kscene, frameCPU *kframe, SceneBase *scene, camera cam, window win, dataCPU<float> *buffer, int lvl)
    {
        Sophus::SE3f kfTofPose = scene->getPose() * kframe->pose.inverse();
        Sophus::SE3f fTokfPose = kfTofPose.inverse();

        std::vector<int> ids = scene->getShapesIds();

        auto kf_pol = kscene->getShape(cam, ids[0]);
        auto f_pol = scene->getShape(cam, ids[0]);

        // for each triangle
        for (auto t_id : ids)
        {
            // if (t_id % 2 != 0)
            //     continue;

            // Polygon kf_pol = mesh.getPolygon(t_id);
            //  if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
            //      continue;
            // if (kf_tri.getArea() < 1.0)
            //     continue;

            if (!scene->isShapeInWindow(win, t_id))
                continue;

            // auto kf_pol = kscene->getShape(t_id);
            // auto f_pol = scene->getShape(t_id);
            kscene->getShape(kf_pol.get(), cam, t_id);
            scene->getShape(f_pol.get(), cam, t_id);

            // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
            //     continue;
            if (f_pol->getArea() < 0.0)
                continue;

            // window pol_win = f_pol->getScreenBounds(cam);
            window pol_win = f_pol->getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y < pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x < pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);
                    // if (!cam.isPixVisible(f_pix))
                    //     continue;

                    vec3<float> f_ray = cam.pixToRay(x, y);

                    // f_pol->prepareForRay(f_ray);
                    f_pol->prepareForPix(f_pix);
                    if (!f_pol->hitsShape())
                        continue;

                    float f_depth = f_pol->getDepth();
                    if (f_depth <= 0.0)
                        continue;

                    float z_depth = z_buffer.get(y, x, lvl);
                    if (z_depth < f_depth && z_depth != z_buffer.nodata)
                        continue;

                    // vec2<float> kf_pix = f_pol->getPix(kf_pol.get());

                    // vec3<float> kf_ray = f_pol->getRay(kf_pol.get());

                    vec3<float> f_ver = f_ray * f_depth;

                    Eigen::Vector3f kf_ver_e = fTokfPose * Eigen::Vector3f(f_ver(0), f_ver(1), f_ver(2));
                    vec3<float> kf_ver(kf_ver_e(0), kf_ver_e(1), kf_ver_e(2));
                    if (kf_ver(2) <= 0.0)
                        continue;

                    vec3<float> kf_ray = kf_ver / kf_ver(2);
                    vec2<float> kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = kframe->image.get(kf_pix(1), kf_pix(0), lvl);
                    if (kf_i == kframe->image.nodata)
                        continue;

                    buffer->set(kf_i, y, x, lvl);
                    z_buffer.set(f_depth, y, x, lvl);
                }
            }
        }
    }

    void renderIdepthWindow(SceneBase *scene, camera cam, window win, dataCPU<float> *buffer, int lvl)
    {
        std::vector<int> shapesIds = scene->getShapesIds();

        auto f_pol = scene->getShape(cam, shapesIds[0]);

        // for each triangle
        for (auto t_id : shapesIds)
        {
            // Triangle kf_tri = keyframeMesh.triangles[t_id];
            // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
            //     continue;
            // if (kf_tri.isBackFace())
            //     continue;

            if (!scene->isShapeInWindow(win, t_id))
                continue;

            // auto f_pol = scene->getShape(t_id);
            scene->getShape(f_pol.get(), cam, t_id);

            // if (f_tri2d.vertices[0](2) <= 0.0 || f_tri2d.vertices[1](2) <= 0.0 || f_tri2d.vertices[2](2) <= 0.0)
            //      continue;
            if (f_pol->getArea() < 0.0)
                continue;

            window pol_win = f_pol->getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y < pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x < pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);
                    // if (!cam.isPixVisible(f_pix))
                    //     continue;
                    // vec3<float> f_ray = cam.pixToRay(f_pix);

                    f_pol->prepareForPix(f_pix);
                    if (!f_pol->hitsShape())
                        continue;

                    float f_depth = f_pol->getDepth();

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

    void renderWeightWindow(SceneBase *scene, camera cam, window win, dataCPU<float> *buffer, int lvl)
    {
        std::vector<int> shapesIds = scene->getShapesIds();

        auto f_pol = scene->getShape(cam, shapesIds[0]);

        // for each triangle
        for (auto t_id : shapesIds)
        {
            // Triangle kf_tri = keyframeMesh.triangles[t_id];
            // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
            //     continue;
            // if (kf_tri.isBackFace())
            //     continue;

            if (!scene->isShapeInWindow(win, t_id))
                continue;

            // auto f_pol = scene->getShape(t_id);
            scene->getShape(f_pol.get(), cam, t_id);

            // if (f_tri2d.vertices[0](2) <= 0.0 || f_tri2d.vertices[1](2) <= 0.0 || f_tri2d.vertices[2](2) <= 0.0)
            //      continue;
            if (f_pol->getArea() < 0.0)
                continue;

            window pol_win = f_pol->getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y < pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x < pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);
                    // if (!cam.isPixVisible(f_pix))
                    //     continue;
                    // vec3<float> f_ray = cam.pixToRay(f_pix);

                    f_pol->prepareForPix(f_pix);
                    if (!f_pol->hitsShape())
                        continue;

                    float f_depth = f_pol->getDepth();

                    if (f_depth <= 0.0)
                        continue;

                    float z_depth = z_buffer.get(y, x, lvl);
                    if (z_depth <= f_depth && z_depth != z_buffer.nodata)
                        continue;

                    float f_weight = f_pol->getWeight();

                    buffer->set(f_weight, y, x, lvl);
                    z_buffer.set(f_depth, y, x, lvl);
                }
            }
        }
    }

    template <typename Type>
    void renderJPoseWindow(SceneBase *kscene, frameCPU *kframe, SceneBase *scene, frameCPU *frame, camera cam, window win, dataCPU<Type> *jpose_buffer, dataCPU<float> *e_buffer, int lvl)
    {
        float min_area = 0.0 * (float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) / 16;
        // float min_angle = M_PI / 64.0;

        Sophus::SE3f kfTofPose = frame->pose * kframe->pose.inverse();
        Sophus::SE3f fTokfPose = kfTofPose.inverse();

        // for each triangle
        std::vector<int> t_ids = scene->getShapesIds();

        auto kf_pol = kscene->getShape(cam, t_ids[0]);
        auto f_pol = scene->getShape(cam, t_ids[0]);

        for (auto t_id : t_ids)
        {
            // if (t_id % 2 != 0)
            //     continue;

            if (!scene->isShapeInWindow(win, t_id))
                continue;

            // auto kf_pol = kscene->getShape(t_id);
            // auto f_pol = scene->getShape(t_id);

            kscene->getShape(kf_pol.get(), cam, t_id);
            scene->getShape(f_pol.get(), cam, t_id);

            if (f_pol->getArea() < min_area)
                continue;

            // window pol_win = f_pol->getScreenBounds(cam);
            window pol_win = f_pol->getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y <= pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x <= pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);
                    if (!cam.isPixVisible(f_pix))
                        continue;
                    vec3<float> f_ray = cam.pixToRay(x, y);

                    // f_pol->prepareForRay(f_ray);
                    f_pol->prepareForPix(f_pix);
                    if (!f_pol->hitsShape())
                        continue;

                    float f_depth = f_pol->getDepth();
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_depth = z_buffer.get(y, x, lvl);
                    if (l_depth < f_depth && l_depth != z_buffer.nodata)
                        continue;

                    vec3<float> f_ver = f_ray * f_depth;

                    // vec3<float> kf_ray = f_pol->getRay(kf_pol.get());
                    // vec2<float> kf_pix = f_pol->getPix(kf_pol.get());

                    Eigen::Vector3f kf_ver_e = fTokfPose * Eigen::Vector3f(f_ver(0), f_ver(1), f_ver(2));
                    vec3<float> kf_ver(kf_ver_e(0), kf_ver_e(1), kf_ver_e(2));

                    if (kf_ver(2) <= 0.0)
                        continue;

                    vec3<float> kf_ray = kf_ver / kf_ver(2);

                    vec2<float> kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = kframe->image.get(kf_pix(1), kf_pix(0), lvl);
                    auto f_i = frame->image.get(y, x, lvl);
                    float dx = frame->dx.get(y, x, lvl);
                    float dy = frame->dy.get(y, x, lvl);

                    if (kf_i == kframe->image.nodata || f_i == frame->image.nodata || dx == frame->dx.nodata || dy == frame->dy.nodata)
                        continue;

                    vec2<float> d_f_i_d_pix(dx, dy);

                    // Eigen::MatrixXf d_pix_d_f_ver = cam.dPixdPoint(f_ver);

                    // Eigen::Vector3f d_f_i_d_f_ver = d_f_i_d_pix * d_pix_d_f_ver;
                    // Eigen::Vector3f d_f_i_d_tra = d_f_i_d_f_ver;
                    // Eigen::Vector3f d_f_i_d_rot = Eigen::Vector3f(-f_ver(2) * d_f_i_d_f_ver(1) + f_ver(1) * d_f_i_d_f_ver(2), f_ver(2) * d_f_i_d_f_ver(0) - f_ver(0) * d_f_i_d_f_ver(2), -f_ver(1) * d_f_i_d_f_ver(0) + f_ver(0) * d_f_i_d_f_ver(1));

                    float v0 = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    float v1 = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    float v2 = -(v0 * f_ver(0) + v1 * f_ver(1)) / f_ver(2);
                    vec3<float> d_f_i_d_tra(v0, v1, v2);
                    vec3<float> d_f_i_d_rot(-f_ver(2) * v1 + f_ver(1) * v2, f_ver(2) * v0 - f_ver(0) * v2, -f_ver(1) * v0 + f_ver(0) * v1);

                    Type j_pose = {d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2)};

                    float residual = (f_i - kf_i);

                    jpose_buffer->set(j_pose, y, x, lvl);
                    e_buffer->set(residual, y, x, lvl);
                    z_buffer.set(f_depth, y, x, lvl);
                }
            }
        }
    }
    /*
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
    */

    template <typename Type1, typename Type2>
    void renderJMapWindow(SceneBase *kscene, frameCPU *kframe, SceneBase *scene, frameCPU *frame, camera cam, window win, dataCPU<Type1> *jmap_buffer, dataCPU<float> *e_buffer, dataCPU<Type2> *pId_buffer, int lvl)
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
        std::vector<int> t_ids = scene->getShapesIds();

        int shapeDoF = scene->getShapesDoF();

        for (auto t_id : t_ids)
        {
            if (!scene->isShapeInWindow(win, t_id))
                continue;

            std::vector<int> p_ids = scene->getShapeParamsIds(t_id);

            auto kf_pol = kscene->getShape(cam, t_id);

            // if (kf_tri_3d.vertices[0](2) <= 0.0 || kf_tri_3d.vertices[1](2) <= 0.0 || kf_tri_3d.vertices[2](2) <= 0.0)
            //     continue;
            if (kf_pol->getArea() < min_area)
                continue;
            // std::array<float, 3> kf_tri_angles = kf_tri_2d.getAngles();
            // if (fabs(kf_tri_angles[0]) < min_angle || fabs(kf_tri_angles[1]) < min_angle || fabs(kf_tri_angles[2]) < min_angle)
            //     continue;

            auto f_pol = scene->getShape(cam, t_id);

            // if (f_tri_3d.vertices[0](2) <= 0.0 || f_tri_3d.vertices[1](2) <= 0.0 || f_tri_3d.vertices[2](2) <= 0.0)
            //     continue;
            if (f_pol->getArea() < min_area)
                continue;
            // std::array<float, 3> f_tri_angles = f_tri_2d.getAngles();
            // if (fabs(f_tri_angles[0]) < min_angle || fabs(f_tri_angles[1]) < min_angle || fabs(f_tri_angles[2]) < min_angle)
            //     continue;

            // window pol_win = f_pol->getScreenBounds(cam);
            window pol_win = f_pol->getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y <= pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x <= pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);
                    if (!cam.isPixVisible(f_pix))
                        continue;

                    vec3<float> f_ray = cam.pixToRay(f_pix);

                    // f_pol->prepareForRay(f_ray);
                    f_pol->prepareForPix(f_pix);

                    if (!f_pol->hitsShape())
                        continue;

                    float f_depth = f_pol->getDepth();
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_idepth = z_buffer.get(y, x, lvl);
                    if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                        continue;

                    vec3<float> f_ver = f_ray * f_depth;

                    // vec3<float> kf_ray = f_pol->getRay(kf_pol.get());

                    Eigen::Vector3f kf_ver_e = fTokfPose * Eigen::Vector3f(f_ver(0), f_ver(1), f_ver(2));
                    vec3<float> kf_ver(kf_ver_e(0), kf_ver_e(1), kf_ver_e(2));

                    if (kf_ver(2) <= 0.0)
                        continue;

                    vec3<float> kf_ray = kf_ver / kf_ver(2);

                    // kf_pol->prepareForRay(kf_ray);
                    // if (!kf_pol->hitsShape())
                    //     continue;

                    vec2<float> kf_pix = cam.rayToPix(kf_ray);

                    // vec3<float> kf_ray = f_pol->getRay(kf_pol.get());
                    // vec2<float> kf_pix = f_pol->getPix(kf_pol.get());

                    kf_pol->prepareForPix(kf_pix);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = kframe->image.get(kf_pix(1), kf_pix(0), lvl);
                    auto f_i = frame->image.get(y, x, lvl);
                    float dx = frame->dx.get(y, x, lvl);
                    float dy = frame->dy.get(y, x, lvl);

                    if (kf_i == kframe->image.nodata || f_i == frame->image.nodata || dx == frame->dx.nodata || dy == frame->dy.nodata)
                        continue;

                    vec2<float> d_f_i_d_pix(dx, dy);

                    float error = f_i - kf_i;

                    // vec3<float> d_f_i_d_f_ver = cam.d_f_i_d_f_ver(d_f_i_d_pix, f_ver);

                    vec3<float> d_f_i_d_f_ver;
                    d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                    Eigen::Vector3f d_f_ver_d_kf_depth_e = kfTofPose.rotationMatrix() * Eigen::Vector3f(kf_ray(0), kf_ray(1), kf_ray(2));

                    vec3<float> d_f_ver_d_kf_depth(d_f_ver_d_kf_depth_e(0), d_f_ver_d_kf_depth_e(1), d_f_ver_d_kf_depth_e(2));

                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    // this could be the jacobian of the depth of the 3 vertices in a triangle
                    // or the jacobian of the normal + depth of a surfel
                    std::vector<float> Jacobian = kf_pol->getJacobian(d_f_i_d_kf_depth);
                    // std::vector<float> Jacobian2 = f_pol->getJacobian(d_f_i_d_kf_depth);

                    Type1 jacs = jmap_buffer->nodata;
                    Type2 ids = pId_buffer->nodata;
                    for (size_t i = 0; i < p_ids.size(); i++)
                    {
                        if (i >= shapeDoF)
                            break;
                        jacs(i) = Jacobian[i];
                        ids(i) = p_ids[i];
                    }

                    e_buffer->set(error, y, x, lvl);
                    jmap_buffer->set(jacs, y, x, lvl);
                    pId_buffer->set(ids, y, x, lvl);
                }
            }
        }
    }

    template <typename Type1, typename Type2, typename Type3>
    void renderJPoseMapWindow(SceneBase *kscene, frameCPU *kframe, SceneBase *scene, frameCPU *frame, camera cam, window win, dataCPU<Type1> *jpose_buffer, dataCPU<Type2> *jmap_buffer, dataCPU<float> *e_buffer, dataCPU<Type3> *pId_buffer, int lvl)
    {
        float min_area = 0.0 * (float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) / 16.0;
        // float min_angle = M_PI / 64.0;

        Sophus::SE3f kfTofPose = frame->pose * kframe->pose.inverse();
        Sophus::SE3f fTokfPose = kfTofPose.inverse();

        // for each triangle
        std::vector<int> t_ids = kscene->getShapesIds();
        int shapeDoF = kscene->getShapesDoF();

        for (auto t_id : t_ids)
        {
            if (!scene->isShapeInWindow(win, t_id))
                continue;

            std::vector<int> p_ids = kscene->getShapeParamsIds(t_id);

            auto kf_pol = kscene->getShape(cam, t_id);
            // if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
            //     continue;
            if (kf_pol->getArea() < min_area)
                continue;
            // std::array<float, 3> kf_tri_angles = kf_tri_2d.getAngles();
            // if (fabs(kf_tri_angles[0]) < min_angle || fabs(kf_tri_angles[1]) < min_angle || fabs(kf_tri_angles[2]) < min_angle)
            //    continue;

            auto f_pol = scene->getShape(cam, t_id);
            // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
            //     continue;
            if (f_pol->getArea() < min_area)
                continue;
            // std::array<float, 3> f_tri_angles = f_tri_2d.getAngles();
            // if (fabs(f_tri_angles[0]) < min_angle || fabs(f_tri_angles[1]) < min_angle || fabs(f_tri_angles[2]) < min_angle)
            //    continue;

            window pol_win = f_pol->getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y <= pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x <= pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);
                    if (!cam.isPixVisible(f_pix))
                        continue;

                    f_pol->prepareForPix(f_pix);
                    if (!f_pol->hitsShape())
                        continue;

                    float f_depth = f_pol->getDepth();
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                    if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                        continue;

                    vec3<float> f_ray = cam.pixToRay(f_pix);
                    vec3<float> f_ver = f_ray * f_depth;

                    // vec3<float> kf_ray = f_pol->getRay(kf_pol.get());
                    // vec2<float> kf_pix = f_pol->getPix(kf_pol.get());

                    Eigen::Vector3f kf_ver_e = fTokfPose * Eigen::Vector3f(f_ver(0), f_ver(1), f_ver(2));
                    vec3<float> kf_ver(kf_ver_e(0), kf_ver_e(1), kf_ver_e(2));
                    if (kf_ver(2) <= 0.0)
                        continue;

                    vec3<float> kf_ray = kf_ver / kf_ver(2);
                    vec2<float> kf_pix = cam.rayToPix(kf_ray);

                    kf_pol->prepareForPix(kf_pix);
                    // if (!kf_pol->hitsShape())
                    //     continue;

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = kframe->image.get(kf_pix(1), kf_pix(0), lvl);
                    auto f_i = frame->image.get(y, x, lvl);
                    float dx = frame->dx.get(y, x, lvl);
                    float dy = frame->dy.get(y, x, lvl);

                    if (kf_i == kframe->image.nodata || f_i == frame->image.nodata || dx == frame->dx.nodata || dy == frame->dy.nodata)
                        continue;

                    vec2<float> d_f_i_d_pix(dx, dy);

                    vec3<float> d_f_i_d_f_ver;
                    d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                    vec3<float> d_f_i_d_tra(d_f_i_d_f_ver(0), d_f_i_d_f_ver(1), d_f_i_d_f_ver(2));
                    vec3<float> d_f_i_d_rot(-f_ver(2) * d_f_i_d_f_ver(1) + f_ver(1) * d_f_i_d_f_ver(2), f_ver(2) * d_f_i_d_f_ver(0) - f_ver(0) * d_f_i_d_f_ver(2), -f_ver(1) * d_f_i_d_f_ver(0) + f_ver(0) * d_f_i_d_f_ver(1));

                    Type1 jpose = {d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2)};

                    jpose_buffer->set(jpose, y, x, lvl);

                    Eigen::Vector3f d_f_ver_d_kf_depth_e = kfTofPose.rotationMatrix() * Eigen::Vector3f(kf_ray(0), kf_ray(1), kf_ray(2));

                    vec3<float> d_f_ver_d_kf_depth(d_f_ver_d_kf_depth_e(0), d_f_ver_d_kf_depth_e(1), d_f_ver_d_kf_depth_e(2));
                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    std::vector<float> Jacobian = kf_pol->getJacobian(d_f_i_d_kf_depth);

                    float error = f_i - kf_i;

                    Type2 jacs = jmap_buffer->nodata;
                    Type3 ids = pId_buffer->nodata;
                    for (size_t i = 0; i < p_ids.size(); i++)
                    {
                        if (i >= shapeDoF)
                            break;
                        jacs(i) = Jacobian[i];
                        ids(i) = p_ids[i];
                    }

                    e_buffer->set(error, y, x, lvl);
                    jmap_buffer->set(jacs, y, x, lvl);
                    pId_buffer->set(ids, y, x, lvl);
                }
            }
        }
    }

    void renderDebugWindow(SceneBase *scene, frameCPU *frame, camera cam, window win, dataCPU<float> *buffer, int lvl)
    {
        std::vector<int> ids = scene->getShapesIds();

        auto f_pol = scene->getShape(cam, ids[0]);

        // for each triangle
        for (auto t_id : ids)
        {
            // Triangle kf_tri = keyframeMesh.triangles[index];
            //  if (kf_tri.vertices[0]->position(2) <= 0.0 || kf_tri.vertices[1]->position(2) <= 0.0 || kf_tri.vertices[2]->position(2) <= 0.0)
            //      continue;
            // if (kf_tri.isBackFace())
            //     continue;
            // auto f_pol = scene->getShape(t_id);

            if (!scene->isShapeInWindow(win, t_id))
                continue;

            scene->getShape(f_pol.get(), cam, t_id);
            // if (f_tri.vertices[0]->position(2) <= 0.0 || f_tri.vertices[1]->position(2) <= 0.0 || f_tri.vertices[2]->position(2) <= 0.0)
            //     continue;
            if (f_pol->getArea() < 0.0)
                continue;

            window pol_win = f_pol->getScreenBounds(cam);

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y <= pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x <= pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);
                    if (!cam.isPixVisible(f_pix))
                        continue;

                    vec3<float> f_ray = cam.pixToRay(f_pix);

                    f_pol->prepareForRay(f_ray);
                    if (!f_pol->hitsShape())
                        continue;

                    float f_depth = f_pol->getDepth();
                    bool isLine = f_pol->isEdge();

                    float f_i = frame->image.get(y, x, lvl);
                    f_i /= 255.0;

                    // z buffer
                    // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                    // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                    //    continue;

                    if (isLine)
                        buffer->set(1.0, y, x, lvl);
                    else
                        buffer->set(1.0 / f_depth, y, x, lvl);
                }
            }
        }
    }

    dataCPU<float> z_buffer;
    ThreadPool pool;
};
