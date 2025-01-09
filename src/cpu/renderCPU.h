#pragma once

#include <Eigen/Core>
#include <thread>

#include "common/camera.h"
#include "common/types.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "threadpoolCPU.h"
#include "params.h"

template <typename sceneType>
class renderCPU
{
public:
    renderCPU(unsigned int width, unsigned int height)
        : z_buffer(width, height, -1)
    {
    }

    dataCPU<float> &getzbuffer()
    {
        return z_buffer;
    }

    void renderIdepthLineSearch(dataCPU<float> &kimage, dataCPU<float> &image, Sophus::SE3f imagePose, camera cam, dataCPU<float> &buffer)
    {
        z_buffer.set(z_buffer.nodata);

        window win(0, cam.width - 1, 0, cam.height - 1);

        renderIdepthLineSearchWindow(kimage, image, imagePose, cam, win, buffer);
    }

    void renderRandom(camera cam, dataCPU<float> &buffer, float min = 0.1, float max = 1.9)
    {
        assert(cam.width == buffer.width && cam.height == buffer.height);

        window win(0, cam.width - 1, 0, cam.height - 1);

        renderRandomWindow(win, buffer, min, max);
    }

    void renderSmooth(camera cam, dataCPU<float> &buffer, float start = 1.0, float end = 2.0)
    {
        assert(cam.width == buffer.width && cam.height == buffer.height);

        window win(0, cam.width - 1, 0, cam.height - 1);

        renderSmoothWindow(cam, win, buffer, start, end);
    }

    void renderInterpolate(camera cam, dataCPU<float> &buffer)
    {
        window win(0, cam.width - 1, 0, cam.height - 1);

        renderInterpolateWindow(cam, win, buffer);
    }

    void renderImageParallel(sceneType &scene, dataCPU<float> &kimage, vec2<float> kimageAffine, Sophus::SE3f kimagePose, Sophus::SE3f imagePose, camera cam, dataCPU<float> &buffer)
    {
        scene1 = scene;
        scene2 = scene;
        scene1.transform(cam, kimagePose);
        scene2.transform(cam, imagePose);

        z_buffer.set(z_buffer.nodata);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam.width / divi_x;
        int height = cam.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderImageWindow(kimage, kimageAffine, cam, win, buffer);
                // pool.enqueue(std::bind(&renderCPU::renderImageWindow, this, kimage, cam, win, buffer, lvl));
            }
        }

        // pool.waitUntilDone();
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

    void renderDebugParallel(sceneType &scene, dataCPU<float> &image, vec2<float> imageAffine, Sophus::SE3f pose, camera cam, dataCPU<float> &buffer)
    {
        scene1 = scene;
        scene2 = scene;
        scene1.transform(cam, Sophus::SE3f());
        scene2.transform(cam, pose);

        z_buffer.set(z_buffer.nodata);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam.width / divi_x;
        int height = cam.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderDebugWindow(image, imageAffine, win, buffer);
                // pool.enqueue(std::bind(&renderCPU::renderDebugWindow, this, image, win, buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    template <typename jmapType, typename idsType>
    void renderJMapParallel(sceneType &scene, dataCPU<float> &kimage, vec2<float> kimageAffine, Sophus::SE3f kimagePose, dataCPU<float> &image, vec2<float> imageAffine, dataCPU<vec2<float>> &d_image_d_pix, Sophus::SE3f imagePose, camera cam, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &e_buffer, dataCPU<idsType> &pId_buffer)
    {
        scene1 = scene;
        scene2 = scene;
        scene1.transform(cam, kimagePose);
        scene2.transform(cam, imagePose);

        Sophus::SE3f relativePose = imagePose * kimagePose.inverse();

        z_buffer.set(z_buffer.nodata);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam.width / divi_x;
        int height = cam.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderJMapWindow(kimage, kimageAffine, image, imageAffine, d_image_d_pix, relativePose, cam, win, jmap_buffer, e_buffer, pId_buffer);
                // pool.enqueue(std::bind(&renderCPU::renderJMapWindow, this, kimage, frame, cam, win, jmap_buffer, e_buffer, pId_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    template <typename jmapType, typename idsType>
    void renderJPoseMapParallel(sceneType &kscene, dataCPU<float> &kimage, vec2<float> kimageAffine, Sophus::SE3f kimagePose, dataCPU<float> &image, vec2<float> imageAffine, dataCPU<vec2<float>> &d_image_d_pix, Sophus::SE3f imagePose, camera cam, dataCPU<vec8<float>> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &e_buffer, dataCPU<idsType> &pId_buffer)
    {
        z_buffer.set(z_buffer.nodata);

        scene1 = kscene;
        scene2 = kscene;
        scene1.transform(cam, kimagePose);
        scene2.transform(cam, imagePose);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam.width / divi_x;
        int height = cam.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderJPoseMapWindow(kimage, kimageAffine, image, imageAffine, d_image_d_pix, imagePose, cam, win, jpose_buffer, jmap_buffer, e_buffer, pId_buffer);
                // pool.enqueue(std::bind(&renderCPU::renderJPoseMapWindow, this, kimage, frame, cam, win, jpose_buffer, jmap_buffer, e_buffer, pId_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJLightAffineParallel(sceneType &scene, dataCPU<float> &kimage, vec2<float> kimageAffine, Sophus::SE3f kimagePose, dataCPU<float> &image, vec2<float> imageAffine, Sophus::SE3f imagePose, camera cam, dataCPU<vec2<float>> &jlightaffine_buffer, dataCPU<float> &e_buffer)
    {
        z_buffer.set(z_buffer.nodata);

        scene1 = scene;
        scene2 = scene;
        scene1.transform(cam, kimagePose);
        scene2.transform(cam, imagePose);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam.width / divi_x;
        int height = cam.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderJLightAffineWindow(kimage, kimageAffine, image, imageAffine, cam, win, jlightaffine_buffer, e_buffer);
                // pool.enqueue(std::bind(&renderCPU::renderJLightAffineWindow, this, kimage, frame, cam, win, jlightaffine_buffer, e_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJPoseParallel(sceneType &scene, dataCPU<float> &kimage, vec2<float> kaffine, Sophus::SE3f kpose, dataCPU<float> &image, vec2<float> affine, dataCPU<vec2<float>> &d_image_d_pix, Sophus::SE3f pose, camera cam, dataCPU<vec8<float>> &jpose_buffer, dataCPU<float> &e_buffer)
    {
        z_buffer.set(z_buffer.nodata);

        scene1 = scene;
        scene2 = scene;
        scene1.transform(cam, kpose);
        scene2.transform(cam, pose);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam.width / divi_x;
        int height = cam.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderJPoseWindow(kimage, kaffine, image, affine, d_image_d_pix, cam, win, jpose_buffer, e_buffer);
                // pool.enqueue(std::bind(&renderCPU::renderJPoseWindow, this, kimage, frame, cam, win, jpose_buffer, e_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderResidualParallel(sceneType &scene, dataCPU<float> &kimage, vec2<float> kaffine, Sophus::SE3f kpose, dataCPU<float> &image, vec2<float> affine, Sophus::SE3f pose, camera cam, dataCPU<float> &e_buffer)
    {
        z_buffer.set(z_buffer.nodata);

        scene1 = scene;
        scene2 = scene;
        scene1.transform(cam, kpose);
        scene2.transform(cam, pose);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam.width / divi_x;
        int height = cam.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderResidualWindow(kimage, kaffine, image, affine, cam, win, e_buffer);
                // pool.enqueue(std::bind(&renderCPU::renderResidualWindow, this, kimage, frame, cam, win, e_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
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

    void renderIdepthParallel(sceneType &scene, Sophus::SE3f pose, camera cam, dataCPU<float> &buffer)
    {
        z_buffer.set(z_buffer.nodata);

        scene2 = scene;
        scene2.transform(cam, pose);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam.width / divi_x;
        int height = cam.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderIdepthWindow(win, buffer);
                // pool.enqueue(std::bind(&renderCPU::renderIdepthWindow, this, win, buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderWeightParallel(sceneType &scene, Sophus::SE3f pose, camera cam, dataCPU<float> &buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata);

        scene2 = scene;
        scene2.transform(cam, pose);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam.width / divi_x;
        int height = cam.height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderWeightWindow(win, buffer, lvl);
                // pool.enqueue(std::bind(&renderCPU::renderWeightWindow, this, win, buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

private:
    void renderSmoothWindow(camera cam, window win, dataCPU<float> &buffer, float start = 1.0, float end = 2.0)
    {
        for (int y = win.min_y; y <= win.max_y; y++)
        {
            for (int x = win.min_x; x <= win.max_x; x++)
            {
                float val = start + (end - start) * float(y) / (cam.width - 1.0);
                buffer.set(val, y, x);
            }
        }
    }

    void renderRandomWindow(window win, dataCPU<float> *buffer, float min = 1.0, float max = 2.0)
    {
        for (int y = win.min_y; y <= win.max_y; y++)
        {
            for (int x = win.min_x; x <= win.max_x; x++)
            {
                if (buffer->get(y, x) == buffer->nodata)
                {
                    float val = (max - min) * float(rand() % 1000) / 1000.0 + min;
                    buffer->set(val, y, x);
                }
            }
        }
    }

    void renderInterpolateWindow(camera cam, window win, dataCPU<float> &buffer)
    {
        for (int y = win.min_y; y <= win.max_y; y++)
        {
            for (int x = win.min_x; x <= win.max_x; x++)
            {
                if (buffer.get(y, x) == buffer.nodata)
                {
                    int size = 10;
                    float acc = 0.0;
                    int count = 0;
                    while (true)
                    {
                        for (int y_ = y - size; y_ <= y + size; y_ += 1)
                        {
                            for (int x_ = x - size; x_ <= x + size; x_ += 1)
                            {
                                if (!cam.isPixVisible(x_, y_))
                                    continue;

                                auto val = buffer.get(y_, x_);
                                if (val == buffer.nodata)
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
                    buffer.set(acc, y, x);
                }
            }
        }
    }

    void renderIdepthLineSearchWindow(dataCPU<float> &kimage, dataCPU<float> &image, Sophus::SE3f imagePose, camera cam, window win, dataCPU<float> &buffer)
    {
        Sophus::SE3f kfTofPose = imagePose; // * kframe->getPose().inverse();
        Sophus::SE3f fTokfPose = kfTofPose.inverse();

        int size = 5;

        dataCPU<float> corrFrame = image;
        dataCPU<float> corrKframe = kimage;

        for (int y = win.min_y; y <= win.max_y; y++)
        {
            for (int x = win.min_x; x <= win.max_x; x++)
            {
                auto f_i = corrFrame.get(y, x);
                if (f_i == corrFrame.nodata)
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

                    auto kf_i = corrKframe.get(kf_pix(1), kf_pix(0));
                    if (kf_i == corrKframe.nodata)
                        continue;

                    float error = (f_i - kf_i) * (f_i - kf_i);

                    float last_error = buffer.get(y, x);

                    if (error < last_error || last_error == buffer.nodata)
                    {
                        buffer.set(error, y, x);

                        vec3<float> kf_ray = cam.pixToRay(kf_pix);
                        Eigen::Vector3f Y_e = fTokfPose * Eigen::Vector3f(f_ray(0), f_ray(1), f_ray(2));
                        vec3<float> Y(Y_e(0), Y_e(1), Y_e(2));
                        Eigen::Vector3f C_e = fTokfPose.translation();
                        vec3<float> C(C_e(0), C_e(1), C_e(2));

                        float f_depth = (C(1) / kf_ray(1) - C(2)) / (Y(2) - Y(1) / kf_ray(1));
                        // float f_depth = depth_min + i * (depth_max - depth_min) / kf_pix_diff_norm;
                        // float f_idepth = 1.0 / f_depth;

                        z_buffer.set(f_depth, y, x);
                    }
                }
            }
        }
    }

    void renderImageWindow(dataCPU<float> &image, camera cam, window win, dataCPU<float> &buffer)
    {
        std::vector<int> ids = scene2.getShapesIds();

        for (auto t_id : ids)
        {
            auto f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol->getCenterPix()))
                continue;

            if (f_pol->getScreenArea() < 0.0)
                continue;

            auto kf_pol = scene1.getShape(t_id);

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y <= pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x <= pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol->isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol->getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    float z_depth = z_buffer.get(y, x);
                    if (z_depth < f_depth && z_depth != z_buffer.nodata)
                        continue;

                    vec2<float> kf_pix = f_pol->getPix(f_pix, kf_pol.get());

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = image.get(kf_pix(1), kf_pix(0));
                    if (kf_i == image.nodata)
                        continue;

                    // buffer->set(std::exp(kf_a) * (kf_i - kf_b), y, x, lvl);
                    buffer.set(kf_i, y, x);
                    z_buffer.set(f_depth, y, x);
                }
            }
        }
    }

    void renderIdepthWindow(window win, dataCPU<float> &buffer)
    {
        std::vector<int> shapesIds = scene2.getShapesIds();

        for (int t_id : shapesIds)
        {
            auto f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() < 0.0)
                continue;

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y <= pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x <= pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);

                    if (f_depth <= 0.0)
                        continue;

                    float z_depth = z_buffer.get(y, x);
                    if (z_depth <= f_depth && z_depth != z_buffer.nodata)
                        continue;

                    buffer.set(1.0 / f_depth, y, x);
                    z_buffer.set(f_depth, y, x);
                }
            }
        }
    }

    void renderWeightWindow(window win, dataCPU<float> &buffer)
    {
        std::vector<int> shapesIds = scene2.getShapesIds();

        for (int t_id : shapesIds)
        {
            auto f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() < 0.0)
                continue;

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y <= pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x <= pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);

                    if (f_depth <= 0.0)
                        continue;

                    float z_depth = z_buffer.get(y, x);
                    if (z_depth <= f_depth && z_depth != z_buffer.nodata)
                        continue;

                    float f_weight = f_pol.getWeight(f_pix);

                    buffer.set(f_weight, y, x);
                    z_buffer.set(f_depth, y, x);
                }
            }
        }
    }

    void renderResidualWindow(dataCPU<float> &kimage, vec2<float> kimageAffine, dataCPU<float> &image, vec2<float> imageAffine, camera cam, window win, dataCPU<float> &e_buffer)
    {
        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

        std::vector<int> t_ids = scene2.getShapesIds();

        float kalpha = std::exp(-kimageAffine(0));
        float kbeta = kimageAffine(1);

        float alpha = std::exp(-imageAffine(0));
        float beta = imageAffine(1);

        for (int t_id : t_ids)
        {
            auto f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() <= min_area)
                continue;

            auto kf_pol = scene1.getShape(t_id);

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y <= pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x <= pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_depth = z_buffer.get(y, x);
                    if (l_depth < f_depth && l_depth != z_buffer.nodata)
                        continue;

                    vec2<float> kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    auto f_i = image.get(y, x);

                    if (kf_i == kimage.nodata || f_i == image.nodata)
                        continue;

                    float f_i_cor = alpha * (f_i - beta);
                    float kf_i_cor = kalpha * (kf_i - kbeta);

                    float residual = (f_i_cor - kf_i_cor);

                    // vec3<float> bar = f_pol.getBarycentric();
                    float weight = 1.0; // bar(0)*bar(1)*bar(2);

                    e_buffer.set(residual * weight, y, x);
                    z_buffer.set(f_depth, y, x);
                }
            }
        }
    }

    void renderJPoseWindow(dataCPU<float> &kimage, vec2<float> kimageAffine, dataCPU<float> &image, vec2<float> imageAffine, dataCPU<vec2<float>> &d_image_d_pix, camera cam, window win, dataCPU<vec8<float>> &jpose_buffer, dataCPU<float> &e_buffer)
    {
        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

        float kalpha = std::exp(-kimageAffine(0));
        float kbeta = kimageAffine(1);

        float alpha = std::exp(-imageAffine(0));
        float beta = imageAffine(1);

        std::vector<int> t_ids = scene2.getShapesIds();

        for (int t_id : t_ids)
        {
            auto f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() <= min_area)
                continue;

            auto kf_pol = scene1.getShape(t_id);

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y <= pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x <= pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_depth = z_buffer.get(y, x);
                    if (l_depth < f_depth && l_depth != z_buffer.nodata)
                        continue;

                    vec2<float> kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    vec3<float> f_ray = cam.pixToRay(x, y);
                    vec3<float> f_ver = f_ray * f_depth;
                    // vec3<float> kf_ray = cam.pixToRay(kf_pix);

                    auto kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    auto f_i = image.get(y, x);
                    vec2<float> d_f_i_d_pix = d_image_d_pix.get(y, x);

                    if (kf_i == kimage.nodata || f_i == image.nodata) // || d_f_i_d_pix == frame->getdIdpixImage().nodata)
                        continue;

                    float v0 = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    float v1 = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    float v2 = -(v0 * f_ver(0) + v1 * f_ver(1)) / f_ver(2);
                    vec3<float> d_f_i_d_tra(v0, v1, v2);
                    vec3<float> d_f_i_d_rot(-f_ver(2) * v1 + f_ver(1) * v2, f_ver(2) * v0 - f_ver(0) * v2, -f_ver(1) * v0 + f_ver(0) * v1);

                    float f_i_cor = alpha * (f_i - beta);
                    float kf_i_cor = kalpha * (kf_i - kbeta);

                    vec2<float> d_f_i_d_affine(-f_i_cor, -alpha);

                    vec8<float> j_pose = {d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2), d_f_i_d_affine(0), d_f_i_d_affine(1)};

                    float residual = (f_i_cor - kf_i_cor);

                    // vec3<float> bar = f_pol.getBarycentric();
                    float weight = 1.0; // bar(0)*bar(1)*bar(2);

                    jpose_buffer.set(j_pose * weight, y, x);
                    e_buffer.set(residual * weight, y, x);
                    z_buffer.set(f_depth, y, x);
                }
            }
        }
    }

    void renderJLightAffineWindow(dataCPU<float> &kimage, vec2<float> kimageAffine, dataCPU<float> &image, vec2<float> imageAffine, camera cam, window win, dataCPU<vec2<float>> &jlightaffine_buffer, dataCPU<float> &e_buffer)
    {
        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

        std::vector<int> t_ids = scene2.getShapesIds();

        float kalpha = std::exp(-kimageAffine(0));
        float kbeta = kimageAffine(1);

        float alpha = std::exp(-imageAffine(0));
        float beta = imageAffine(1);

        for (int t_id : t_ids)
        {
            auto f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() <= min_area)
                continue;

            auto kf_pol = scene1.getShape(t_id);

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y <= pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x <= pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_depth = z_buffer.get(y, x);
                    if (l_depth < f_depth && l_depth != z_buffer.nodata)
                        continue;

                    vec2<float> kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    auto f_i = image.get(y, x);

                    if (kf_i == kimage.nodata || f_i == image.nodata)
                        continue;

                    float f_i_cor = alpha * (f_i - beta);
                    float kf_i_cor = kalpha * (kf_i - kbeta);

                    vec2<float> j_lightaffine(-f_i_cor, -alpha);

                    float residual = f_i_cor - kf_i_cor;

                    jlightaffine_buffer.set(j_lightaffine, y, x);
                    e_buffer.set(residual, y, x);
                    z_buffer.set(f_depth, y, x);
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

    template <typename jmapType, typename idsType>
    void renderJMapWindow(dataCPU<float> &kimage, vec2<float> kimageAffine, dataCPU<float> &image, vec2<float> imageAffine, dataCPU<vec2<float>> &d_image_d_pix, Sophus::SE3f imagePose, camera cam, window win, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &e_buffer, dataCPU<idsType> &pId_buffer)
    {
        float min_area = 0.0; //(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3 / 4;

        Sophus::SE3f kfTofPose = imagePose;
        // Sophus::SE3f fTokfPose = kfTofPose.inverse();

        float kalpha = std::exp(-kimageAffine(0));
        float kbeta = kimageAffine(1);

        float alpha = std::exp(-imageAffine(0));
        float beta = imageAffine(1);

        std::vector<int> t_ids = scene2.getShapesIds();

        // int shapeDoF = scene2->getShapesDoF();

        for (int t_id : t_ids)
        {
            // std::vector<int> p_ids = scene2->getShapeParamsIds(t_id);

            auto f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() < min_area)
                continue;

            auto kf_pol = scene1.getShape(t_id);

            if (kf_pol.getScreenArea() < min_area)
                continue;

            idsType ids = kf_pol.getParamIds();

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y <= pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x <= pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_idepth = z_buffer.get(y, x);
                    if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                        continue;

                    vec2<float> kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    vec3<float> f_ray = cam.pixToRay(f_pix);
                    vec3<float> f_ver = f_ray * f_depth;
                    vec3<float> kf_ray = cam.pixToRay(kf_pix);

                    auto kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    auto f_i = image.get(y, x);
                    vec2<float> d_f_i_d_pix = d_image_d_pix.get(y, x);

                    if (kf_i == kimage.nodata || f_i == image.nodata) // || d_f_i_d_pix == frame->getdIdpixImage().nodata)
                        continue;

                    float f_i_cor = alpha * (f_i - beta);
                    float kf_i_cor = kalpha * (kf_i - kbeta);

                    float residual = f_i_cor - kf_i_cor;

                    vec3<float> d_f_i_d_f_ver;
                    d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                    Eigen::Vector3f d_f_ver_d_kf_depth_e = kfTofPose.rotationMatrix() * Eigen::Vector3f(kf_ray(0), kf_ray(1), kf_ray(2));

                    vec3<float> d_f_ver_d_kf_depth(d_f_ver_d_kf_depth_e(0), d_f_ver_d_kf_depth_e(1), d_f_ver_d_kf_depth_e(2));

                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    // this could be the jacobian of the depth of the 3 vertices in a triangle
                    // or the jacobian of the normal + depth of a surfel
                    jmapType jacs = kf_pol.getParamJacobian(kf_pix) * d_f_i_d_kf_depth;

                    // vec3<float> bar = f_pol.getBarycentric();
                    float weight = 1.0; // bar(0)*bar(1)*bar(2);

                    e_buffer.set(residual * weight, y, x);
                    jmap_buffer.set(jacs * weight, y, x);
                    pId_buffer.set(ids, y, x);
                }
            }
        }
    }

    template <typename jmapType, typename idsType>
    void renderJPoseMapWindow(dataCPU<float> &kimage, vec2<float> kimageAffine, dataCPU<float> &image, vec2<float> imageAffine, dataCPU<vec2<float>> &d_image_d_pix, Sophus::SE3f imagePose, camera cam, window win, dataCPU<vec8<float>> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &e_buffer, dataCPU<idsType> &pId_buffer)
    {
        float min_area = 0.0; //(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3.0 / 4.0;

        Sophus::SE3f kfTofPose = imagePose;
        // Sophus::SE3f fTokfPose = kfTofPose.inverse();

        float kalpha = std::exp(-kimageAffine(0));
        float kbeta = kimageAffine(1);

        float alpha = std::exp(-imageAffine(0));
        float beta = imageAffine(1);

        std::vector<int> t_ids = scene2.getShapesIds();
        // int shapeDoF = scene2->getShapesDoF();

        for (auto t_id : t_ids)
        {
            auto f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            float f_pol_area = f_pol.getScreenArea();
            if (f_pol_area <= min_area)
                continue;

            auto kf_pol = scene1.getShape(t_id);

            float kf_pol_area = kf_pol.getScreenArea();
            if (kf_pol_area <= min_area)
                continue;

            /*
            float p_area;
            if (kf_pol_area > f_pol_area)
                p_area = f_pol_area / kf_pol_area;
            else
                p_area = kf_pol_area / f_pol_area;

            if (p_area < 0.5)
                continue;
            */

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            idsType ids = kf_pol.getParamIds();

            for (int y = pol_win.min_y; y <= pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x <= pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    // if (!cam.isPixVisible(f_pix))
                    //     continue;

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_idepth = z_buffer.get(f_pix(1), f_pix(0));
                    if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                        continue;

                    vec2<float> kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    auto f_i = image.get(y, x);
                    vec2<float> d_f_i_d_pix = d_image_d_pix.get(y, x);

                    if (kf_i == kimage.nodata || f_i == image.nodata) // || d_f_i_d_pix == frame->getdIdpixImage().nodata)
                        continue;

                    vec3<float> f_ray = cam.pixToRay(f_pix);
                    vec3<float> f_ver = f_ray * f_depth;
                    vec3<float> kf_ray = cam.pixToRay(kf_pix);

                    vec3<float> d_f_i_d_f_ver;
                    d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                    vec3<float> d_f_i_d_tra(d_f_i_d_f_ver(0), d_f_i_d_f_ver(1), d_f_i_d_f_ver(2));
                    vec3<float> d_f_i_d_rot(-f_ver(2) * d_f_i_d_f_ver(1) + f_ver(1) * d_f_i_d_f_ver(2), f_ver(2) * d_f_i_d_f_ver(0) - f_ver(0) * d_f_i_d_f_ver(2), -f_ver(1) * d_f_i_d_f_ver(0) + f_ver(0) * d_f_i_d_f_ver(1));

                    float f_i_cor = alpha * (f_i - beta);
                    float kf_i_cor = kalpha * (kf_i - kbeta);

                    vec2<float> d_f_i_d_f_affine(-f_i_cor, -alpha);

                    vec8<float> jpose = {d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2), d_f_i_d_f_affine(0), d_f_i_d_f_affine(1)};

                    Eigen::Vector3f d_f_ver_d_kf_depth_e = kfTofPose.rotationMatrix() * Eigen::Vector3f(kf_ray(0), kf_ray(1), kf_ray(2));

                    vec3<float> d_f_ver_d_kf_depth(d_f_ver_d_kf_depth_e(0), d_f_ver_d_kf_depth_e(1), d_f_ver_d_kf_depth_e(2));
                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    jmapType jacs = kf_pol.getParamJacobian(kf_pix) * d_f_i_d_kf_depth;

                    float error = f_i_cor - kf_i_cor;

                    // vec3<float> bar = f_pol.getBarycentric();
                    float weight = 1.0; // bar(0)*bar(1)*bar(2);

                    e_buffer.set(error * weight, y, x);
                    jpose_buffer.set(jpose * weight, y, x);
                    jmap_buffer.set(jacs * weight, y, x);
                    pId_buffer.set(ids, y, x);
                }
            }
        }
    }

    void renderDebugWindow(dataCPU<float> &image, window win, dataCPU<float> &buffer)
    {
        std::vector<int> ids = scene2.getShapesIds();

        for (auto t_id : ids)
        {
            auto f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() < 0.0)
                continue;

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y <= pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x <= pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    // float f_depth = f_pol.getDepth(f_pix);
                    bool isLine = f_pol.isEdge(f_pix);

                    float f_i = image.get(y, x);
                    f_i /= 255.0;

                    // z buffer
                    // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                    // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                    //    continue;

                    if (isLine)
                        buffer.set(1.0, y, x);
                    else
                        // buffer->set(1.0 / f_depth, y, x, lvl);
                        buffer.set(f_i, y, x);
                }
            }
        }
    }

    sceneType scene1;
    sceneType scene2;
    dataCPU<float> z_buffer;
    ThreadPool<RENDERER_NTHREADS> pool;
};
