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

    void renderIdepthLineSearch(frameCPU *kframe, frameCPU *frame, camera cam, dataCPU<float> *buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        window win(0, cam.width-1, 0, cam.height-1);

        renderIdepthLineSearchWindow(kframe, frame, cam, win, buffer, lvl);
    }

    void renderRandom(camera cam, dataCPU<float> *buffer, int lvl)
    {
        window win(0, cam.width-1, 0, cam.height-1);

        renderRandomWindow(cam, win, buffer, lvl);
    }

    void renderSmooth(camera cam, dataCPU<float> *buffer, int lvl, float start = 1.0, float end = 2.0)
    {
        window win(0, cam.width-1, 0, cam.height-1);

        renderSmoothWindow(cam, win, buffer, lvl, start, end);
    }

    void renderInterpolate(camera cam, dataCPU<float> *buffer, int lvl)
    {
        window win(0, cam.width-1, 0, cam.height-1);

        renderInterpolateWindow(cam, win, buffer, lvl);
    }

    void renderImage(sceneType *kscene, dataCPU<float> *kimage, Sophus::SE3f pose, camera cam, dataCPU<float> *buffer, int lvl)
    {
        scene1 = kscene->clone();
        scene2 = kscene->clone();
        // scene1->transform(pose);
        scene2->transform(pose);
        scene1->project(cam);
        scene2->project(cam);

        z_buffer.set(z_buffer.nodata, lvl);

        window win(0, cam.width-1, 0, cam.height-1);

        renderImageWindow(kimage, cam, win, buffer, lvl);
    }

    void renderImageParallel(sceneType *kscene, dataCPU<float> *kimage, Sophus::SE3f pose, camera cam, dataCPU<float> *buffer, int lvl)
    {
        scene1 = kscene->clone();
        scene2 = kscene->clone();
        // scene1->transform(kframe->getPose());
        scene2->transform(pose);
        scene1->project(cam);
        scene2->project(cam);

        z_buffer.set(z_buffer.nodata, lvl);

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

                window win(min_x, max_x-1, min_y, max_y-1);

                renderImageWindow(kimage, cam, win, buffer, lvl);
                //pool.enqueue(std::bind(&renderCPU::renderImageWindow, this, kimage, cam, win, buffer, lvl));
            }
        }

        //pool.waitUntilDone();
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

    void renderDebug(sceneType *scene, dataCPU<float> *image, Sophus::SE3f pose, camera cam, dataCPU<float> *buffer, int lvl)
    {
        scene1 = scene->clone();
        scene2 = scene->clone();
        // scene1->transform(frame->getPose());
        scene2->transform(pose);
        scene1->project(cam);
        scene2->project(cam);

        z_buffer.set(z_buffer.nodata, lvl);

        window win(0, cam.width-1, 0, cam.height-1);

        renderDebugWindow(image, win, buffer, lvl);
    }

    void renderDebugParallel(sceneType *scene, dataCPU<float> *image, Sophus::SE3f pose, camera cam, dataCPU<float> *buffer, int lvl)
    {
        scene1 = scene->clone();
        scene2 = scene->clone();
        scene1.transform(cam, Sophus::SE3f());
        scene2.transform(cam, pose);

        z_buffer.set(z_buffer.nodata, lvl);

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

                window win(min_x, max_x-1, min_y, max_y-1);

                renderDebugWindow(image, win, buffer, lvl);
                //pool.enqueue(std::bind(&renderCPU::renderDebugWindow, this, image, win, buffer, lvl));
            }
        }

        //pool.waitUntilDone();
    }

    template <typename jmapType, typename idsType>
    void renderJMap(sceneType *kscene, dataCPU<float> *kimage, frameCPU *frame, camera cam, dataCPU<jmapType> *jmap_buffer, dataCPU<float> *e_buffer, dataCPU<idsType> *pId_buffer, int lvl)
    {
        scene1 = kscene->clone();
        scene2 = kscene->clone();
        //scene1->transform(kframe->getPose());
        scene2->transform(frame->getPose());
        scene1->project(cam);
        scene2->project(cam);

        z_buffer.set(z_buffer.nodata, lvl);

        window win(0, cam.width-1, 0, cam.height-1);

        renderJMapWindow(&kimage, frame, cam, win, jmap_buffer, e_buffer, pId_buffer, lvl);
    }

    template <typename jmapType, typename idsType>
    void renderJMapParallel(sceneType *kscene, dataCPU<float> *kimage, frameCPU *frame, camera &cam, dataCPU<jmapType> *jmap_buffer, dataCPU<float> *e_buffer, dataCPU<idsType> *pId_buffer, int lvl)
    {
        scene1 = kscene->clone();
        scene2 = kscene->clone();
        //scene1->transform(kframe->getPose());
        scene2.transform(frame->getPose());
        scene1.project(cam);
        scene2.project(cam);

        z_buffer.set(z_buffer.nodata, lvl);

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

                window win(min_x, max_x-1, min_y, max_y-1);

                renderJMapWindow(kimage, frame, cam, win, jmap_buffer, e_buffer, pId_buffer, lvl);
                //pool.enqueue(std::bind(&renderCPU::renderJMapWindow, this, kimage, frame, cam, win, jmap_buffer, e_buffer, pId_buffer, lvl));
            }
        }

        //pool.waitUntilDone();
    }

    template <typename jmapType, typename idsType>
    void renderJPoseMap(sceneType *kscene, dataCPU<float> *kimage, frameCPU *frame, camera cam, dataCPU<vec8<float>> *jpose_buffer, dataCPU<jmapType> *jmap_buffer, dataCPU<float> *e_buffer, dataCPU<idsType> *pId_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        scene1 = kscene->clone();
        scene2 = kscene->clone();
        //scene1->transform(kframe->getPose());
        scene2->transform(frame->getPose());
        scene1->project(cam);
        scene2->project(cam);

        window win(0, cam.width-1, 0, cam.height-1);

        renderJPoseMapWindow(kimage, frame, cam, win, jpose_buffer, jmap_buffer, e_buffer, pId_buffer, lvl);
    }

    template <typename jmapType, typename idsType>
    void renderJPoseMapParallel(sceneType *kscene, dataCPU<float> *kimage, frameCPU *frame, camera cam, dataCPU<vec8<float>> *jpose_buffer, dataCPU<jmapType> *jmap_buffer, dataCPU<float> *e_buffer, dataCPU<idsType> *pId_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        scene1 = kscene->clone();
        scene2 = kscene->clone();
        //scene1->transform(kframe->getPose());
        scene2.transform(frame->getPose());
        scene1.project(cam);
        scene2.project(cam);

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

                window win(min_x, max_x-1, min_y, max_y-1);

                renderJPoseMapWindow(kimage, frame, cam, win, jpose_buffer, jmap_buffer, e_buffer, pId_buffer, lvl);
                //pool.enqueue(std::bind(&renderCPU::renderJPoseMapWindow, this, kimage, frame, cam, win, jpose_buffer, jmap_buffer, e_buffer, pId_buffer, lvl));
            }
        }

        //pool.waitUntilDone();
    }

    void renderJLightAffineParallel(sceneType *kscene, dataCPU<float> *kimage, frameCPU *frame, camera cam, dataCPU<vec2<float>> *jlightaffine_buffer, dataCPU<float> *e_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        scene1 = kscene->clone();
        scene2 = kscene->clone();
        //scene1->transform(kframe->getPose());
        scene2.transform(frame->getPose());
        scene1.project(cam);
        scene2.project(cam);

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

                window win(min_x, max_x-1, min_y, max_y-1);

                renderJLightAffineWindow(kimage, frame, cam, win, jlightaffine_buffer, e_buffer, lvl);
                //pool.enqueue(std::bind(&renderCPU::renderJLightAffineWindow, this, kimage, frame, cam, win, jlightaffine_buffer, e_buffer, lvl));
            }
        }

        //pool.waitUntilDone();
    }

    void renderJPose(sceneType *kscene, dataCPU<float> *kimage, frameCPU *frame, camera cam, dataCPU<vec8<float>> *jpose_buffer, dataCPU<float> *e_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        scene1 = kscene->clone();
        scene2 = kscene->clone();
        //scene1->transform(kframe->getPose());
        scene2->transform(frame->getPose());
        scene1->project(cam);
        scene2->project(cam);

        window win(0, cam.width-1, 0, cam.height-1);

        renderJPoseWindow(kimage, frame, cam, win, jpose_buffer, e_buffer, lvl);
    }

    void renderJPoseParallel(sceneType *kscene, dataCPU<float> *kimage, frameCPU *frame, camera cam, dataCPU<vec8<float>> *jpose_buffer, dataCPU<float> *e_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        scene1 = kscene->clone();
        scene2 = kscene->clone();
        //scene1->transform(kframe->getPose());
        scene2.transform(frame->getPose());
        scene1.project(cam);
        scene2.project(cam);

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

                window win(min_x, max_x-1, min_y, max_y-1);

                renderJPoseWindow(kimage, frame, cam, win, jpose_buffer, e_buffer, lvl);
                //pool.enqueue(std::bind(&renderCPU::renderJPoseWindow, this, kimage, frame, cam, win, jpose_buffer, e_buffer, lvl));
            }
        }

        //pool.waitUntilDone();
    }

    void renderResidual(sceneType *kscene, dataCPU<float> *kimage, frameCPU *frame, camera cam, dataCPU<float> *e_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        scene1 = kscene->clone();
        scene2 = kscene->clone();
        //scene1->transform(kframe->getPose());
        scene2->transform(frame->getPose());
        scene1->project(cam);
        scene2->project(cam);

        window win(0, cam.width-1, 0, cam.height-1);

        renderResidualWindow(kimage, frame, cam, win, e_buffer, lvl);
    }

    void renderResidualParallel(sceneType *kscene, dataCPU<float> *kimage, frameCPU *frame, camera cam, dataCPU<float> *e_buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        scene1 = kscene->clone();
        scene2 = kscene->clone();
        //scene1->transform(kframe->getPose());
        scene2.transform(frame->getPose());
        scene1.project(cam);
        scene2.project(cam);

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

                window win(min_x, max_x-1, min_y, max_y-1);

                renderResidualWindow(kimage, frame, cam, win, e_buffer, lvl);
                //pool.enqueue(std::bind(&renderCPU::renderResidualWindow, this, kimage, frame, cam, win, e_buffer, lvl));
            }
        }

        //pool.waitUntilDone();
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

    void renderIdepth(sceneType *scene, Sophus::SE3f pose, camera cam, dataCPU<float> *buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        scene2 = scene->clone();
        scene2.transform(pose);
        scene2.project(cam);

        window win(0, cam.width-1, 0, cam.height-1);

        renderIdepthWindow(win, buffer, lvl);
    }

    void renderIdepthParallel(sceneType *scene, Sophus::SE3f pose, camera cam, dataCPU<float> *buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        scene2 = scene->clone();
        scene2.transform(pose);
        scene2.project(cam);

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

                window win(min_x, max_x-1, min_y, max_y-1);

                renderIdepthWindow(win, buffer, lvl);
                //pool.enqueue(std::bind(&renderCPU::renderIdepthWindow, this, win, buffer, lvl));
            }
        }

        //pool.waitUntilDone();
    }

    void renderWeight(sceneType *scene, Sophus::SE3f pose, camera cam, dataCPU<float> *buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        scene2 = scene->clone();
        scene2.transform(pose);
        scene2.project(cam);

        window win(0, cam.width-1, 0, cam.height-1);

        renderWeightWindow(win, buffer, lvl);
    }

    void renderWeightParallel(sceneType *scene, Sophus::SE3f pose, camera cam, dataCPU<float> *buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        scene2 = scene->clone();
        scene2.transform(pose);
        scene2.project(cam);

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

                window win(min_x, max_x-1, min_y, max_y-1);

                renderIdepthWindow(win, buffer, lvl);
                //pool.enqueue(std::bind(&renderCPU::renderWeightWindow, this, win, buffer, lvl));
            }
        }

        //pool.waitUntilDone();
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
                        for (int y_ = y - size; y_ <= y + size; y_ += 1)
                        {
                            for (int x_ = x - size; x_ <= x + size; x_ += 1)
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
        Sophus::SE3f kfTofPose = frame->getPose();// * kframe->getPose().inverse();
        Sophus::SE3f fTokfPose = kfTofPose.inverse();

        int size = 5;

        dataCPU<float> corrFrame = frame->getRawImage();
        dataCPU<float> corrKframe = kframe->getRawImage();

        for (int y = win.min_y; y < win.max_y; y++)
        {
            for (int x = win.min_x; x < win.max_x; x++)
            {
                auto f_i = corrFrame.get(y, x, lvl);
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

                    auto kf_i = corrKframe.get(kf_pix(1), kf_pix(0), lvl);
                    if (kf_i == corrKframe.nodata)
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

    void renderImageWindow(dataCPU<float> *image, camera cam, window win, dataCPU<float> *buffer, int lvl)
    {
        std::vector<int> ids = scene2->getShapesIds();

        for (auto t_id : ids)
        {
            auto f_pol = scene2->getShape(t_id);

            if(!win.isPixInWindow(f_pol->getCenterPix()))
                continue;

            if (f_pol->getScreenArea() < 0.0)
                continue;

            auto kf_pol = scene1->getShape(t_id);

            window pol_win = f_pol->getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y < pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x < pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol->isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol->getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    float z_depth = z_buffer.get(y, x, lvl);
                    if (z_depth < f_depth && z_depth != z_buffer.nodata)
                        continue;

                    vec2<float> kf_pix = f_pol->getPix(f_pix, kf_pol.get());

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = image->get(kf_pix(1), kf_pix(0), lvl);
                    if (kf_i == image->nodata)
                        continue;

                    // buffer->set(std::exp(kf_a) * (kf_i - kf_b), y, x, lvl);
                    buffer->set(kf_i, y, x, lvl);
                    z_buffer.set(f_depth, y, x, lvl);
                }
            }
        }
    }

    void renderIdepthWindow(window win, dataCPU<float> *buffer, int lvl)
    {
        std::vector<int> shapesIds = scene2.getShapesIds();

        for (int t_id : shapesIds)
        {
            auto f_pol = scene2.getShape(t_id);

            if(!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() < 0.0)
                continue;

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y < pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x < pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);

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

    void renderWeightWindow(window win, dataCPU<float> *buffer, int lvl)
    {
        std::vector<int> shapesIds = scene2.getShapesIds();

        for (int t_id : shapesIds)
        {
            auto f_pol = scene2.getShape(t_id);

            if(!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol->getScreenArea() < 0.0)
                continue;

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y < pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x < pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);

                    if (f_depth <= 0.0)
                        continue;

                    float z_depth = z_buffer.get(y, x, lvl);
                    if (z_depth <= f_depth && z_depth != z_buffer.nodata)
                        continue;

                    float f_weight = f_pol.getWeight(f_pix);

                    buffer->set(f_weight, y, x, lvl);
                    z_buffer.set(f_depth, y, x, lvl);
                }
            }
        }
    }

    void renderResidualWindow(dataCPU<float> *kimage, frameCPU *frame, camera cam, window win, dataCPU<float> *e_buffer, int lvl)
    {
        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

        std::vector<int> t_ids = scene2.getShapesIds();

        vec2<float> affine = frame->getAffine();
        float alpha = std::exp(-affine(0));
        float beta = affine(1);

        for (int t_id : t_ids)
        {
            auto f_pol = scene2.getShape(t_id);

            if(!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() <= min_area)
                continue;

            auto kf_pol = scene1.getShape(t_id);

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y < pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x < pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_depth = z_buffer.get(y, x, lvl);
                    if (l_depth < f_depth && l_depth != z_buffer.nodata)
                        continue;

                    vec2<float> kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = kimage->get(kf_pix(1), kf_pix(0), lvl);
                    auto f_i = frame->getRawImage().get(y, x, lvl);

                    if (kf_i == kimage->nodata || f_i == frame->getRawImage().nodata)
                        continue;

                    float f_i_cor = alpha*(f_i - beta);

                    float residual = f_i_cor - kf_i;

                    e_buffer->set(residual, y, x, lvl);
                    z_buffer.set(f_depth, y, x, lvl);
                }
            }
        }
    }

    void renderJPoseWindow(dataCPU<float> *kimage, frameCPU *frame, camera cam, window win, dataCPU<vec8<float>> *jpose_buffer, dataCPU<float> *e_buffer, int lvl)
    {
        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

        vec2<float> affine = frame->getAffine();
        float alpha = std::exp(-affine(0));
        float beta = affine(1);

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

            for (int y = pol_win.min_y; y < pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x < pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_depth = z_buffer.get(y, x, lvl);
                    if (l_depth < f_depth && l_depth != z_buffer.nodata)
                        continue;

                    vec2<float> kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    vec3<float> f_ray = cam.pixToRay(x, y);
                    vec3<float> f_ver = f_ray * f_depth;
                    vec3<float> kf_ray = cam.pixToRay(kf_pix);

                    auto kf_i = kimage->get(kf_pix(1), kf_pix(0), lvl);
                    auto f_i = frame->getRawImage().get(y, x, lvl);
                    vec2<float> d_f_i_d_pix = frame->getdIdpixImage().get(y, x, lvl);

                    if (kf_i == kimage->nodata || f_i == frame->getRawImage().nodata || d_f_i_d_pix == frame->getdIdpixImage().nodata)
                        continue;

                    float v0 = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    float v1 = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    float v2 = -(v0 * f_ver(0) + v1 * f_ver(1)) / f_ver(2);
                    vec3<float> d_f_i_d_tra(v0, v1, v2);
                    vec3<float> d_f_i_d_rot(-f_ver(2) * v1 + f_ver(1) * v2, f_ver(2) * v0 - f_ver(0) * v2, -f_ver(1) * v0 + f_ver(0) * v1);

                    float f_i_cor = alpha*(f_i - beta);

                    vec2<float> d_f_i_d_affine(-f_i_cor, -alpha);

                    vec8<float> j_pose = {d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2), d_f_i_d_affine(0), d_f_i_d_affine(1)};

                    float residual = f_i_cor - kf_i;

                    jpose_buffer->set(j_pose, y, x, lvl);
                    e_buffer->set(residual, y, x, lvl);
                    z_buffer.set(f_depth, y, x, lvl);
                }
            }
        }
    }

    void renderJLightAffineWindow(dataCPU<float> *kimage, frameCPU *frame, camera cam, window win, dataCPU<vec2<float>> *jlightaffine_buffer, dataCPU<float> *e_buffer, int lvl)
    {
        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

        std::vector<int> t_ids = scene2.getShapesIds();

        vec2<float> affine = frame->getAffine();
        float alpha = std::exp(-affine(0));
        float beta = affine(1);

        for (int t_id : t_ids)
        {
            auto f_pol = scene2.getShape(t_id);

            if(!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() <= min_area)
                continue;

            auto kf_pol = scene1.getShape(t_id);

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y < pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x < pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_depth = z_buffer.get(y, x, lvl);
                    if (l_depth < f_depth && l_depth != z_buffer.nodata)
                        continue;

                    vec2<float> kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = kimage->get(kf_pix(1), kf_pix(0), lvl);
                    auto f_i = frame->getRawImage().get(y, x, lvl);

                    if (kf_i == kimage->nodata || f_i == frame->getRawImage().nodata)
                        continue;

                    float f_i_cor = alpha*(f_i - beta);

                    vec2<float> j_lightaffine(-f_i_cor, -alpha);

                    float residual = f_i_cor - kf_i;

                    jlightaffine_buffer->set(j_lightaffine, y, x, lvl);
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

    template <typename jmapType, typename idsType>
    void renderJMapWindow(dataCPU<float> *kimage, frameCPU *frame, camera cam, window win, dataCPU<jmapType> *jmap_buffer, dataCPU<float> *e_buffer, dataCPU<idsType> *pId_buffer, int lvl)
    {
        float min_area = 0.0; //(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3 / 4;

        Sophus::SE3f kfTofPose = frame->getPose();
        Sophus::SE3f fTokfPose = kfTofPose.inverse();

        vec2<float> affine = frame->getAffine();
        float alpha = std::exp(-affine(0));
        float beta = affine(1);

        std::vector<int> t_ids = scene2.getShapesIds();

        //int shapeDoF = scene2->getShapesDoF();

        for (int t_id : t_ids)
        {
            //std::vector<int> p_ids = scene2->getShapeParamsIds(t_id);

            auto f_pol = scene2.getShape(t_id);

            if(win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() < min_area)
                continue;

            auto kf_pol = scene1.getShape(t_id);

            if (kf_pol.getScreenArea() < min_area)
                continue;

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y < pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x < pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_idepth = z_buffer.get(y, x, lvl);
                    if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                        continue;
                
                    vec2<float> kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    vec3<float> f_ray = cam.pixToRay(f_pix);
                    vec3<float> f_ver = f_ray * f_depth;
                    vec3<float> kf_ray = cam.pixToRay(kf_pix);

                    auto kf_i = kimage->get(kf_pix(1), kf_pix(0), lvl);
                    auto f_i = frame->getRawImage().get(y, x, lvl);
                    vec2<float> d_f_i_d_pix = frame->getdIdpixImage().get(y, x, lvl);

                    if (kf_i == kimage->nodata || f_i == frame->getRawImage().nodata || d_f_i_d_pix == frame->getdIdpixImage().nodata)
                        continue;

                    float f_i_cor = alpha*(f_i - beta);

                    float residual = f_i_cor - kf_i;

                    vec3<float> d_f_i_d_f_ver;
                    d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                    Eigen::Vector3f d_f_ver_d_kf_depth_e = kfTofPose.rotationMatrix() * Eigen::Vector3f(kf_ray(0), kf_ray(1), kf_ray(2));

                    vec3<float> d_f_ver_d_kf_depth(d_f_ver_d_kf_depth_e(0), d_f_ver_d_kf_depth_e(1), d_f_ver_d_kf_depth_e(2));

                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    // this could be the jacobian of the depth of the 3 vertices in a triangle
                    // or the jacobian of the normal + depth of a surfel
                    jmapType jacs = kf_pol.getParamJacobian(kf_pix)*d_f_i_d_kf_depth;
                    idsType ids = kf_pol.getParamIds();

                    e_buffer->set(residual, y, x, lvl);
                    jmap_buffer->set(jacs, y, x, lvl);
                    pId_buffer->set(ids, y, x, lvl);
                }
            }
        }
    }

    template <typename jmapType, typename idsType>
    void renderJPoseMapWindow(dataCPU<float> *kimage, frameCPU *frame, camera cam, window win, dataCPU<vec8<float>> *jpose_buffer, dataCPU<jmapType> *jmap_buffer, dataCPU<float> *e_buffer, dataCPU<idsType> *pId_buffer, int lvl)
    {
        float min_area = 0.0;//(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3.0 / 4.0;

        Sophus::SE3f kfTofPose = frame->getPose();
        Sophus::SE3f fTokfPose = kfTofPose.inverse();

        vec2<float> affine = frame->getAffine();
        float alpha = std::exp(-affine(0));
        float beta = affine(1);

        std::vector<int> t_ids = scene2.getShapesIds();
        //int shapeDoF = scene2->getShapesDoF();

        for (auto t_id : t_ids)
        {
            auto f_pol = scene2.getShape(t_id);

            if(!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            float f_pol_area = f_pol.getScreenArea();
            if (f_pol_area <= 0.0)
                continue;

            auto kf_pol = scene1.getShape(t_id);

            float kf_pol_area = kf_pol.getScreenArea();
            if (kf_pol_area <= 0.0)
                continue;

            float p_area;
            if (kf_pol_area > f_pol_area)
                p_area = f_pol_area / kf_pol_area;
            else
                p_area = kf_pol_area / f_pol_area;

            if (p_area < 0.5)
                continue;

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y < pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x < pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                    if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                        continue;

                    vec2<float> kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = kimage->get(kf_pix(1), kf_pix(0), lvl);
                    auto f_i = frame->getRawImage().get(y, x, lvl);
                    vec2<float> d_f_i_d_pix = frame->getdIdpixImage().get(y, x, lvl);

                    if (kf_i == kimage->nodata || f_i == frame->getRawImage().nodata || d_f_i_d_pix == frame->getdIdpixImage().nodata)
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

                    float f_i_cor = alpha*(f_i - beta);
                    
                    vec2<float> d_f_i_d_f_affine(-f_i_cor, -alpha);

                    vec8<float> jpose = {d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2), d_f_i_d_f_affine(0), d_f_i_d_f_affine(1)};

                    jpose_buffer->set(jpose, y, x, lvl);

                    Eigen::Vector3f d_f_ver_d_kf_depth_e = kfTofPose.rotationMatrix() * Eigen::Vector3f(kf_ray(0), kf_ray(1), kf_ray(2));

                    vec3<float> d_f_ver_d_kf_depth(d_f_ver_d_kf_depth_e(0), d_f_ver_d_kf_depth_e(1), d_f_ver_d_kf_depth_e(2));
                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    jmapType jacs = kf_pol.getParamJacobian(kf_pix)*d_f_i_d_kf_depth;
                    idsType ids = kf_pol.getParamIds();

                    float error = f_i_cor - kf_i;

                    e_buffer->set(error, y, x, lvl);
                    jmap_buffer->set(jacs, y, x, lvl);
                    pId_buffer->set(ids, y, x, lvl);
                }
            }
        }
    }

    void renderDebugWindow(dataCPU<float> *image, window win, dataCPU<float> *buffer, int lvl)
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

            for (int y = pol_win.min_y; y < pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x < pol_win.max_x; x++)
                {
                    vec2<float> f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    bool isLine = f_pol.isEdge(f_pix);

                    float f_i = image->get(y, x, lvl);
                    f_i /= 255.0;

                    // z buffer
                    // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                    // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                    //    continue;

                    if (isLine)
                        buffer->set(1.0, y, x, lvl);
                    else
                        // buffer->set(1.0 / f_depth, y, x, lvl);
                        buffer->set(f_i, y, x, lvl);
                }
            }
        }
    }

    sceneType scene1;
    sceneType scene2;
    dataCPU<float> z_buffer;
    ThreadPool<RENDERER_NTHREADS> pool;
};
