#pragma once

#include <thread>

#include "params.h"
#include "common/types.h"
#include "common/camera.h"
#include "common/window.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/GeometryMesh.h"
#include "threadpoolCPU.h"

class renderCPU
{
public:
    renderCPU(unsigned int width, unsigned int height)
        : z_buffer(width, height, -1)
    {
    }

    void renderIdepthLineSearch(dataCPU<float> &kimage, dataCPU<float> &image, Sophus::SE3f imagePose, camera cam, dataCPU<float> &buffer, int lvl)
    {
        window win(0, cam.width - 1, 0, cam.height - 1);

        renderIdepthLineSearchWindow(kimage, image, imagePose, buffer, z_buffer.get(lvl), cam, win);
    }

    void renderRandom(camera cam, dataCPU<float> &buffer, float min = 0.1, float max = 1.9)
    {
        assert(cam.width == buffer.width && cam.height == buffer.height);

        window win(0, cam.width - 1, 0, cam.height - 1);

        renderRandomWindow(buffer, win, min, max);
    }

    void renderSmooth(camera cam, dataCPU<float> &buffer, float start = 1.0, float end = 2.0)
    {
        assert(cam.width == buffer.width && cam.height == buffer.height);

        window win(0, cam.width - 1, 0, cam.height - 1);

        renderSmoothWindow(buffer, cam, win, start, end);
    }

    void renderInterpolate(camera cam, dataCPU<float> &buffer)
    {
        window win(0, cam.width - 1, 0, cam.height - 1);
        //renderInterpolateWindow(cam, win, buffer);

        dataCPU<float> buffer2 = buffer;

        float nodata = buffer.getPercentNoData();
        while(nodata > 0.0)
        {
            renderInterpolateWindow(buffer, buffer2, cam, win);
            buffer = buffer2;
            nodata = buffer.getPercentNoData();
        }
    }

    void renderImageParallel(keyFrameCPU &kframe, SE3f localPose, dataMipMapCPU<float> &buffer, cameraMipMap cam, int lvl)
    {
        z_buffer.setToNoData(lvl);

        SE3f relativePose = localPose;

        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        scene1.project(cam[lvl]);
        scene2.transform(relativePose);
        scene2.project(cam[lvl]);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam[lvl].width / divi_x;
        int height = cam[lvl].height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderImageWindow(kframe.getRawImage(lvl), buffer.get(lvl), z_buffer.get(lvl), cam[lvl], win);
                // pool.enqueue(std::bind(&renderCPU::renderImageWindow, this, kimage, cam, win, buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderDebugParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<float> &buffer, cameraMipMap cam, int lvl)
    {
        z_buffer.setToNoData(lvl);

        SE3f relativePose = frame.getLocalPose();

        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();
    
        scene1.project(cam[lvl]);
        scene2.transform(relativePose);
        scene2.project(cam[lvl]);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam[lvl].width / divi_x;
        int height = cam[lvl].height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderDebugWindow(frame.getRawImage(lvl), buffer.get(lvl), win);
                // pool.enqueue(std::bind(&renderCPU::renderDebugWindow, this, image, win, buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJMapParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<jmapType> &jmap_buffer, dataMipMapCPU<float> &e_buffer, dataMipMapCPU<idsType> &pId_buffer, cameraMipMap cam, int lvl)
    {
        z_buffer.setToNoData(lvl);

        SE3f relativePose = frame.getLocalPose();
    
        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        scene1.project(cam[lvl]);
        scene2.transform(relativePose);
        scene2.project(cam[lvl]);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam[lvl].width / divi_x;
        int height = cam[lvl].height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderJMapWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getLocalExp(), frame.getdIdpixImage(lvl), relativePose, jmap_buffer.get(lvl), e_buffer.get(lvl), pId_buffer.get(lvl), z_buffer.get(lvl), cam[lvl], win);
                // pool.enqueue(std::bind(&renderCPU::renderJMapWindow, this, kimage, frame, cam, win, jmap_buffer, e_buffer, pId_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJPoseMapParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<vec6f> &jpose_buffer, dataMipMapCPU<jmapType> &jmap_buffer, dataMipMapCPU<float> &e_buffer, dataMipMapCPU<idsType> &pId_buffer, cameraMipMap cam, int lvl)
    {
        z_buffer.setToNoData(lvl);

        SE3f relativePose = frame.getLocalPose();

        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        scene1.project(cam[lvl]);
        scene2.transform(relativePose);
        scene2.project(cam[lvl]);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam[lvl].width / divi_x;
        int height = cam[lvl].height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderJPoseMapWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getLocalExp(), frame.getdIdpixImage(lvl), relativePose, jpose_buffer.get(lvl), jmap_buffer.get(lvl), e_buffer.get(lvl), pId_buffer.get(lvl), z_buffer.get(lvl), cam[lvl], win);
                // pool.enqueue(std::bind(&renderCPU::renderJPoseMapWindow, this, kimage, frame, cam, win, jpose_buffer, jmap_buffer, e_buffer, pId_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJExpParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<vec2f> &jexp_buffer, dataMipMapCPU<float> &e_buffer, cameraMipMap cam, int lvl)
    {
        z_buffer.setToNoData(lvl);

        SE3f relativePose = frame.getLocalPose();

        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        scene1.project(cam[lvl]);
        scene2.transform(relativePose);
        scene2.project(cam[lvl]);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam[lvl].width / divi_x;
        int height = cam[lvl].height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderJExpWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getLocalExp(), jexp_buffer.get(lvl), e_buffer.get(lvl), z_buffer.get(lvl), cam[lvl], win);
                // pool.enqueue(std::bind(&renderCPU::renderJLightAffineWindow, this, kimage, frame, cam, win, jlightaffine_buffer, e_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJPoseParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<vec6f> &jpose_buffer, dataMipMapCPU<float> &e_buffer, cameraMipMap cam, int lvl)
    {
        z_buffer.setToNoData(lvl);

        SE3f relativePose = frame.getLocalPose();

        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        scene1.project(cam[lvl]);
        scene2.transform(relativePose);
        scene2.project(cam[lvl]);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam[lvl].width / divi_x;
        int height = cam[lvl].height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderJPoseWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getLocalExp(), frame.getdIdpixImage(lvl), jpose_buffer.get(lvl), e_buffer.get(lvl), z_buffer.get(lvl), cam[lvl], win);
                // pool.enqueue(std::bind(&renderCPU::renderJPoseWindow, this, kimage, frame, cam, win, jpose_buffer, e_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderResidualParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<float> &e_buffer, cameraMipMap cam, int lvl)
    {
        z_buffer.setToNoData(lvl);

        SE3f relativePose = frame.getLocalPose();

        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        scene1.project(cam[lvl]);
        scene2.transform(relativePose);
        scene2.project(cam[lvl]);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam[lvl].width / divi_x;
        int height = cam[lvl].height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderResidualWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getLocalExp(), e_buffer.get(lvl), z_buffer.get(lvl), cam[lvl], win);
                // pool.enqueue(std::bind(&renderCPU::renderResidualWindow, this, kimage, frame, cam, win, e_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderIdepthParallel(keyFrameCPU &kframe, SE3f localPose, dataMipMapCPU<float> &buffer, cameraMipMap cam, int lvl)
    {
        z_buffer.setToNoData(lvl);

        SE3f relativePose = localPose;

        scene2 = kframe.getGeometry();

        scene2.transform(relativePose);
        scene2.project(cam[lvl]);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam[lvl].width / divi_x;
        int height = cam[lvl].height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderIdepthWindow(buffer.get(lvl), z_buffer.get(lvl), win);
                // pool.enqueue(std::bind(&renderCPU::renderIdepthWindow, this, win, buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderWeightParallel(keyFrameCPU &kframe, SE3f localPose, dataMipMapCPU<float> &buffer, cameraMipMap cam, int lvl)
    {
        z_buffer.setToNoData(lvl);

        SE3f relativePose = localPose;

        scene2 = kframe.getGeometry();

        scene2.transform(relativePose);
        scene2.project(cam[lvl]);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        int width = cam[lvl].width / divi_x;
        int height = cam[lvl].height / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                int min_x = tx * width;
                int max_x = (tx + 1) * width;
                int min_y = ty * height;
                int max_y = (ty + 1) * height;

                window win(min_x, max_x - 1, min_y, max_y - 1);

                renderWeightWindow(buffer.get(lvl), z_buffer.get(lvl), win);
                // pool.enqueue(std::bind(&renderCPU::renderWeightWindow, this, win, buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

private:
    void renderSmoothWindow(dataCPU<float> &buffer, camera cam, window win, float start = 1.0, float end = 2.0)
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

    void renderRandomWindow(dataCPU<float> &buffer, window win, float min = 1.0, float max = 2.0)
    {
        for (int y = win.min_y; y <= win.max_y; y++)
        {
            for (int x = win.min_x; x <= win.max_x; x++)
            {
                float val = (max - min) * float(rand() % 1000) / 1000.0 + min;
                buffer.set(val, y, x);
            }
        }
    }

    void renderInterpolateWindow(dataCPU<float> &src_buffer, dataCPU<float> &dst_buffer, camera cam, window win)
    {
        for (int y = win.min_y; y <= win.max_y; y++)
        {
            for (int x = win.min_x; x <= win.max_x; x++)
            {
                if (src_buffer.get(y, x) == src_buffer.nodata)
                {
                    int size = 1;
                    float acc = 0.0;
                    int count = 0;

                    for (int y_ = y - size; y_ <= y + size; y_ += 1)
                    {
                        for (int x_ = x - size; x_ <= x + size; x_ += 1)
                        {
                            if (!cam.isPixVisible(x_, y_))
                                continue;

                            auto val = src_buffer.get(y_, x_);
                            if (val == src_buffer.nodata)
                                continue;

                            acc += val;
                            count += 1;
                        }
                    }
                    if (count > 0)
                    {
                        dst_buffer.set(acc / count, y, x);
                    }
                }
            }
        }
    }

    void renderIdepthLineSearchWindow(dataCPU<float> &kimage, dataCPU<float> &image, SE3f imagePose, dataCPU<float> &buffer, dataCPU<float> &z_buffer, camera cam, window win)
    {
        SE3f kfTofPose = imagePose; // * kframe->getPose().inverse();
        SE3f fTokfPose = kfTofPose.inverse();

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

                vec2f f_pix(x, y);
                // if (!cam.isPixVisible(f_pix))
                //     continue;

                vec3f f_ray = cam.pixToRay(x, y);

                float depth_min = 0.1;
                float depth_max = 10.0;

                vec3f f_ver_min = f_ray * depth_min;
                vec3f kf_ver_min = fTokfPose * f_ver_min;
                vec3f kf_ray_min = kf_ver_min / kf_ver_min(2);
                vec2f kf_pix_min = cam.rayToPix(kf_ray_min);

                vec3f f_ver_max = f_ray * depth_max;
                vec3f kf_ver_max = fTokfPose * f_ver_max;
                vec3f kf_ray_max = kf_ver_max / kf_ver_max(2);
                vec2f kf_pix_max = cam.rayToPix(kf_ray_max);

                vec2f kf_pix_diff = kf_pix_max - kf_pix_min;
                float kf_pix_diff_norm = kf_pix_diff.norm();
                kf_pix_diff = kf_pix_diff / kf_pix_diff_norm;

                for (int i = 0; i < kf_pix_diff_norm; i++)
                {
                    vec2f kf_pix = kf_pix_min + kf_pix_diff * i;

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

                        vec3f kf_ray = cam.pixToRay(kf_pix);
                        vec3f Y = fTokfPose * f_ray;
                        vec3f C = fTokfPose.translation();

                        float f_depth = (C(1) / kf_ray(1) - C(2)) / (Y(2) - Y(1) / kf_ray(1));
                        // float f_depth = depth_min + i * (depth_max - depth_min) / kf_pix_diff_norm;
                        // float f_idepth = 1.0 / f_depth;

                        z_buffer.set(f_depth, y, x);
                    }
                }
            }
        }
    }

    void renderImageWindow(dataCPU<imageType> &image, dataCPU<float> &buffer, dataCPU<float> &z_buffer, camera cam, window win)
    {
        std::vector<int> ids = scene2.getShapesIds();

        for (auto t_id : ids)
        {
            shapeType f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() < 0.0)
                continue;

            shapeType kf_pol = scene1.getShape(t_id);

            window pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            for (int y = pol_win.min_y; y <= pol_win.max_y; y++)
            {
                for (int x = pol_win.min_x; x <= pol_win.max_x; x++)
                {
                    vec2f f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    float z_depth = z_buffer.get(y, x);
                    if (z_depth < f_depth && z_depth != z_buffer.nodata)
                        continue;

                    vec2f kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = image.get(kf_pix(1), kf_pix(0));
                    if (kf_i == image.nodata)
                        continue;

                    // buffer->set(std::exp(kf_a) * (kf_i - kf_b), y, x, lvl);
                    buffer.set(kf_i, y, x);
                    z_buffer.set(f_depth, y, x);
                }
            }
        }
    }

    void renderIdepthWindow(dataCPU<float> &buffer, dataCPU<float> &z_buffer, window win)
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
                    vec2f f_pix(x, y);

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

    void renderWeightWindow(dataCPU<float> &buffer, dataCPU<float> &z_buffer, window win)
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
                    vec2f f_pix(x, y);

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

    void renderResidualWindow(dataCPU<float> &kimage, dataCPU<float> &image, vec2f imageExp, dataCPU<float> &e_buffer, dataCPU<float> &z_buffer, camera cam, window win)
    {
        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

        std::vector<int> t_ids = scene2.getShapesIds();

        float alpha = std::exp(-imageExp(0));
        float beta = imageExp(1);

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
                    vec2f f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_depth = z_buffer.get(y, x);
                    if (l_depth < f_depth && l_depth != z_buffer.nodata)
                        continue;

                    vec2f kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    auto f_i = image.get(y, x);

                    if (kf_i == kimage.nodata || f_i == image.nodata)
                        continue;

                    float f_i_cor = alpha * (f_i - beta);

                    float residual = (f_i_cor - kf_i);

                    e_buffer.set(residual, y, x);
                    z_buffer.set(f_depth, y, x);
                }
            }
        }
    }

    void renderJPoseWindow(dataCPU<float> &kimage, dataCPU<float> &image, vec2f imageAffine, dataCPU<vec2f> &d_image_d_pix, dataCPU<vec6f> &jpose_buffer, dataCPU<float> &e_buffer, dataCPU<float> &z_buffer, camera cam, window win)
    {
        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

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
                    vec2f f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_depth = z_buffer.get(y, x);
                    if (l_depth < f_depth && l_depth != z_buffer.nodata)
                        continue;

                    vec2f kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    vec3f f_ray = cam.pixToRay(x, y);
                    vec3f f_ver = f_ray * f_depth;
                    // vec3<float> kf_ray = cam.pixToRay(kf_pix);

                    auto kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    auto f_i = image.get(y, x);
                    vec2f d_f_i_d_pix = d_image_d_pix.get(y, x);

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata)
                        continue;

                    float v0 = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    float v1 = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    float v2 = -(v0 * f_ver(0) + v1 * f_ver(1)) / f_ver(2);
                    vec3f d_f_i_d_tra(v0, v1, v2);
                    vec3f d_f_i_d_rot(-f_ver(2) * v1 + f_ver(1) * v2, f_ver(2) * v0 - f_ver(0) * v2, -f_ver(1) * v0 + f_ver(0) * v1);

                    float f_i_cor = alpha * (f_i - beta);

                    vec6f j_pose = {d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2)};

                    float residual = (f_i_cor - kf_i);

                    jpose_buffer.set(j_pose, y, x);
                    e_buffer.set(residual, y, x);
                    z_buffer.set(f_depth, y, x);
                }
            }
        }
    }

    void renderJPoseExpWindow(dataCPU<float> &kimage, dataCPU<float> &image, vec2f imageExp, dataCPU<vec2f> &d_image_d_pix, dataCPU<vec6f> &jpose_buffer, dataCPU<vec2f> &jexp_buffer, dataCPU<float> &e_buffer, dataCPU<float> &z_buffer, camera cam, window win)
    {
        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

        float alpha = std::exp(-imageExp(0));
        float beta = imageExp(1);

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
                    vec2f f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_depth = z_buffer.get(y, x);
                    if (l_depth < f_depth && l_depth != z_buffer.nodata)
                        continue;

                    vec2f kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    vec3f f_ray = cam.pixToRay(x, y);
                    vec3f f_ver = f_ray * f_depth;
                    // vec3<float> kf_ray = cam.pixToRay(kf_pix);

                    auto kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    auto f_i = image.get(y, x);
                    vec2f d_f_i_d_pix = d_image_d_pix.get(y, x);

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata)
                        continue;

                    float v0 = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    float v1 = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    float v2 = -(v0 * f_ver(0) + v1 * f_ver(1)) / f_ver(2);
                    vec3f d_f_i_d_tra(v0, v1, v2);
                    vec3f d_f_i_d_rot(-f_ver(2) * v1 + f_ver(1) * v2, f_ver(2) * v0 - f_ver(0) * v2, -f_ver(1) * v0 + f_ver(0) * v1);

                    float f_i_cor = alpha * (f_i - beta);

                    vec2f d_f_i_d_exp(-f_i_cor, -alpha);

                    vec6f j_pose = {d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2)};
                    vec2f j_exp = {d_f_i_d_exp(0), d_f_i_d_exp(1)};

                    float residual = (f_i_cor - kf_i);

                    jpose_buffer.set(j_pose, y, x);
                    jexp_buffer.set(j_exp, y, x);
                    e_buffer.set(residual, y, x);
                    z_buffer.set(f_depth, y, x);
                }
            }
        }
    }

    void renderJExpWindow(dataCPU<float> &kimage, dataCPU<float> &image, vec2f imageExp, dataCPU<vec2f> &jexp_buffer, dataCPU<float> &e_buffer, dataCPU<float> &z_buffer, camera cam, window win)
    {
        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

        std::vector<int> t_ids = scene2.getShapesIds();

        float alpha = std::exp(-imageExp(0));
        float beta = imageExp(1);

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
                    vec2f f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_depth = z_buffer.get(y, x);
                    if (l_depth < f_depth && l_depth != z_buffer.nodata)
                        continue;

                    vec2f kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    auto kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    auto f_i = image.get(y, x);

                    if (kf_i == kimage.nodata || f_i == image.nodata)
                        continue;

                    float f_i_cor = alpha * (f_i - beta);

                    vec2f jexp(-f_i_cor, -alpha);

                    float residual = f_i_cor - kf_i;

                    jexp_buffer.set(jexp, y, x);
                    e_buffer.set(residual, y, x);
                    z_buffer.set(f_depth, y, x);
                }
            }
        }
    }

    void renderJMapWindow(dataCPU<float> &kimage, dataCPU<float> &image, vec2f imageExp, dataCPU<vec2f> &d_image_d_pix, SE3f imagePose, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &e_buffer, dataCPU<idsType> &pId_buffer, dataCPU<float> &z_buffer, camera cam, window win)
    {
        float min_area = 0.0; //(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3 / 4;

        SE3f kfTofPose = imagePose;
        // Sophus::SE3f fTokfPose = kfTofPose.inverse();

        float alpha = std::exp(-imageExp(0));
        float beta = imageExp(1);

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
                    vec2f f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    float f_depth = f_pol.getDepth(f_pix);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_idepth = z_buffer.get(y, x);
                    if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                        continue;

                    vec2f kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    vec3f f_ray = cam.pixToRay(f_pix);
                    vec3f f_ver = f_ray * f_depth;
                    vec3f kf_ray = cam.pixToRay(kf_pix);

                    imageType kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    imageType f_i = image.get(y, x);
                    vec2f d_f_i_d_pix = d_image_d_pix.get(y, x);

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata)
                        continue;

                    float f_i_cor = alpha * (f_i - beta);

                    float residual = f_i_cor - kf_i;

                    vec3f d_f_i_d_f_ver;
                    d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                    vec3f d_f_ver_d_kf_depth = kfTofPose.rotationMatrix() * kf_ray;

                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    // this could be the jacobian of the depth of the 3 vertices in a triangle
                    // or the jacobian of the normal + depth of a surfel
                    jmapType jacs = kf_pol.getParamJacobian(kf_pix) * d_f_i_d_kf_depth;

                    e_buffer.set(residual, y, x);
                    jmap_buffer.set(jacs, y, x);
                    pId_buffer.set(ids, y, x);
                }
            }
        }
    }

    void renderJPoseMapWindow(dataCPU<float> &kimage, dataCPU<float> &image, vec2f imageExp, dataCPU<vec2f> &d_image_d_pix, SE3f imagePose, dataCPU<vec6f> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &e_buffer, dataCPU<idsType> &pId_buffer, dataCPU<float> &z_buffer, camera cam, window win)
    {
        float min_area = 0.0; //(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3.0 / 4.0;

        SE3f kfTofPose = imagePose;
        // Sophus::SE3f fTokfPose = kfTofPose.inverse();

        float alpha = std::exp(-imageExp(0));
        float beta = imageExp(1);

        std::vector<int> t_ids = scene2.getShapesIds();
        // int shapeDoF = scene2->getShapesDoF();

        for (auto t_id : t_ids)
        {
            shapeType f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            float f_pol_area = f_pol.getScreenArea();
            if (f_pol_area <= min_area)
                continue;

            shapeType kf_pol = scene1.getShape(t_id);

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
                    vec2f f_pix(x, y);

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

                    vec2f kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    imageType f_i = image.get(y, x);
                    vec2f d_f_i_d_pix = d_image_d_pix.get(y, x);

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata)
                        continue;

                    vec3f f_ray = cam.pixToRay(f_pix);
                    vec3f f_ver = f_ray * f_depth;
                    vec3f kf_ray = cam.pixToRay(kf_pix);

                    vec3f d_f_i_d_f_ver;
                    d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                    vec3f d_f_i_d_tra(d_f_i_d_f_ver(0), d_f_i_d_f_ver(1), d_f_i_d_f_ver(2));
                    vec3f d_f_i_d_rot(-f_ver(2) * d_f_i_d_f_ver(1) + f_ver(1) * d_f_i_d_f_ver(2), f_ver(2) * d_f_i_d_f_ver(0) - f_ver(0) * d_f_i_d_f_ver(2), -f_ver(1) * d_f_i_d_f_ver(0) + f_ver(0) * d_f_i_d_f_ver(1));

                    float f_i_cor = alpha * (f_i - beta);

                    vec6f jpose = {d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2)};

                    vec3f d_f_ver_d_kf_depth = kfTofPose.rotationMatrix() * kf_ray;

                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    jmapType jacs = kf_pol.getParamJacobian(kf_pix) * d_f_i_d_kf_depth;

                    float error = f_i_cor - kf_i;

                    e_buffer.set(error, y, x);
                    jpose_buffer.set(jpose, y, x);
                    jmap_buffer.set(jacs, y, x);
                    pId_buffer.set(ids, y, x);
                }
            }
        }
    }

    void renderJPoseExpMapWindow(dataCPU<imageType> &kimage, dataCPU<imageType> &image, vec2f imageExp, dataCPU<vec2f> &d_image_d_pix, SE3f imagePose, dataCPU<vec6f> &jpose_buffer, dataCPU<vec2f> &jexp_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &e_buffer, dataCPU<idsType> &pId_buffer, dataCPU<float> &z_buffer, camera cam, window win)
    {
        float min_area = 0.0; //(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3.0 / 4.0;

        SE3f kfTofPose = imagePose;
        // Sophus::SE3f fTokfPose = kfTofPose.inverse();

        float alpha = std::exp(-imageExp(0));
        float beta = imageExp(1);

        std::vector<int> t_ids = scene2.getShapesIds();
        // int shapeDoF = scene2->getShapesDoF();

        for (auto t_id : t_ids)
        {
            shapeType f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            float f_pol_area = f_pol.getScreenArea();
            if (f_pol_area <= min_area)
                continue;

            shapeType kf_pol = scene1.getShape(t_id);

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
                    vec2f f_pix(x, y);

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

                    vec2f kf_pix = f_pol.getPix(f_pix, kf_pol);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    imageType f_i = image.get(y, x);
                    vec2f d_f_i_d_pix = d_image_d_pix.get(y, x);

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata)
                        continue;

                    vec3f f_ray = cam.pixToRay(f_pix);
                    vec3f f_ver = f_ray * f_depth;
                    vec3f kf_ray = cam.pixToRay(kf_pix);

                    vec3f d_f_i_d_f_ver;
                    d_f_i_d_f_ver(0) = d_f_i_d_pix(0) * cam.fx / f_ver(2);
                    d_f_i_d_f_ver(1) = d_f_i_d_pix(1) * cam.fy / f_ver(2);
                    d_f_i_d_f_ver(2) = -(d_f_i_d_f_ver(0) * f_ver(0) + d_f_i_d_f_ver(1) * f_ver(1)) / f_ver(2);

                    vec3f d_f_i_d_tra(d_f_i_d_f_ver(0), d_f_i_d_f_ver(1), d_f_i_d_f_ver(2));
                    vec3f d_f_i_d_rot(-f_ver(2) * d_f_i_d_f_ver(1) + f_ver(1) * d_f_i_d_f_ver(2), f_ver(2) * d_f_i_d_f_ver(0) - f_ver(0) * d_f_i_d_f_ver(2), -f_ver(1) * d_f_i_d_f_ver(0) + f_ver(0) * d_f_i_d_f_ver(1));

                    float f_i_cor = alpha * (f_i - beta);

                    vec2f d_f_i_d_f_exp(-f_i_cor, -alpha);

                    vec6f jpose = {d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2)};
                    vec2f jexp = {d_f_i_d_f_exp(0), d_f_i_d_f_exp(1)};

                    vec3f d_f_ver_d_kf_depth = kfTofPose.rotationMatrix() * kf_ray;

                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    jmapType jacs = kf_pol.getParamJacobian(kf_pix) * d_f_i_d_kf_depth;

                    float error = f_i_cor - kf_i;

                    e_buffer.set(error, y, x);
                    jpose_buffer.set(jpose, y, x);
                    jexp_buffer.set(jexp, y, x);
                    jmap_buffer.set(jacs, y, x);
                    pId_buffer.set(ids, y, x);
                }
            }
        }
    }

    void renderDebugWindow(dataCPU<imageType> &image, dataCPU<imageType> &buffer, window win)
    {
        std::vector<int> ids = scene2.getShapesIds();

        for (auto t_id : ids)
        {
            shapeType f_pol = scene2.getShape(t_id);

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
                    vec2f f_pix(x, y);

                    if (!f_pol.isPixInShape(f_pix))
                        continue;

                    // float f_depth = f_pol.getDepth(f_pix);
                    bool isLine = f_pol.isEdge(f_pix);

                    imageType f_i = image.get(y, x);
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

    geometryType scene1;
    geometryType scene2;
    dataMipMapCPU<float> z_buffer;
    ThreadPool<RENDERER_NTHREADS> pool;
};
