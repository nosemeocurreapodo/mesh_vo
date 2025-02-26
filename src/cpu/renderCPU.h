#pragma once

#include <thread>

#include "params.h"
#include "common/types.h"
#include "common/camera.h"
#include "common/window.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/keyFrameCPU.h"
#include "cpu/GeometryMesh.h"
#include "threadpoolCPU.h"

class renderCPU
{
public:
    renderCPU(unsigned int width, unsigned int height)
        : z_buffer(width, height, -1)
    {
    }

    void renderIdepthLineSearch(keyFrameCPU &kframe, frameCPU &frame, cameraType cam, int lvl)
    {
        SE3f relativePose = frame.getLocalPose();

        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        float width = 1.0 / divi_x;
        float height = 1.0 / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                float min_x = tx * width;
                float max_x = (tx + 1) * width;
                float min_y = ty * height;
                float max_y = (ty + 1) * height;

                window<float> win(min_x, max_x, min_y, max_y);

                renderIdepthLineSearchWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getLocalExp(), relativePose, cam, win);
                // pool.enqueue(std::bind(&renderCPU::renderJMapWindow, this, kimage, frame, cam, win, jmap_buffer, e_buffer, pId_buffer, lvl));
            }
        }

        // pool.waitUntilDone();

        kframe.setGeometry(scene2);
    }

    void renderDepthFromClosestShape(keyFrameCPU &kframe, cameraType cam, int lvl)
    {
        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        float width = 1.0 / divi_x;
        float height = 1.0 / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                float min_x = tx * width;
                float max_x = (tx + 1) * width;
                float min_y = ty * height;
                float max_y = (ty + 1) * height;

                window<float> win(min_x, max_x, min_y, max_y);

                renderDepthFromClosestShapeWindow(cam, win);
                // pool.enqueue(std::bind(&renderCPU::renderJMapWindow, this, kimage, frame, cam, win, jmap_buffer, e_buffer, pId_buffer, lvl));
            }
        }

        // pool.waitUntilDone();

        kframe.setGeometry(scene2);
    }

    void renderRandom(dataCPU<float> &buffer, float min = 0.1, float max = 1.9)
    {
        window<int> win(0, buffer.width - 1, 0, buffer.height - 1);
        renderRandomWindow(buffer, win, min, max);
    }

    void renderVerticallySmooth(dataCPU<float> &buffer, float start = 1.0, float end = 2.0)
    {
        window<int> win(0, buffer.width - 1, 0, buffer.height - 1);
        renderVerticallySmoothWindow(buffer, win, start, end);
    }

    void renderInterpolate(dataCPU<float> &buffer)
    {
        window<int> win(0, buffer.width - 1, 0, buffer.height - 1);
        // renderInterpolateWindow(cam, win, buffer);

        dataCPU<float> buffer2 = buffer;

        float nodata = buffer.getPercentNoData();
        while (nodata > 0.0)
        {
            renderInterpolateWindow(buffer, buffer2, win);
            buffer = buffer2;
            nodata = buffer.getPercentNoData();
        }
    }

    void renderImageParallel(keyFrameCPU &kframe, SE3f localPose, dataMipMapCPU<float> &buffer, cameraType cam, int lvl)
    {
        z_buffer.setToNoData();

        SE3f kfTofPose = localPose;
        SE3f fTokfPose = kfTofPose.inverse();

        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        scene1.project(cam);
        scene2.transform(kfTofPose);
        scene2.project(cam);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        float width = 1.0 / divi_x;
        float height = 1.0 / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                float min_x = tx * width;
                float max_x = (tx + 1) * width;
                float min_y = ty * height;
                float max_y = (ty + 1) * height;

                window<float> win(min_x, max_x, min_y, max_y);

                renderImageWindow(kframe.getRawImage(lvl), kfTofPose, fTokfPose, buffer.get(lvl), cam, win);
                // pool.enqueue(std::bind(&renderCPU::renderImageWindow, this, kimage, cam, win, buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderDebugParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<float> &buffer, cameraType cam, int lvl)
    {
        z_buffer.setToNoData();

        SE3f kfTofPose = frame.getLocalPose();
        SE3f fTokfPose = kfTofPose.inverse();

        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        scene1.project(cam);
        scene2.transform(kfTofPose);
        scene2.project(cam);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        float width = 1.0 / divi_x;
        float height = 1.0 / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                float min_x = tx * width;
                float max_x = (tx + 1) * width;
                float min_y = ty * height;
                float max_y = (ty + 1) * height;

                window<float> win(min_x, max_x, min_y, max_y);

                renderDebugWindow(frame.getRawImage(lvl), buffer.get(lvl), win);
                // pool.enqueue(std::bind(&renderCPU::renderDebugWindow, this, image, win, buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJMapParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<jmapType> &jmap_buffer, dataMipMapCPU<float> &e_buffer, dataMipMapCPU<idsType> &pId_buffer, cameraType cam, int lvl)
    {
        z_buffer.setToNoData();

        SE3f kfTofPose = frame.getLocalPose();
        SE3f fTokfPose = kfTofPose.inverse();

        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        scene1.project(cam);
        scene2.transform(kfTofPose);
        scene2.project(cam);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        float width = 1.0 / divi_x;
        float height = 1.0 / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                float min_x = tx * width;
                float max_x = (tx + 1) * width;
                float min_y = ty * height;
                float max_y = (ty + 1) * height;

                window<float> win(min_x, max_x, min_y, max_y);

                renderJMapWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getLocalExp(), frame.getdIdpixImage(lvl), kfTofPose, fTokfPose, jmap_buffer.get(lvl), e_buffer.get(lvl), pId_buffer.get(lvl), cam, win);
                // pool.enqueue(std::bind(&renderCPU::renderJMapWindow, this, kimage, frame, cam, win, jmap_buffer, e_buffer, pId_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJPoseMapParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<vec6f> &jpose_buffer, dataMipMapCPU<jmapType> &jmap_buffer, dataMipMapCPU<float> &e_buffer, dataMipMapCPU<idsType> &pId_buffer, cameraType cam, int lvl)
    {
        z_buffer.setToNoData();

        SE3f kfTofPose = frame.getLocalPose();
        SE3f fTokfPose = kfTofPose.inverse();

        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        scene1.project(cam);
        scene2.transform(kfTofPose);
        scene2.project(cam);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        float width = 1.0 / divi_x;
        float height = 1.0 / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                float min_x = tx * width;
                float max_x = (tx + 1) * width;
                float min_y = ty * height;
                float max_y = (ty + 1) * height;

                window<float> win(min_x, max_x, min_y, max_y);

                renderJPoseMapWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getLocalExp(), frame.getdIdpixImage(lvl), kfTofPose, fTokfPose, jpose_buffer.get(lvl), jmap_buffer.get(lvl), e_buffer.get(lvl), pId_buffer.get(lvl), cam, win);
                // pool.enqueue(std::bind(&renderCPU::renderJPoseMapWindow, this, kimage, frame, cam, win, jpose_buffer, jmap_buffer, e_buffer, pId_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJExpParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<vec2f> &jexp_buffer, dataMipMapCPU<float> &e_buffer, cameraType cam, int lvl)
    {
        z_buffer.setToNoData();

        SE3f kfTofPose = frame.getLocalPose();
        SE3f fTokfPose = kfTofPose.inverse();

        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        scene1.project(cam);
        scene2.transform(kfTofPose);
        scene2.project(cam);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        float width = 1.0 / divi_x;
        float height = 1.0 / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                float min_x = tx * width;
                float max_x = (tx + 1) * width;
                float min_y = ty * height;
                float max_y = (ty + 1) * height;

                window win(min_x, max_x, min_y, max_y);

                renderJExpWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getLocalExp(), fTokfPose, jexp_buffer.get(lvl), e_buffer.get(lvl), cam, win);
                // pool.enqueue(std::bind(&renderCPU::renderJLightAffineWindow, this, kimage, frame, cam, win, jlightaffine_buffer, e_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJIntrinsicParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<cameraParamType> &jintrinsic_buffer, dataMipMapCPU<float> &e_buffer, cameraType cam, int lvl)
    {
        z_buffer.setToNoData();

        SE3f kfTofPose = frame.getLocalPose();
        SE3f fTokfPose = kfTofPose.inverse();

        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        scene1.project(cam);
        scene2.transform(kfTofPose);
        scene2.project(cam);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        float width = 1.0 / divi_x;
        float height = 1.0 / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                float min_x = tx * width;
                float max_x = (tx + 1) * width;
                float min_y = ty * height;
                float max_y = (ty + 1) * height;

                window win(min_x, max_x, min_y, max_y);

                renderJIntrinsicWindow(kframe.getRawImage(lvl), kframe.getdIdpixImage(lvl), frame.getRawImage(lvl), frame.getLocalExp(), fTokfPose, frame.getdIdpixImage(lvl), frame.getLocalPose(), jintrinsic_buffer.get(lvl), e_buffer.get(lvl), cam, win);
                // pool.enqueue(std::bind(&renderCPU::renderJPoseWindow, this, kimage, frame, cam, win, jpose_buffer, e_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJPoseParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<vec6f> &jpose_buffer, dataMipMapCPU<float> &e_buffer, cameraType cam, int lvl)
    {
        z_buffer.setToNoData();

        SE3f kfTofPose = frame.getLocalPose();
        SE3f fTokfPose = kfTofPose.inverse();

        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        scene1.project(cam);
        scene2.transform(kfTofPose);
        scene2.project(cam);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        float width = 1.0 / divi_x;
        float height = 1.0 / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                float min_x = tx * width;
                float max_x = (tx + 1) * width;
                float min_y = ty * height;
                float max_y = (ty + 1) * height;

                window<float> win(min_x, max_x, min_y, max_y);

                renderJPoseWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getLocalExp(), fTokfPose, frame.getdIdpixImage(lvl), jpose_buffer.get(lvl), e_buffer.get(lvl), cam, win);
                // pool.enqueue(std::bind(&renderCPU::renderJPoseWindow, this, kimage, frame, cam, win, jpose_buffer, e_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderResidualParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<float> &e_buffer, cameraType cam, int lvl)
    {
        z_buffer.setToNoData();

        SE3f kfTofPose = frame.getLocalPose();
        SE3f fTokfPose = kfTofPose.inverse();

        scene1 = kframe.getGeometry();
        scene2 = kframe.getGeometry();

        scene1.project(cam);
        scene2.transform(kfTofPose);
        scene2.project(cam);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        float width = 1.0 / divi_x;
        float height = 1.0 / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                float min_x = tx * width;
                float max_x = (tx + 1) * width;
                float min_y = ty * height;
                float max_y = (ty + 1) * height;

                window win(min_x, max_x, min_y, max_y);

                renderResidualWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getLocalExp(), fTokfPose, e_buffer.get(lvl), cam, win);
                // pool.enqueue(std::bind(&renderCPU::renderResidualWindow, this, kimage, frame, cam, win, e_buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderDepthParallel(keyFrameCPU &kframe, SE3f localPose, dataMipMapCPU<float> &buffer, cameraType cam, int lvl)
    {
        z_buffer.setToNoData();

        SE3f kfTofPose = localPose;

        scene2 = kframe.getGeometry();

        scene2.transform(kfTofPose);
        scene2.project(cam);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        float width = 1.0 / divi_x;
        float height = 1.0 / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                float min_x = tx * width;
                float max_x = (tx + 1) * width;
                float min_y = ty * height;
                float max_y = (ty + 1) * height;

                window<float> win(min_x, max_x, min_y, max_y);

                renderDepthWindow(buffer.get(lvl), win);
                // pool.enqueue(std::bind(&renderCPU::renderIdepthWindow, this, win, buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

    void renderWeightParallel(keyFrameCPU &kframe, SE3f localPose, dataMipMapCPU<float> &buffer, cameraType cam, int lvl)
    {
        z_buffer.setToNoData();

        SE3f kfTofPose = localPose;

        scene2 = kframe.getGeometry();

        scene2.transform(kfTofPose);
        scene2.project(cam);

        int divi_y = pool.getNumThreads();
        int divi_x = 1;

        float width = 1.0 / divi_x;
        float height = 1.0 / divi_y;

        for (int ty = 0; ty < divi_y; ty++)
        {
            for (int tx = 0; tx < divi_x; tx++)
            {
                float min_x = tx * width;
                float max_x = (tx + 1) * width;
                float min_y = ty * height;
                float max_y = (ty + 1) * height;

                window win(min_x, max_x, min_y, max_y);

                renderWeightWindow(buffer.get(lvl), win);
                // pool.enqueue(std::bind(&renderCPU::renderWeightWindow, this, win, buffer, lvl));
            }
        }

        // pool.waitUntilDone();
    }

private:
    void renderVerticallySmoothWindow(dataCPU<float> &buffer, window<int> win, float start = 1.0, float end = 2.0)
    {
        for (int y = win.min_y; y <= win.max_y; y++)
        {
            float val = start + (end - start) * float(y) / (buffer.height - 1);
            assert(val > 0);
            for (int x = win.min_x; x <= win.max_x; x++)
            {
                buffer.setTexel(val, y, x);
            }
        }
    }

    void renderRandomWindow(dataCPU<float> &buffer, window<int> win, float min = 1.0, float max = 2.0)
    {
        for (int y = win.min_y; y <= win.max_y; y++)
        {
            for (int x = win.min_x; x <= win.max_x; x++)
            {
                float val = (max - min) * float(rand() % 1000) / 1000.0 + min;
                buffer.setTexel(val, y, x);
            }
        }
    }

    void renderInterpolateWindow(dataCPU<float> &src_buffer, dataCPU<float> &dst_buffer, window<int> win)
    {
        for (int y = win.min_y; y <= win.max_y; y++)
        {
            for (int x = win.min_x; x <= win.max_x; x++)
            {
                if (src_buffer.getTexel(y, x) == src_buffer.nodata)
                {
                    int size = 1;
                    float acc = 0.0;
                    int count = 0;

                    for (int y_ = y - size; y_ <= y + size; y_ += 1)
                    {
                        for (int x_ = x - size; x_ <= x + size; x_ += 1)
                        {
                            if (!win.isPixInWindow(x_, y_))
                                continue;

                            auto val = src_buffer.getTexel(y_, x_);
                            if (val == src_buffer.nodata)
                                continue;

                            acc += val;
                            count += 1;
                        }
                    }
                    if (count > 0)
                    {
                        dst_buffer.setTexel(acc / count, y, x);
                    }
                }
            }
        }
    }

    void renderIdepthLineSearchWindow(dataCPU<float> &kimage, dataCPU<float> &image, vec2f imageAffine, SE3f imagePose, cameraType cam, window<float> win)
    {
        float min_idepth = 1.0 / 1.5;
        float max_idepth = 1.0 / 0.5;
        float step_idepth = 0.01;

        float win_size = 2;

        float alpha = std::exp(-imageAffine(0));
        float beta = imageAffine(1);

        SE3f kfToPose = imagePose;

        std::vector<int> t_ids = scene1.getParamIds();

        for (int t_id : t_ids)
        {
            vertex vert = scene1.getVertex(t_id);

            if (vert.weight > 1.0 / mesh_vo::mapping_param_initial_var)
                continue;

            vec3f kf_ray = vert.ray;
            vec2f kf_pix = cam.rayToPix(kf_ray);

            if (!win.isPixInWindow(kf_pix))
                continue;

            float best_residual = 100000000000.0;
            for (float kf_idepth = min_idepth; kf_idepth < max_idepth; kf_idepth += step_idepth)
            {
                vec3f kf_ver = kf_ray / kf_idepth;

                vec3f f_ver = kfToPose * kf_ver;

                if (f_ver(2) <= 0)
                    continue;

                vec3f f_ray = f_ver / f_ver(2);
                vec2f f_pix = cam.rayToPix(f_ray);

                if (!cam.isPixVisible(f_pix))
                    continue;

                float residual = 0.0;
                int count = 0;
                for (int y = -win_size; y <= win_size; y++)
                {
                    for (int x = -win_size; x <= win_size; x++)
                    {
                        vec2f shift_pix(x, y);
                        vec2f kf_pix_ = kf_pix + shift_pix;
                        vec2f f_pix_ = f_pix + shift_pix;

                        if (!cam.isPixVisible(f_pix_))
                            continue;
                        if (!cam.isPixVisible(kf_pix_))
                            continue;

                        imageType kf_i = kimage.get(kf_pix_(1), kf_pix_(0));
                        imageType f_i = image.get(f_pix_(1), f_pix_(0));

                        if (kf_i == kimage.nodata || f_i == image.nodata)
                            continue;

                        float f_i_cor = alpha * (f_i - beta);
                        residual += (f_i_cor - kf_i) * (f_i_cor - kf_i);
                        count += 1;
                    }
                }

                if (count < 0)
                    continue;

                residual /= count;

                if (residual < best_residual)
                {
                    best_residual = residual;
                    vertex best_vertex(kf_ver, kf_ray, kf_pix, 1.0 / mesh_vo::mapping_param_initial_var);
                    scene2.setVertex(t_id, best_vertex);
                }
            }
        }
    }

    void renderDepthFromClosestShapeWindow(cameraType cam, window<float> win)
    {
        std::vector<int> p_ids = scene1.getParamIds();
        std::vector<int> s_ids = scene1.getShapesIds();

        for (int p_id : p_ids)
        {
            vertex vert = scene1.getVertex(p_id);

            if (vert.weight > 1.0 / mesh_vo::mapping_param_initial_var)
                continue;

            // if (!win.isPixInWindow(vert.pix))
            //     continue;

            float closest_depth = -1.0;
            float closest_distance = 100000000.0;
            for (int s_id : s_ids)
            {
                shapeType shape = scene1.getShape(s_id);
                float distance = (vert.pix - shape.getCenterPix()).norm();
                if (distance < closest_distance)
                {
                    float depth = vert.ver(2);
                    if (depth <= 0.0)
                        continue;
                    closest_distance = distance;
                    closest_depth = depth;
                }
            }

            if (closest_depth > 0.0)
            {
                vertex best_vertex(vert.ray * closest_depth, vert.ray, vert.pix, 1.0 / mesh_vo::mapping_param_initial_var);
                scene2.setVertex(p_id, best_vertex);
            }
        }
    }

    void renderImageWindow(dataCPU<imageType> &image, SE3f kfTofPose, SE3f fTokfPose, dataCPU<float> &buffer, cameraType cam, window<float> win)
    {
        int width = buffer.width;
        int height = buffer.height;

        std::vector<int> ids = scene2.getShapesIds();

        for (int t_id : ids)
        {
            shapeType f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() < 0.0)
                continue;

            shapeType kf_pol = scene1.getShape(t_id);

            window<float> pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            window<int> pol_win_2(int(pol_win.min_x * (width - 1)),
                                  int(pol_win.max_x * (width - 1)),
                                  int(pol_win.min_y * (height - 1)),
                                  int(pol_win.max_y * (height - 1)));

            for (int y = pol_win_2.min_y; y <= pol_win_2.max_y; y += 1)
            {
                for (int x = pol_win_2.min_x; x <= pol_win_2.max_x; x += 1)
                {
                    vec2i f_pix_tex(x, y);
                    vec2f f_pix(float(x) / (width - 1), float(y) / (height - 1));

                    f_pol.usePixel(f_pix);

                    if (!f_pol.isPixInShape())
                        continue;

                    vec3f f_ver = f_pol.getVer();
                    float f_depth = f_ver(2);
                    if (f_depth <= 0.0)
                        continue;

                    float z_depth = z_buffer.getTexel(f_pix_tex(1), f_pix_tex(0));
                    if (z_depth < f_depth && z_depth != z_buffer.nodata)
                        continue;

                    vec3f kf_ver = fTokfPose * f_ver;
                    vec3f kf_ray = kf_ver / kf_ver(2);
                    vec2f kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = image.get(kf_pix(1), kf_pix(0));
                    if (kf_i == image.nodata)
                        continue;

                    // buffer->set(std::exp(kf_a) * (kf_i - kf_b), y, x, lvl);
                    buffer.setTexel(kf_i, f_pix_tex(1), f_pix_tex(0));
                    z_buffer.setTexel(f_depth, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderDepthWindow(dataCPU<float> &buffer, window<float> win)
    {
        int width = buffer.width;
        int height = buffer.height;

        std::vector<int> shapesIds = scene2.getShapesIds();

        for (int t_id : shapesIds)
        {
            shapeType f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() < 0.0)
                continue;

            window<float> pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            window<int> pol_win_2(int(pol_win.min_x * (width - 1)),
                                  int(pol_win.max_x * (width - 1)),
                                  int(pol_win.min_y * (height - 1)),
                                  int(pol_win.max_y * (height - 1)));

            for (int y = pol_win_2.min_y; y <= pol_win_2.max_y; y += 1)
            {
                for (int x = pol_win_2.min_x; x <= pol_win_2.max_x; x += 1)
                {
                    vec2i f_pix_tex(x, y);
                    vec2f f_pix(float(x) / (width - 1), float(y) / (height - 1));

                    // for (float y = pol_win.min_y; y <= pol_win.max_y; y += step_y)
                    //{
                    //     for (float x = pol_win.min_x; x <= pol_win.max_x; x += step_x)
                    //     {
                    //         vec2i f_pix_tex(int(x * (width - 1)), int(y * (height - 1)));
                    //         vec2f f_pix(x, y);

                    f_pol.usePixel(f_pix);

                    if (!f_pol.isPixInShape())
                        continue;

                    float f_depth = f_pol.getDepth();

                    if (f_depth <= 0.0)
                        continue;

                    float z_depth = z_buffer.getTexel(f_pix_tex(1), f_pix_tex(0));
                    if (z_depth <= f_depth && z_depth != z_buffer.nodata)
                        continue;

                    buffer.setTexel(f_depth, f_pix_tex(1), f_pix_tex(0));
                    z_buffer.setTexel(f_depth, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderWeightWindow(dataCPU<float> &buffer, window<float> win)
    {
        int width = buffer.width;
        int height = buffer.height;

        std::vector<int> shapesIds = scene2.getShapesIds();

        for (int t_id : shapesIds)
        {
            shapeType f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() < 0.0)
                continue;

            window<float> pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            window<int> pol_win_2(int(pol_win.min_x * (width - 1)),
                                  int(pol_win.max_x * (width - 1)),
                                  int(pol_win.min_y * (height - 1)),
                                  int(pol_win.max_y * (height - 1)));

            for (int y = pol_win_2.min_y; y <= pol_win_2.max_y; y += 1)
            {
                for (int x = pol_win_2.min_x; x <= pol_win_2.max_x; x += 1)
                {
                    vec2i f_pix_tex(x, y);
                    vec2f f_pix(float(x) / (width - 1), float(y) / (height - 1));

                    f_pol.usePixel(f_pix);

                    if (!f_pol.isPixInShape())
                        continue;

                    float f_depth = f_pol.getDepth();

                    if (f_depth <= 0.0)
                        continue;

                    float z_depth = z_buffer.getTexel(f_pix_tex(1), f_pix_tex(0));
                    if (z_depth <= f_depth && z_depth != z_buffer.nodata)
                        continue;

                    float f_weight = f_pol.getWeight();

                    buffer.setTexel(f_weight, f_pix_tex(1), f_pix_tex(0));
                    z_buffer.setTexel(f_depth, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderResidualWindow(dataCPU<float> &kimage, dataCPU<float> &image, vec2f imageExp, SE3f fTokfPose, dataCPU<float> &e_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

        std::vector<int> t_ids = scene2.getShapesIds();

        float alpha = std::exp(-imageExp(0));
        float beta = imageExp(1);

        for (int t_id : t_ids)
        {
            shapeType f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() <= min_area)
                continue;

            shapeType kf_pol = scene1.getShape(t_id);

            window<float> pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            window<int> pol_win_2(int(pol_win.min_x * (width - 1)),
                                  int(pol_win.max_x * (width - 1)),
                                  int(pol_win.min_y * (height - 1)),
                                  int(pol_win.max_y * (height - 1)));

            for (int y = pol_win_2.min_y; y <= pol_win_2.max_y; y += 1)
            {
                for (int x = pol_win_2.min_x; x <= pol_win_2.max_x; x += 1)
                {
                    vec2i f_pix_tex(x, y);
                    vec2f f_pix(float(x) / (width - 1), float(y) / (height - 1));

                    f_pol.usePixel(f_pix);

                    if (!f_pol.isPixInShape())
                        continue;

                    vec3f f_ver = f_pol.getVer();
                    float f_depth = f_ver(2);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_depth = z_buffer.getTexel(f_pix_tex(1), f_pix_tex(0));
                    if (l_depth < f_depth && l_depth != z_buffer.nodata)
                        continue;

                    vec3f kf_ver = fTokfPose * f_ver;
                    vec3f kf_ray = kf_ver / kf_ver(2);
                    vec2f kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    imageType f_i = image.getTexel(f_pix_tex(1), f_pix_tex(0));

                    if (kf_i == kimage.nodata || f_i == image.nodata)
                        continue;

                    float f_i_cor = alpha * (f_i - beta);

                    float residual = (f_i_cor - kf_i);

                    e_buffer.setTexel(residual, f_pix_tex(1), f_pix_tex(0));
                    z_buffer.setTexel(f_depth, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderJIntrinsicWindow(dataCPU<float> &kimage, dataCPU<vec2f> &d_kimage_d_pix, dataCPU<float> &image, vec2f imageExp, SE3f fTokfPose, dataCPU<vec2f> &d_image_d_pix, SE3f imagePose, dataCPU<cameraParamType> &jintrinsic_buffer, dataCPU<float> &e_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3 / 4;

        float alpha = std::exp(-imageExp(0));
        float beta = imageExp(1);

        std::vector<int> t_ids = scene2.getShapesIds();

        // int shapeDoF = scene2->getShapesDoF();

        for (int t_id : t_ids)
        {
            // std::vector<int> p_ids = scene2->getShapeParamsIds(t_id);

            shapeType f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() < min_area)
                continue;

            shapeType kf_pol = scene1.getShape(t_id);

            if (kf_pol.getScreenArea() < min_area)
                continue;

            idsType ids = kf_pol.getParamIds();

            window<float> pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            window<int> pol_win_2(int(pol_win.min_x * (width - 1)),
                                  int(pol_win.max_x * (width - 1)),
                                  int(pol_win.min_y * (height - 1)),
                                  int(pol_win.max_y * (height - 1)));

            for (int y = pol_win_2.min_y; y <= pol_win_2.max_y; y += 1)
            {
                for (int x = pol_win_2.min_x; x <= pol_win_2.max_x; x += 1)
                {
                    vec2i f_pix_tex(x, y);
                    vec2f f_pix(float(x) / (width - 1), float(y) / (height - 1));

                    f_pol.usePixel(f_pix);

                    if (!f_pol.isPixInShape())
                        continue;

                    vec3f f_ver = f_pol.getVer();
                    vec3f f_ray = f_ver / f_ver(2);
                    float f_depth = f_ver(2);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_idepth = z_buffer.getTexel(f_pix_tex(1), f_pix_tex(0));
                    if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                        continue;

                    vec3f kf_ver = fTokfPose * f_ver;
                    float kf_depth = kf_ver(2);
                    vec3f kf_ray = kf_ver / kf_ver(2);
                    vec2f kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    imageType f_i = image.getTexel(f_pix_tex(1), f_pix_tex(0));
                    vec2f d_kf_i_d_pix = d_kimage_d_pix.get(kf_pix(1), kf_pix(0));
                    vec2f d_f_i_d_pix = d_image_d_pix.getTexel(f_pix_tex(1), f_pix_tex(0));

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata || d_kf_i_d_pix == d_kimage_d_pix.nodata)
                        continue;

                    float f_i_cor = alpha * (f_i - beta);

                    float residual = f_i_cor - kf_i;

                    cameraParamType d_f_i_d_intrinsics = d_f_i_d_pix.transpose() * cam.d_pix_d_intrinsics(f_ray);
                    cameraParamType d_kf_i_d_intrinsics = d_kf_i_d_pix.transpose() * cam.d_pix_d_intrinsics(kf_ray);
                    cameraParamType d_res_d_intrinsics = d_f_i_d_intrinsics - d_kf_i_d_intrinsics;

                    e_buffer.setTexel(residual, f_pix_tex(1), f_pix_tex(0));
                    jintrinsic_buffer.setTexel(d_res_d_intrinsics, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderJPoseWindow(dataCPU<float> &kimage, dataCPU<float> &image, vec2f imageAffine, SE3f fTokfPose, dataCPU<vec2f> &d_image_d_pix, dataCPU<vec6f> &jpose_buffer, dataCPU<float> &e_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

        float alpha = std::exp(-imageAffine(0));
        float beta = imageAffine(1);

        std::vector<int> t_ids = scene2.getShapesIds();

        for (int t_id : t_ids)
        {
            shapeType f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() <= min_area)
                continue;

            shapeType kf_pol = scene1.getShape(t_id);

            window<float> pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            window<int> pol_win_2(int(pol_win.min_x * (width - 1)),
                                  int(pol_win.max_x * (width - 1)),
                                  int(pol_win.min_y * (height - 1)),
                                  int(pol_win.max_y * (height - 1)));

            for (int y = pol_win_2.min_y; y <= pol_win_2.max_y; y += 1)
            {
                for (int x = pol_win_2.min_x; x <= pol_win_2.max_x; x += 1)
                {
                    vec2i f_pix_tex(x, y);
                    vec2f f_pix(float(x) / (width - 1), float(y) / (height - 1));

                    f_pol.usePixel(f_pix);

                    if (!f_pol.isPixInShape())
                        continue;

                    vec3f f_ver = f_pol.getVer();
                    float f_depth = f_ver(2);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_depth = z_buffer.getTexel(f_pix_tex(1), f_pix_tex(0));
                    if (l_depth < f_depth && l_depth != z_buffer.nodata)
                        continue;

                    vec3f kf_ver = fTokfPose * f_ver;
                    vec3f kf_ray = kf_ver / kf_ver(2);
                    vec2f kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    imageType f_i = image.getTexel(f_pix_tex(1), f_pix_tex(0));
                    vec2f d_f_i_d_pix = d_image_d_pix.getTexel(f_pix_tex(1), f_pix_tex(0));

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata)
                        continue;

                    mat<float, 2, 3> d_pix_d_f_ver = cam.d_pix_d_ver(f_ver);
                    vec3f d_f_i_d_f_ver = d_f_i_d_pix.transpose() * d_pix_d_f_ver;

                    vec3f d_f_i_d_tra = d_f_i_d_f_ver;
                    // vec3f d_f_i_d_rot(-f_ver(2) * v1 + f_ver(1) * v2, f_ver(2) * v0 - f_ver(0) * v2, -f_ver(1) * v0 + f_ver(0) * v1);
                    vec3f d_f_i_d_rot(-f_ver(2) * d_f_i_d_tra(1) + f_ver(1) * d_f_i_d_tra(2), f_ver(2) * d_f_i_d_tra(0) - f_ver(0) * d_f_i_d_tra(2), -f_ver(1) * d_f_i_d_tra(0) + f_ver(0) * d_f_i_d_tra(1));

                    vec6f j_pose;
                    j_pose << d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2);

                    float f_i_cor = alpha * (f_i - beta);

                    float residual = (f_i_cor - kf_i);

                    jpose_buffer.setTexel(j_pose, f_pix_tex(1), f_pix_tex(0));
                    e_buffer.setTexel(residual, f_pix_tex(1), f_pix_tex(0));
                    z_buffer.setTexel(f_depth, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderJPoseExpWindow(dataCPU<float> &kimage, dataCPU<float> &image, vec2f imageExp, SE3f fTokfPose, dataCPU<vec2f> &d_image_d_pix, dataCPU<vec6f> &jpose_buffer, dataCPU<vec2f> &jexp_buffer, dataCPU<float> &e_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

        float alpha = std::exp(-imageExp(0));
        float beta = imageExp(1);

        std::vector<int> t_ids = scene2.getShapesIds();

        for (int t_id : t_ids)
        {
            shapeType f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() <= min_area)
                continue;

            shapeType kf_pol = scene1.getShape(t_id);

            window<float> pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            window<int> pol_win_2(int(pol_win.min_x * (width - 1)),
                                  int(pol_win.max_x * (width - 1)),
                                  int(pol_win.min_y * (height - 1)),
                                  int(pol_win.max_y * (height - 1)));

            for (int y = pol_win_2.min_y; y <= pol_win_2.max_y; y += 1)
            {
                for (int x = pol_win_2.min_x; x <= pol_win_2.max_x; x += 1)
                {
                    vec2i f_pix_tex(x, y);
                    vec2f f_pix(float(x) / (width - 1), float(y) / (height - 1));

                    f_pol.usePixel(f_pix);

                    if (!f_pol.isPixInShape())
                        continue;

                    vec3f f_ver = f_pol.getVer();
                    float f_depth = f_ver(2);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_depth = z_buffer.getTexel(f_pix_tex(1), f_pix_tex(0));
                    if (l_depth < f_depth && l_depth != z_buffer.nodata)
                        continue;

                    vec3f kf_ver = fTokfPose * f_ver;
                    vec3f kf_ray = kf_ver / kf_ver(2);
                    vec2f kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    imageType f_i = image.getTexel(f_pix_tex(1), f_pix_tex(0));
                    vec2f d_f_i_d_pix = d_image_d_pix.getTexel(f_pix_tex(1), f_pix_tex(0));

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata)
                        continue;

                    mat<float, 2, 3> d_pix_d_f_ver = cam.d_pix_d_ver(f_ver);
                    vec3f d_f_i_d_f_ver = d_f_i_d_pix.transpose() * d_pix_d_f_ver;

                    vec3f d_f_i_d_tra = d_f_i_d_f_ver;
                    vec3f d_f_i_d_rot(-f_ver(2) * d_f_i_d_tra(1) + f_ver(1) * d_f_i_d_tra(2), f_ver(2) * d_f_i_d_tra(0) - f_ver(0) * d_f_i_d_tra(2), -f_ver(1) * d_f_i_d_tra(0) + f_ver(0) * d_f_i_d_tra(1));

                    float f_i_cor = alpha * (f_i - beta);

                    vec2f d_f_i_d_exp(-f_i_cor, -alpha);

                    vec6f j_pose;
                    j_pose << d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2);
                    vec2f j_exp = vec2f(d_f_i_d_exp(0), d_f_i_d_exp(1));

                    float residual = (f_i_cor - kf_i);

                    jpose_buffer.setTexel(j_pose, f_pix_tex(1), f_pix_tex(0));
                    jexp_buffer.setTexel(j_exp, f_pix_tex(1), f_pix_tex(0));
                    e_buffer.setTexel(residual, f_pix_tex(1), f_pix_tex(0));
                    z_buffer.setTexel(f_depth, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderJExpWindow(dataCPU<float> &kimage, dataCPU<float> &image, vec2f imageExp, SE3f fTokfPose, dataCPU<vec2f> &jexp_buffer, dataCPU<float> &e_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

        std::vector<int> t_ids = scene2.getShapesIds();

        float alpha = std::exp(-imageExp(0));
        float beta = imageExp(1);

        for (int t_id : t_ids)
        {
            shapeType f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() <= min_area)
                continue;

            shapeType kf_pol = scene1.getShape(t_id);

            window<float> pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            window<int> pol_win_2(int(pol_win.min_x * (width - 1)),
                                  int(pol_win.max_x * (width - 1)),
                                  int(pol_win.min_y * (height - 1)),
                                  int(pol_win.max_y * (height - 1)));

            for (int y = pol_win_2.min_y; y <= pol_win_2.max_y; y += 1)
            {
                for (int x = pol_win_2.min_x; x <= pol_win_2.max_x; x += 1)
                {
                    vec2i f_pix_tex(x, y);
                    vec2f f_pix(float(x) / (width - 1), float(y) / (height - 1));

                    f_pol.usePixel(f_pix);

                    if (!f_pol.isPixInShape())
                        continue;

                    vec3f f_ver = f_pol.getVer();
                    float f_depth = f_ver(2);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_depth = z_buffer.get(y, x);
                    if (l_depth < f_depth && l_depth != z_buffer.nodata)
                        continue;

                    vec3f kf_ver = fTokfPose * f_ver;
                    vec3f kf_ray = kf_ver / kf_ver(2);
                    vec2f kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    imageType f_i = image.getTexel(f_pix_tex(1), f_pix_tex(0));

                    if (kf_i == kimage.nodata || f_i == image.nodata)
                        continue;

                    float f_i_cor = alpha * (f_i - beta);

                    vec2f jexp(-f_i_cor, -alpha);

                    float residual = f_i_cor - kf_i;

                    jexp_buffer.setTexel(jexp, f_pix_tex(1), f_pix_tex(0));
                    e_buffer.setTexel(residual, f_pix_tex(1), f_pix_tex(0));
                    z_buffer.setTexel(f_depth, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderJMapWindow(dataCPU<float> &kimage, dataCPU<float> &image, vec2f imageExp, dataCPU<vec2f> &d_image_d_pix, SE3f kfTofPose, SE3f fTokfPose, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &e_buffer, dataCPU<idsType> &pId_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3 / 4;

        float alpha = std::exp(-imageExp(0));
        float beta = imageExp(1);

        std::vector<int> t_ids = scene2.getShapesIds();

        // int shapeDoF = scene2->getShapesDoF();

        for (int t_id : t_ids)
        {
            // std::vector<int> p_ids = scene2->getShapeParamsIds(t_id);

            shapeType f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() < min_area)
                continue;

            shapeType kf_pol = scene1.getShape(t_id);

            if (kf_pol.getScreenArea() < min_area)
                continue;

            idsType ids = kf_pol.getParamIds();

            window<float> pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            window<int> pol_win_2(int(pol_win.min_x * (width - 1)),
                                  int(pol_win.max_x * (width - 1)),
                                  int(pol_win.min_y * (height - 1)),
                                  int(pol_win.max_y * (height - 1)));

            for (int y = pol_win_2.min_y; y <= pol_win_2.max_y; y += 1)
            {
                for (int x = pol_win_2.min_x; x <= pol_win_2.max_x; x += 1)
                {
                    vec2i f_pix_tex(x, y);
                    vec2f f_pix(float(x) / (width - 1), float(y) / (height - 1));

                    f_pol.usePixel(f_pix);

                    if (!f_pol.isPixInShape())
                        continue;

                    vec3f f_ver = f_pol.getVer();
                    float f_depth = f_ver(2);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_idepth = z_buffer.getTexel(f_pix_tex(1), f_pix_tex(0));
                    if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                        continue;

                    vec3f kf_ver = fTokfPose * f_ver;
                    vec3f kf_ray = kf_ver / kf_ver(2);
                    vec2f kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    imageType f_i = image.getTexel(f_pix_tex(1), f_pix_tex(0));
                    vec2f d_f_i_d_pix = d_image_d_pix.getTexel(f_pix_tex(1), f_pix_tex(0));

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata)
                        continue;

                    float f_i_cor = alpha * (f_i - beta);

                    float residual = f_i_cor - kf_i;

                    mat<float, 2, 3> d_pix_d_f_ver = cam.d_pix_d_ver(f_ver);
                    vec3f d_f_i_d_f_ver = d_f_i_d_pix.transpose() * d_pix_d_f_ver;

                    vec3f d_f_ver_d_kf_depth = kfTofPose.rotationMatrix() * kf_ray;
                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.transpose() * d_f_ver_d_kf_depth;

                    // this could be the jacobian of the depth of the 3 vertices in a triangle
                    // or the jacobian of the normal + depth of a surfel
                    jmapType jacs = d_f_i_d_kf_depth * f_pol.d_depth_d_vert_depth_param(kf_pol);

                    e_buffer.setTexel(residual, f_pix_tex(1), f_pix_tex(0));
                    jmap_buffer.setTexel(jacs, f_pix_tex(1), f_pix_tex(0));
                    pId_buffer.setTexel(ids, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderJPoseMapWindow(dataCPU<float> &kimage, dataCPU<float> &image, vec2f imageExp, dataCPU<vec2f> &d_image_d_pix, SE3f kfTofPose, SE3f fTokfPose, dataCPU<vec6f> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &e_buffer, dataCPU<idsType> &pId_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3.0 / 4.0;

        float alpha = std::exp(-imageExp(0));
        float beta = imageExp(1);

        std::vector<int> t_ids = scene2.getShapesIds();
        // int shapeDoF = scene2->getShapesDoF();

        for (int t_id : t_ids)
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

            window<float> pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            idsType ids = kf_pol.getParamIds();

            window<int> pol_win_2(int(pol_win.min_x * (width - 1)),
                                  int(pol_win.max_x * (width - 1)),
                                  int(pol_win.min_y * (height - 1)),
                                  int(pol_win.max_y * (height - 1)));

            for (int y = pol_win_2.min_y; y <= pol_win_2.max_y; y += 1)
            {
                for (int x = pol_win_2.min_x; x <= pol_win_2.max_x; x += 1)
                {
                    vec2i f_pix_tex(x, y);
                    vec2f f_pix(float(x) / (width - 1), float(y) / (height - 1));

                    f_pol.usePixel(f_pix);

                    // if (!cam.isPixVisible(f_pix))
                    //     continue;

                    if (!f_pol.isPixInShape())
                        continue;

                    vec3f f_ver = f_pol.getVer();
                    float f_depth = f_ver(2);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_idepth = z_buffer.getTexel(f_pix_tex(1), f_pix_tex(0));
                    if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                        continue;

                    vec3f kf_ver = fTokfPose * f_ver;
                    vec3f kf_ray = kf_ver / kf_ver(2);
                    vec2f kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    imageType f_i = image.getTexel(f_pix_tex(1), f_pix_tex(0));
                    vec2f d_f_i_d_pix = d_image_d_pix.getTexel(f_pix_tex(1), f_pix_tex(0));

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata)
                        continue;

                    mat<float, 2, 3> d_pix_d_f_ver = cam.d_pix_d_ver(f_ver);
                    vec3f d_f_i_d_f_ver = d_f_i_d_pix.transpose() * d_pix_d_f_ver;

                    vec3f d_f_i_d_tra = d_f_i_d_f_ver;
                    vec3f d_f_i_d_rot(-f_ver(2) * d_f_i_d_tra(1) + f_ver(1) * d_f_i_d_tra(2), f_ver(2) * d_f_i_d_tra(0) - f_ver(0) * d_f_i_d_tra(2), -f_ver(1) * d_f_i_d_tra(0) + f_ver(0) * d_f_i_d_tra(1));

                    float f_i_cor = alpha * (f_i - beta);

                    vec6f jpose;
                    jpose << d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2);

                    vec3f d_f_ver_d_kf_depth = kfTofPose.rotationMatrix() * kf_ray;

                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    jmapType jacs = d_f_i_d_kf_depth * f_pol.d_depth_d_vert_depth_param(kf_pol);

                    float error = f_i_cor - kf_i;

                    e_buffer.setTexel(error, f_pix_tex(1), f_pix_tex(0));
                    jpose_buffer.setTexel(jpose, f_pix_tex(1), f_pix_tex(0));
                    jmap_buffer.setTexel(jacs, f_pix_tex(1), f_pix_tex(0));
                    pId_buffer.setTexel(ids, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderJPoseExpMapWindow(dataCPU<imageType> &kimage, dataCPU<imageType> &image, vec2f imageExp, dataCPU<vec2f> &d_image_d_pix, SE3f kfTofPose, SE3f fTokfPose, dataCPU<vec6f> &jpose_buffer, dataCPU<vec2f> &jexp_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<float> &e_buffer, dataCPU<idsType> &pId_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3.0 / 4.0;

        float alpha = std::exp(-imageExp(0));
        float beta = imageExp(1);

        std::vector<int> t_ids = scene2.getShapesIds();
        // int shapeDoF = scene2->getShapesDoF();

        for (int t_id : t_ids)
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

            window<float> pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            idsType ids = kf_pol.getParamIds();

            window<int> pol_win_2(int(pol_win.min_x * (width - 1)),
                                  int(pol_win.max_x * (width - 1)),
                                  int(pol_win.min_y * (height - 1)),
                                  int(pol_win.max_y * (height - 1)));

            for (int y = pol_win_2.min_y; y <= pol_win_2.max_y; y += 1)
            {
                for (int x = pol_win_2.min_x; x <= pol_win_2.max_x; x += 1)
                {
                    vec2i f_pix_tex(x, y);
                    vec2f f_pix(float(x) / (width - 1), float(y) / (height - 1));

                    f_pol.usePixel(f_pix);

                    // if (!cam.isPixVisible(f_pix))
                    //     continue;

                    if (!f_pol.isPixInShape())
                        continue;

                    vec3f f_ver = f_pol.getVer();
                    float f_depth = f_ver(2);
                    if (f_depth <= 0.0)
                        continue;

                    // z-buffer
                    float l_idepth = z_buffer.getTexel(f_pix_tex(1), f_pix_tex(0));
                    if (l_idepth < f_depth && l_idepth != z_buffer.nodata)
                        continue;

                    vec3f kf_ver = fTokfPose * f_ver;
                    vec3f kf_ray = kf_ver / kf_ver(2);
                    vec2f kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    imageType f_i = image.getTexel(f_pix_tex(1), f_pix_tex(0));
                    vec2f d_f_i_d_pix = d_image_d_pix.getTexel(f_pix_tex(1), f_pix_tex(0));

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata)
                        continue;

                    mat<float, 2, 3> d_pix_d_f_ver = cam.d_pix_d_ver(f_ver);
                    vec3f d_f_i_d_f_ver = d_f_i_d_pix.transpose() * d_pix_d_f_ver;

                    vec3f d_f_i_d_tra = d_f_i_d_f_ver;
                    vec3f d_f_i_d_rot(-f_ver(2) * d_f_i_d_tra(1) + f_ver(1) * d_f_i_d_tra(2), f_ver(2) * d_f_i_d_tra(0) - f_ver(0) * d_f_i_d_tra(2), -f_ver(1) * d_f_i_d_tra(0) + f_ver(0) * d_f_i_d_tra(1));

                    float f_i_cor = alpha * (f_i - beta);

                    vec2f d_f_i_d_f_exp(-f_i_cor, -alpha);

                    vec6f jpose;
                    jpose << d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2);

                    vec2f jexp = vec2f(d_f_i_d_f_exp(0), d_f_i_d_f_exp(1));

                    vec3f d_f_ver_d_kf_depth = kfTofPose.rotationMatrix() * kf_ray;

                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    jmapType jacs = d_f_i_d_kf_depth * f_pol.d_depth_d_vert_depth_param(kf_pol);

                    float error = f_i_cor - kf_i;

                    e_buffer.setTexel(error, f_pix_tex(1), f_pix_tex(0));
                    jpose_buffer.setTexel(jpose, f_pix_tex(1), f_pix_tex(0));
                    jexp_buffer.setTexel(jexp, f_pix_tex(1), f_pix_tex(0));
                    jmap_buffer.setTexel(jacs, f_pix_tex(1), f_pix_tex(0));
                    pId_buffer.setTexel(ids, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderDebugWindow(dataCPU<imageType> &image, dataCPU<imageType> &buffer, window<float> win)
    {
        int width = buffer.width;
        int height = buffer.height;

        std::vector<int> ids = scene2.getShapesIds();

        for (int t_id : ids)
        {
            shapeType f_pol = scene2.getShape(t_id);

            if (!win.isPixInWindow(f_pol.getCenterPix()))
                continue;

            if (f_pol.getScreenArea() < 0.0)
                continue;

            window<float> pol_win = f_pol.getScreenBounds();

            pol_win.intersect(win);

            window<int> pol_win_2(int(pol_win.min_x * (width - 1)),
                                  int(pol_win.max_x * (width - 1)),
                                  int(pol_win.min_y * (height - 1)),
                                  int(pol_win.max_y * (height - 1)));

            for (int y = pol_win_2.min_y; y <= pol_win_2.max_y; y += 1)
            {
                for (int x = pol_win_2.min_x; x <= pol_win_2.max_x; x += 1)
                {
                    vec2i f_pix_tex(x, y);
                    vec2f f_pix(float(x) / (width - 1), float(y) / (height - 1));

                    f_pol.usePixel(f_pix);

                    if (!f_pol.isPixInShape())
                        continue;

                    // float f_depth = f_pol.getDepth(f_pix);
                    bool isLine = f_pol.isEdge();

                    imageType f_i = image.get(y, x);
                    f_i /= 255.0;

                    // z buffer
                    // float l_idepth = z_buffer.get(f_pix(1), f_pix(0), lvl);
                    // if (l_idepth > f_idepth && l_idepth != z_buffer.nodata)
                    //    continue;

                    if (isLine)
                        buffer.setTexel(1.0, f_pix_tex(1), f_pix_tex(0));
                    else
                        // buffer->set(1.0 / f_depth, y, x, lvl);
                        buffer.setTexel(f_i, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    geometryType scene1;
    geometryType scene2;
    dataCPU<float> z_buffer;
    ThreadPool<mesh_vo::renderer_nthreads> pool;
};
