#pragma once

#include <thread>
#include <vector>

#include "params.h"
#include "common/types.h"
#include "common/camera.h"
#include "common/window.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/keyFrameCPU.h"
#include "cpu/GeometryMesh.h"
#include "threadpoolCPU.h"

class renderCPU2
{
public:
    renderCPU2(unsigned int width, unsigned int height)
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

                renderIdepthLineSearchWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), relativePose, cam, win);
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

    void renderImageParallel(keyFrameCPU &kframe, SE3f localPose, dataMipMapCPU<imageType> &buffer, cameraType cam, int lvl)
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
                // pool.enqueue(std::bind(&renderCPU::renderImageWindow, this, kframe.getRawImage(lvl), kfTofPose, fTokfPose, buffer.get(lvl), cam, win));
            }
        }

        // pool.waitUntilDone();
    }

    void renderDebugParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<imageType> &buffer, cameraType cam, int lvl)
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
                // pool.enqueue(std::bind(&renderCPU::renderDebugWindow, this, frame.getRawImage(lvl), buffer.get(lvl), win));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJMapParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<jmapType> &jmap_buffer, dataMipMapCPU<errorType> &e_buffer, dataMipMapCPU<idsType> &pId_buffer, cameraType cam, int lvl)
    {
        z_buffer.setToNoData();

        SE3f kfTofPose = frame.getLocalPose();
        SE3f fTokfPose = kfTofPose.inverse();

        jvelType kfTofVel = frame.getLocalVel();
        jvelType fTokfVel = -kfTofVel;

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

                renderJMapWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getdIdpixImage(lvl), kfTofPose, fTokfPose, jmap_buffer.get(lvl), e_buffer.get(lvl), pId_buffer.get(lvl), cam, win);
                // pool.enqueue(std::bind(&renderCPU::renderJMapWindow, this, kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getLocalExp(), frame.getdIdpixImage(lvl), kfTofPose, fTokfPose, jmap_buffer.get(lvl), e_buffer.get(lvl), pId_buffer.get(lvl), cam, win));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJPoseMapParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<jposeType> &jpose_buffer, dataMipMapCPU<jmapType> &jmap_buffer, dataMipMapCPU<errorType> &e_buffer, dataMipMapCPU<idsType> &pId_buffer, cameraType cam, int lvl)
    {
        z_buffer.setToNoData();

        SE3f kfTofPose = frame.getLocalPose();
        SE3f fTokfPose = kfTofPose.inverse();

        jvelType kfTofVel = frame.getLocalVel();
        jvelType fTokfVel = -kfTofVel;

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

                renderJPoseMapWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getdIdpixImage(lvl), kfTofPose, fTokfPose, jpose_buffer.get(lvl), jmap_buffer.get(lvl), e_buffer.get(lvl), pId_buffer.get(lvl), cam, win);
                // pool.enqueue(std::bind(&renderCPU::renderJPoseMapWindow, this, kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getdIdpixImage(lvl), kfTofPose, fTokfPose, jpose_buffer.get(lvl), jmap_buffer.get(lvl), e_buffer.get(lvl), pId_buffer.get(lvl), cam, win));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJIntrinsicParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<jcamType> &jintrinsic_buffer, dataMipMapCPU<errorType> &e_buffer, cameraType cam, int lvl)
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

                renderJIntrinsicWindow(kframe.getRawImage(lvl), kframe.getdIdpixImage(lvl), frame.getRawImage(lvl), fTokfPose, frame.getdIdpixImage(lvl), frame.getLocalPose(), jintrinsic_buffer.get(lvl), e_buffer.get(lvl), cam, win);
                // pool.enqueue(std::bind(&renderCPU::renderJIntrinsicParallel, this, kframe.getRawImage(lvl), kframe.getdIdpixImage(lvl), frame.getRawImage(lvl), frame.getLocalExp(), fTokfPose, frame.getdIdpixImage(lvl), frame.getLocalPose(), jintrinsic_buffer.get(lvl), e_buffer.get(lvl), cam, win));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJPoseParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<vec6f> &jpose_buffer, dataMipMapCPU<errorType> &e_buffer, cameraType cam, int lvl)
    // void renderJPoseParallel(geometryType &geometry, dataCPU<imageType> &kimage, dataCPU<imageType> &image, dataCPU<jimgType> &dIdpix, SE3f kfTofPose, dataCPU<jposeType> &jpose_buffer, dataCPU<errorType> &e_buffer, cameraType cam)
    {
        z_buffer.setToNoData();

        SE3f kfTofPose = frame.getLocalPose();
        SE3f fTokfPose = kfTofPose.inverse();

        jvelType kfTofVel = frame.getLocalVel();
        jvelType fTokfVel = -kfTofVel;

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

                renderJPoseWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), fTokfPose, frame.getdIdpixImage(lvl), jpose_buffer.get(lvl), e_buffer.get(lvl), cam, win);
                // pool.enqueue(std::bind(&renderCPU::renderJPoseWindow, this, kimage, image, fTokfPose, dIdpix, jpose_buffer, e_buffer, cam, win));
            }
        }

        // pool.waitUntilDone();
    }

    void renderJPoseVelParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<jposeType> &jpose_buffer, dataMipMapCPU<jvelType> &jvel_buffer, dataMipMapCPU<errorType> &e_buffer, cameraType cam, int lvl)
    // void renderJPoseParallel(geometryType &geometry, dataCPU<imageType> &kimage, dataCPU<imageType> &image, dataCPU<jimgType> &dIdpix, SE3f kfTofPose, dataCPU<jposeType> &jpose_buffer, dataCPU<errorType> &e_buffer, cameraType cam)
    {
        z_buffer.setToNoData();

        SE3f kfTofPose = frame.getLocalPose();
        SE3f fTokfPose = kfTofPose.inverse();

        jvelType kfTofVel = frame.getLocalVel();
        jvelType fTokfVel = -kfTofVel;

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

                renderJPoseVelWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), fTokfPose, fTokfVel, frame.getdIdpixImage(lvl), jpose_buffer.get(lvl), jvel_buffer.get(lvl), e_buffer.get(lvl), cam, win);
                // pool.enqueue(std::bind(&renderCPU::renderJPoseWindow, this, kimage, image, fTokfPose, dIdpix, jpose_buffer, e_buffer, cam, win));
            }
        }

        // pool.waitUntilDone();
    }

    void renderResidualParallel(keyFrameCPU &kframe, frameCPU &frame, dataMipMapCPU<errorType> &e_buffer, cameraType cam, int lvl)
    {
        z_buffer.setToNoData();

        SE3f kfTofPose = frame.getLocalPose();
        SE3f fTokfPose = kfTofPose.inverse();

        jvelType kfTofVel = frame.getLocalVel();
        jvelType fTokfVel = -kfTofVel;

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

                renderResidualWindow(kframe.getRawImage(lvl), frame.getRawImage(lvl), fTokfPose, e_buffer.get(lvl), cam, win);
                // pool.enqueue(std::bind(&renderCPU::renderResidualWindow, this, kframe.getRawImage(lvl), frame.getRawImage(lvl), frame.getLocalExp(), fTokfPose, e_buffer.get(lvl), cam, win));
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
                // pool.enqueue(std::bind(&renderCPU::renderDepthWindow, this, buffer.get(lvl), win));
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
                // pool.enqueue(std::bind(&renderCPU::renderWeightWindow, this, buffer.get(lvl), win));
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

    void renderIdepthLineSearchWindow(dataCPU<imageType> &kimage, dataCPU<imageType> &image, SE3f imagePose, cameraType cam, window<float> win)
    {
        float min_idepth = 1.0 / 10.0;
        float max_idepth = 1.0 / 0.1;

        float x_step = 1.0 / kimage.width;
        float y_step = 1.0 / kimage.height;
        float x_size = 5.0 / kimage.width; // 2 pixels
        float y_size = 5.0 / kimage.height;

        SE3f kfToPose = imagePose;
        SE3f fToKfPose = imagePose.inverse();

        std::vector<int> t_ids = scene1.getParamIds();

        for (int t_id : t_ids)
        {
            vertex vert = scene1.getVertex(t_id);

            // if (vert.weight > 1.0 / mesh_vo::mapping_param_initial_var)
            //     continue;

            vec3f kf_ray = vert.ray;
            vec2f kf_pix = vert.pix;

            if (!win.isPixInWindow(kf_pix))
                continue;

            // vec3f kf_ver_min = kf_ray / min_idepth;
            // vec3f f_ver_min = kfToPose * kf_ver_min;
            // vec3f f_ray_min = f_ver_min / f_ver_min(2);
            // vec2f f_pix_min = cam.rayToPix(f_ray_min);

            // vec3f kf_ver_max = kf_ray / max_idepth;
            // vec3f f_ver_max = kfToPose * kf_ver_max;
            // vec3f f_ray_max = f_ver_max / f_ver_max(2);
            // vec2f f_pix_max = cam.rayToPix(f_ray_max);

            // vec2f f_pix_diff = f_pix_max - f_pix_min;
            // float pix_dist = f_pix_diff.norm();
            // if (pix_dist < 1.0)
            //     continue;

            float idepth_step = (max_idepth - min_idepth) / 10.0; // pix_dist

            float best_residual = 100000000000.0;
            for (float kf_idepth = min_idepth; kf_idepth <= max_idepth; kf_idepth += idepth_step)
            {
                vec3f kf_ver = kf_ray / kf_idepth;
                vec3f f_ver = kfToPose * kf_ver;
                vec3f f_ray = f_ver / f_ver(2);
                vec2f f_pix = cam.rayToPix(f_ray);

                if (!cam.isPixVisible(f_pix))
                    continue;

                float residual = 0.0;
                int count = 0;
                for (float y = -y_size; y <= y_size; y += y_step)
                {
                    for (float x = -x_size; x <= x_size; x += x_step)
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

                        residual += (f_i - kf_i) * (f_i - kf_i);
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

    void renderImageWindow(dataCPU<imageType> &image, SE3f kfTofPose, SE3f fTokfPose, dataCPU<imageType> &buffer, cameraType cam, window<float> win)
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

    void renderResidualWindow(dataCPU<imageType> &kimage, dataCPU<imageType> &image, SE3f fTokfPose, dataCPU<errorType> &e_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

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

                    if (kf_i == kimage.nodata || f_i == image.nodata)
                        continue;

                    errorType residual = errorType(f_i) - errorType(kf_i);

                    e_buffer.setTexel(residual, f_pix_tex(1), f_pix_tex(0));
                    z_buffer.setTexel(f_depth, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderResidualWindow(dataCPU<imageType> &kimage, dataCPU<imageType> &image, SE3f fTokfPose, jvelType fVel, dataCPU<errorType> &e_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

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

                    float fdt = mesh_vo::line_capture_time * (f_pix(1) - 0.5);
                    SE3f fTokfPoseVel = SE3f::exp(fVel * fdt) * fTokfPose;

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

                    vec3f kf_ver = fTokfPoseVel * f_ver;
                    vec3f kf_ray = kf_ver / kf_ver(2);
                    vec2f kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    imageType f_i = image.getTexel(f_pix_tex(1), f_pix_tex(0));

                    if (kf_i == kimage.nodata || f_i == image.nodata)
                        continue;

                    errorType residual = errorType(f_i) - errorType(kf_i);

                    e_buffer.setTexel(residual, f_pix_tex(1), f_pix_tex(0));
                    z_buffer.setTexel(f_depth, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderJIntrinsicWindow(dataCPU<imageType> &kimage, dataCPU<jimgType> &d_kimage_d_pix, dataCPU<imageType> &image, SE3f fTokfPose, dataCPU<jimgType> &d_image_d_pix, SE3f imagePose, dataCPU<jcamType> &jintrinsic_buffer, dataCPU<errorType> &e_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3 / 4;

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
                    jimgType d_kf_i_d_pix = d_kimage_d_pix.get(kf_pix(1), kf_pix(0));
                    jimgType d_f_i_d_pix = d_image_d_pix.getTexel(f_pix_tex(1), f_pix_tex(0));

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata || d_kf_i_d_pix == d_kimage_d_pix.nodata)
                        continue;

                    errorType residual = errorType(f_i) - errorType(kf_i);

                    jcamType d_f_i_d_intrinsics = d_f_i_d_pix.transpose() * cam.d_pix_d_intrinsics(f_ray);
                    jcamType d_kf_i_d_intrinsics = d_kf_i_d_pix.transpose() * cam.d_pix_d_intrinsics(kf_ray);
                    jcamType d_res_d_intrinsics = d_f_i_d_intrinsics - d_kf_i_d_intrinsics;

                    e_buffer.setTexel(residual, f_pix_tex(1), f_pix_tex(0));
                    jintrinsic_buffer.setTexel(d_res_d_intrinsics, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderJPoseWindow(dataCPU<imageType> &kimage, dataCPU<imageType> &image, SE3f fTokfPose, dataCPU<jimgType> &d_image_d_pix, dataCPU<jposeType> &jpose_buffer, dataCPU<errorType> &e_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

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
                    jimgType d_f_i_d_pix = d_image_d_pix.getTexel(f_pix_tex(1), f_pix_tex(0));

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata)
                        continue;

                    mat<float, 2, 3> d_pix_d_f_ver = cam.d_pix_d_ver(f_ver);
                    vec3f d_f_i_d_f_ver = d_f_i_d_pix.transpose() * d_pix_d_f_ver;

                    vec3f d_f_i_d_tra = d_f_i_d_f_ver;
                    // vec3f d_f_i_d_rot(-f_ver(2) * v1 + f_ver(1) * v2, f_ver(2) * v0 - f_ver(0) * v2, -f_ver(1) * v0 + f_ver(0) * v1);
                    vec3f d_f_i_d_rot(-f_ver(2) * d_f_i_d_tra(1) + f_ver(1) * d_f_i_d_tra(2), f_ver(2) * d_f_i_d_tra(0) - f_ver(0) * d_f_i_d_tra(2), -f_ver(1) * d_f_i_d_tra(0) + f_ver(0) * d_f_i_d_tra(1));

                    vec6f j_pose;
                    j_pose << d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2);

                    errorType residual = errorType(f_i) - errorType(kf_i);

                    jpose_buffer.setTexel(j_pose, f_pix_tex(1), f_pix_tex(0));
                    e_buffer.setTexel(residual, f_pix_tex(1), f_pix_tex(0));
                    z_buffer.setTexel(f_depth, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderJPoseWindow(dataCPU<imageType> &kimage, dataCPU<imageType> &image, SE3f fTokfPose, jvelType fTokfVel, dataCPU<jimgType> &d_image_d_pix, dataCPU<jposeType> &jpose_buffer, dataCPU<errorType> &e_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

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

                    float dt = mesh_vo::line_capture_time * (f_pix(1) - 0.5);
                    SE3f fTokfPoseVel = SE3f::exp(fTokfVel * dt) * fTokfPose;

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

                    vec3f kf_ver = fTokfPoseVel * f_ver;
                    vec3f kf_ray = kf_ver / kf_ver(2);
                    vec2f kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    imageType f_i = image.getTexel(f_pix_tex(1), f_pix_tex(0));
                    jimgType d_f_i_d_pix = d_image_d_pix.getTexel(f_pix_tex(1), f_pix_tex(0));

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata)
                        continue;

                    mat<float, 2, 3> d_pix_d_f_ver = cam.d_pix_d_ver(f_ver);
                    vec3f d_f_i_d_f_ver = d_f_i_d_pix.transpose() * d_pix_d_f_ver;

                    vec3f d_f_i_d_tra = d_f_i_d_f_ver;
                    // vec3f d_f_i_d_rot(-f_ver(2) * v1 + f_ver(1) * v2, f_ver(2) * v0 - f_ver(0) * v2, -f_ver(1) * v0 + f_ver(0) * v1);
                    vec3f d_f_i_d_rot(-f_ver(2) * d_f_i_d_tra(1) + f_ver(1) * d_f_i_d_tra(2), f_ver(2) * d_f_i_d_tra(0) - f_ver(0) * d_f_i_d_tra(2), -f_ver(1) * d_f_i_d_tra(0) + f_ver(0) * d_f_i_d_tra(1));

                    vec6f j_pose;
                    j_pose << d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2);

                    errorType residual = errorType(f_i) - errorType(kf_i);

                    jpose_buffer.setTexel(j_pose, f_pix_tex(1), f_pix_tex(0));
                    e_buffer.setTexel(residual, f_pix_tex(1), f_pix_tex(0));
                    z_buffer.setTexel(f_depth, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderJPoseVelWindow(dataCPU<imageType> &kimage, dataCPU<imageType> &image, SE3f fTokfPose, jvelType fTokfVel, dataCPU<jimgType> &d_image_d_pix, dataCPU<jposeType> &jpose_buffer, dataCPU<jvelType> &jvel_buffer, dataCPU<errorType> &e_buffer, cameraType cam, window<float> win)
    {
        // line pose = pose * dt * (line - middle_line) * vel

        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / MESH_WIDTH) * (float(cam.height) / MESH_HEIGHT) * 3 / 4;

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
                    // pose = last_pose * SE3f::exp(pinc + vinc * (y - image.height / 2))

                    vec2i f_pix_tex(x, y);
                    vec2f f_pix(float(x) / (width - 1), float(y) / (height - 1));

                    float dt = mesh_vo::line_capture_time * (f_pix(1) - 0.5);
                    SE3f fTokfPoseVel = SE3f::exp(fTokfVel * dt) * fTokfPose;

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

                    vec3f kf_ver = fTokfPoseVel * f_ver;
                    vec3f kf_ray = kf_ver / kf_ver(2);
                    vec2f kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    imageType f_i = image.getTexel(f_pix_tex(1), f_pix_tex(0));
                    jimgType d_f_i_d_pix = d_image_d_pix.getTexel(f_pix_tex(1), f_pix_tex(0));

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata)
                        continue;

                    mat<float, 2, 3> d_pix_d_f_ver = cam.d_pix_d_ver(f_ver);
                    vec3f d_f_i_d_f_ver = d_f_i_d_pix.transpose() * d_pix_d_f_ver;

                    vec3f d_f_i_d_tra = d_f_i_d_f_ver;
                    // vec3f d_f_i_d_rot(-f_ver(2) * v1 + f_ver(1) * v2, f_ver(2) * v0 - f_ver(0) * v2, -f_ver(1) * v0 + f_ver(0) * v1);
                    vec3f d_f_i_d_rot(-f_ver(2) * d_f_i_d_tra(1) + f_ver(1) * d_f_i_d_tra(2), f_ver(2) * d_f_i_d_tra(0) - f_ver(0) * d_f_i_d_tra(2), -f_ver(1) * d_f_i_d_tra(0) + f_ver(0) * d_f_i_d_tra(1));

                    jposeType j_pose;
                    j_pose << d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2);

                    jvelType j_vel = j_pose * dt;

                    errorType residual = errorType(f_i) - errorType(kf_i);

                    jpose_buffer.setTexel(j_pose, f_pix_tex(1), f_pix_tex(0));
                    jvel_buffer.setTexel(j_vel, f_pix_tex(1), f_pix_tex(0));
                    e_buffer.setTexel(residual, f_pix_tex(1), f_pix_tex(0));
                    z_buffer.setTexel(f_depth, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderJMapWindow(dataCPU<imageType> &kimage, dataCPU<imageType> &image, dataCPU<jimgType> &d_image_d_pix, SE3f kfTofPose, SE3f fTokfPose, dataCPU<jmapType> &jmap_buffer, dataCPU<errorType> &e_buffer, dataCPU<idsType> &pId_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3 / 4;

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

                    errorType residual = errorType(f_i) - errorType(kf_i);

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

    void renderJMapWindow(dataCPU<imageType> &kimage, dataCPU<imageType> &image, dataCPU<jimgType> &d_image_d_pix, SE3f kfTofPose, SE3f fTokfPose, jvelType fTokfVel, dataCPU<jmapType> &jmap_buffer, dataCPU<errorType> &e_buffer, dataCPU<idsType> &pId_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3 / 4;

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

                    float dt = mesh_vo::line_capture_time * (f_pix(1) - 0.5);
                    SE3f fTokfPoseVel = SE3f::exp(fTokfVel * dt) * fTokfPose;

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

                    vec3f kf_ver = fTokfPoseVel * f_ver;
                    vec3f kf_ray = kf_ver / kf_ver(2);
                    vec2f kf_pix = cam.rayToPix(kf_ray);

                    if (!cam.isPixVisible(kf_pix))
                        continue;

                    imageType kf_i = kimage.get(kf_pix(1), kf_pix(0));
                    imageType f_i = image.getTexel(f_pix_tex(1), f_pix_tex(0));
                    vec2f d_f_i_d_pix = d_image_d_pix.getTexel(f_pix_tex(1), f_pix_tex(0));

                    if (kf_i == kimage.nodata || f_i == image.nodata || d_f_i_d_pix == d_image_d_pix.nodata)
                        continue;

                    errorType residual = errorType(f_i) - errorType(kf_i);

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

    void renderJPoseMapWindow(dataCPU<imageType> &kimage, dataCPU<imageType> &image, dataCPU<jimgType> &d_image_d_pix, SE3f kfTofPose, SE3f fTokfPose, dataCPU<jposeType> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<errorType> &e_buffer, dataCPU<idsType> &pId_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3.0 / 4.0;

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

                    vec6f jpose;
                    jpose << d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2);

                    vec3f d_f_ver_d_kf_depth = kfTofPose.rotationMatrix() * kf_ray;

                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    jmapType jacs = d_f_i_d_kf_depth * f_pol.d_depth_d_vert_depth_param(kf_pol);

                    errorType error = errorType(f_i) - errorType(kf_i);

                    e_buffer.setTexel(error, f_pix_tex(1), f_pix_tex(0));
                    jpose_buffer.setTexel(jpose, f_pix_tex(1), f_pix_tex(0));
                    jmap_buffer.setTexel(jacs, f_pix_tex(1), f_pix_tex(0));
                    pId_buffer.setTexel(ids, f_pix_tex(1), f_pix_tex(0));
                }
            }
        }
    }

    void renderJPoseMapWindow(dataCPU<imageType> &kimage, dataCPU<imageType> &image, dataCPU<jimgType> &d_image_d_pix, SE3f kfTofPose, SE3f fTokfPose, jvelType fTokfVel, dataCPU<jposeType> &jpose_buffer, dataCPU<jmapType> &jmap_buffer, dataCPU<errorType> &e_buffer, dataCPU<idsType> &pId_buffer, cameraType cam, window<float> win)
    {
        int width = e_buffer.width;
        int height = e_buffer.height;

        float min_area = 0.0; //(float(cam.width) / (MESH_WIDTH - 1)) * (float(cam.height) / (MESH_HEIGHT - 1)) * 3.0 / 4.0;

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

                    float dt = mesh_vo::line_capture_time * (f_pix(1) - 0.5);
                    SE3f fTokfPoseVel = SE3f::exp(fTokfVel * dt) * fTokfPose;
                    SE3f kfTofPoseVel = fTokfPoseVel.inverse();

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

                    vec3f kf_ver = fTokfPoseVel * f_ver;
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

                    vec6f jpose;
                    jpose << d_f_i_d_tra(0), d_f_i_d_tra(1), d_f_i_d_tra(2), d_f_i_d_rot(0), d_f_i_d_rot(1), d_f_i_d_rot(2);

                    vec3f d_f_ver_d_kf_depth = kfTofPoseVel.rotationMatrix() * kf_ray;

                    float d_f_i_d_kf_depth = d_f_i_d_f_ver.dot(d_f_ver_d_kf_depth);

                    jmapType jacs = d_f_i_d_kf_depth * f_pol.d_depth_d_vert_depth_param(kf_pol);

                    errorType error = errorType(f_i) - errorType(kf_i);

                    e_buffer.setTexel(error, f_pix_tex(1), f_pix_tex(0));
                    jpose_buffer.setTexel(jpose, f_pix_tex(1), f_pix_tex(0));
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

    /*
    // Draw a single triangle with perspective-correct texture mapping and z-buffer
    void drawTriangle(//std::vector<Color> &framebuffer,
                      //std::vector<float> &zbuffer,
                      dataCPU<float> &framebuffer,
                      dataCPU<float> &zbuffer,
                      //int width,
                      //int height,
                      // Projected screen coords (x, y) and reciprocal w for each vertex
                      //float x0, float y0, float invW0, float z0, float u0, float v0,
                      //float x1, float y1, float invW1, float z1, float u1, float v1,
                      //float x2, float y2, float invW2, float z2, float u2, float v2,
                      //const std::vector<Color> &texture,
                      dataCPU<float> &texture,
                      int texWidth,
                      int texHeight,
                      ShapeTriangleFlat triangle,
                      window<float> win)
    {
        // Compute bounding box of the triangle
        //float minX = std::fmin(std::fmin(x0, x1), x2);
        //float maxX = std::fmax(std::fmax(x0, x1), x2);
        //float minY = std::fmin(std::fmin(y0, y1), y2);
        //float maxY = std::fmax(std::fmax(y0, y1), y2);

        window<float> pol_win = triangle.getScreenBounds();

        // Clamp to screen
        //int startX = (int)std::floor(std::fmax(0.0f, minX));
        //int endX = (int)std::ceil(std::fmin((float)width - 1, maxX));
        //int startY = (int)std::floor(std::fmax(0.0f, minY));
        //int endY = (int)std::ceil(std::fmin((float)height - 1, maxY));

        pol_win.intersect(win);

        // Precompute the area for the triangle (using edge function)
        //float area = edgeFunction(x0, y0, x1, y1, x2, y2);

        //float area = triangle.getScreenArea();

        //for (int py = startY; py <= endY; ++py)
        //{
        //    for (int px = startX; px <= endX; ++px)
        //    {
        for (int py = pol_win.min_y; py <= pol_win.max_y; ++py)
        {
            for (int px = pol_win.min_x; px <= pol_win.max_x; ++px)
            {
                // +0.5f for sampling at pixel center
                //float fx = px + 0.5f;
                //float fy = py + 0.5f;

                vec2f pix(px + 0.5f, py + 0.5f);

                // Compute barycentric coordinates
                //float w0 = edgeFunction(x1, y1, x2, y2, fx, fy);
                //float w1 = edgeFunction(x2, y2, x0, y0, fx, fy);
                //float w2 = edgeFunction(x0, y0, x1, y1, fx, fy);

                triangle.usePixel(pix);


                //if ((w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f && area > 0.0f) ||
                //    (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f && area < 0.0f))
                //{

                if(triangle.isPixInShape())
                {
                    // Perspective-correct interpolation
                    // First convert from "raw" barycentric (w0,w1,w2) to normalized
                    // but also incorporate the per-vertex 1/w.
                    //float alpha = w0 / area;
                    //float beta = w1 / area;
                    //float gamma = w2 / area;

                    // Each vertex had invW = 1.0 / wClip
                    // Weighted sum of reciprocal W:
                    //float invW = alpha * invW0 + beta * invW1 + gamma * invW2;
                    float invW = triangle.getInvDepth();

                    // Final Z in clip space: we do a similar interpolation
                    //float depth = (alpha * z0 * invW0 +
                    //               beta * z1 * invW1 +
                    //               gamma * z2 * invW2) /
                    //              invW;

                    float depth = triangle.getDepth();

                    // Check z-buffer
                    //int index = py * width + px;
                    //if (depth < zbuffer[index])
                    if(depth < zbuffer.getTexel(px, py))
                    {
                        // Compute perspective-correct (u, v)
                        float u = (alpha * u0 * invW0 +
                                   beta * u1 * invW1 +
                                   gamma * u2 * invW2) /
                                  invW;

                        float v = (alpha * v0 * invW0 +
                                   beta * v1 * invW1 +
                                   gamma * v2 * invW2) /
                                  invW;

                        // Update z-buffer
                        //zbuffer[index] = depth;
                        zbuffer.setTexel(depth, px, py);

                        // Sample the texture
                        //Color texColor = sampleTexture(texture, texWidth, texHeight, u, v);
                        //framebuffer[index] = texColor;

                        float texColor = texture.get(u, v);
                        framebuffer.setTexel(texColor, px, py);
                    }
                }
            }
        }
    }
    */

    geometryType scene1;
    geometryType scene2;
    dataCPU<float> z_buffer;
    ThreadPool<mesh_vo::renderer_nthreads> pool;
};
