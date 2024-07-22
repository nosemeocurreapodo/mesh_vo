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
#include "cpu/IndexThreadReduce.h"
#include "params.h"

// Simple thread pool example
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

class ThreadPool
{
public:
    ThreadPool(size_t numThreads = std::thread::hardware_concurrency());
    ~ThreadPool();
    void enqueue(std::function<void()> task);
    void waitUntilDone();

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    std::condition_variable doneCondition;
    bool stop;
    size_t activeTasks;
    void workerThread();
};

/*
int main() {
    ThreadPool pool(4);
    for (int i = 0; i < 8; ++i) {
        pool.enqueue([i] { std::cout << "Task " << i << " executed\n"; });
    }
    return 0;
}
*/

class renderCPU
{
public:
    renderCPU(unsigned int width, unsigned int height)
        : z_buffer(width, height, -1),
          pool()
    {
    }

    void renderIdepth(SceneBase &scene, camera &cam, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderImage(SceneBase &scene, camera &cam, frameCPU &kframe, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);
    void renderImage(dataCPU<float> &poseIdepth, camera &cam, frameCPU &kframe, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl);

    void renderDebug(SceneBase &scene, camera &cam, frameCPU &pose, dataCPU<float> &buffer, int lvl);

    void renderJPose(SceneBase &scene, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<std::array<float, 3>> &jtra_buffer, dataCPU<std::array<float, 3>> &jrot_buffer, dataCPU<float> &e_buffer, int lvl);
    void renderJPose(dataCPU<float> &frame2Idepth, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<std::array<float, 3>> &jtra_buffer, dataCPU<std::array<float, 3>> &jrot_buffer, dataCPU<float> &e_buffer, int lvl);
    void renderJMap(SceneBase &scene, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<std::array<float, 3>> &j_buffer, dataCPU<float> &e_buffer, dataCPU<std::array<int, 3>> &pId_buffer, int lvl);

    void renderJPoseMap(SceneBase &scene, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<std::array<float, 3>> &j1_buffer, dataCPU<std::array<float, 3>> &j2_buffer, dataCPU<std::array<float, 3>> &j3_buffer, dataCPU<float> &e_buffer, dataCPU<std::array<int, 3>> &pId_buffer, int lvl);

    void renderJPoseParallel(dataCPU<float> &frame2Idepth, camera &cam, frameCPU &frame1, frameCPU &frame2, dataCPU<std::array<float, 3>> &jtra_buffer, dataCPU<std::array<float, 3>> &jrot_buffer, dataCPU<float> &e_buffer, int lvl)
    {
        int divi = 16;

        std::array<int, 2> windowSize;
        windowSize[0] = cam.width;
        windowSize[1] = cam.height / divi;

        for (int ty = 0; ty < divi; ty++)
        {
            for (int tx = 0; tx < 1; tx++)
            {
                int min_x = tx * windowSize[0];
                int max_x = (tx + 1) * windowSize[0];
                int min_y = ty * windowSize[1];
                int max_y = (ty + 1) * windowSize[1];

                camera cam_window = cam;
                cam_window.setWindow(min_x, max_x, min_y, max_y);

                //renderJPoseWindow(&frame2Idepth, cam_window, &frame1, &frame2, &jtra_buffer, &jrot_buffer, &e_buffer, lvl);
                pool.enqueue(std::bind(&renderCPU::renderJPoseWindow, this, &frame2Idepth, cam_window, &frame1, &frame2, &jtra_buffer, &jrot_buffer, &e_buffer, lvl));
            }
        }

        pool.waitUntilDone();
    }

    void renderIdepthParallel(SceneBase &scene, camera &cam, Sophus::SE3f &pose, dataCPU<float> &buffer, int lvl)
    {
        z_buffer.set(z_buffer.nodata, lvl);

        frameMesh = scene.clone();
        frameMesh->transform(pose);

        int divi = 2;

        std::array<int, 2> windowSize;
        windowSize[0] = cam.width / divi;
        windowSize[1] = cam.height / divi;

        for (int ty = 0; ty < divi; ty++)
        {
            for (int tx = 0; tx < divi; tx++)
            {
                int min_x = tx * windowSize[0];
                int max_x = (tx + 1) * windowSize[0];
                int min_y = ty * windowSize[1];
                int max_y = (ty + 1) * windowSize[1];

                camera cam_window = cam;
                cam_window.setWindow(min_x, max_x, min_y, max_y);

                //renderIdepthWindow(cam_window, &buffer, lvl);
                pool.enqueue(std::bind(&renderCPU::renderIdepthWindow, this, cam_window, &buffer, lvl));
            }
        }

        pool.waitUntilDone();
    }

private:
    void renderJPoseWindow(dataCPU<float> *frameIdepth, camera cam, frameCPU *kframe, frameCPU *frame, dataCPU<std::array<float, 3>> *jtra_buffer, dataCPU<std::array<float, 3>> *jrot_buffer, dataCPU<float> *e_buffer, int lvl)
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

                float residual = (f_i - kf_i);

                jtra_buffer->set(d_f_i_d_tra, y, x, lvl);
                jrot_buffer->set(d_f_i_d_rot, y, x, lvl);
                e_buffer->set(residual, y, x, lvl);
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

    // IndexThreadReduce<Error> errorTreadReduce;
    // IndexThreadReduce<HGMapped> hgMappedTreadReduce;
};
