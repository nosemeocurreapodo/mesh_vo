#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "mpdr/common/types.h"
#include "mpdr/common/mesh_helpers.h"
#include "mpdr/common/helpers.h"

#include "mpdr/backends/cpu/texturecpu.h"
#include "mpdr/backends/cpu/meshcpu.h"
#include "mpdr/backends/cpu/renderercpu.h"

#ifdef COMPILE_GL
#include "mpdr/backends/gl/devicegl_glad.h"
#include "mpdr/backends/gl/texturegl.h"
#include "mpdr/backends/gl/meshgl.h"
#include "mpdr/backends/gl/renderergl.h"
#endif

#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"
#include "common/FrameWindow.h"

#include "poseEstimator.h"
#include "poseDepthEstimator.h"
#include "utils/tictoc.h"

// ------------------------------
// ThreadSafeQueue (move-capable + close + timed wait)
// ------------------------------
template <typename T>
class ThreadSafeQueue
{
public:
    ThreadSafeQueue() = default;

    void push(T value)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_)
                return;
            queue_.push(std::move(value));
        }
        cv_.notify_one();
    }

    bool try_pop(T &out)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
            return false;
        out = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool pop(T &out)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]
                 { return closed_ || !queue_.empty(); });
        if (queue_.empty())
            return false;
        out = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    template <class Rep, class Period>
    bool wait_pop_for(T &out, const std::chrono::duration<Rep, Period> &dur)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, dur, [&]
                     { return closed_ || !queue_.empty(); });
        if (queue_.empty())
            return false;
        out = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cv_;
    bool closed_{false};
};

// ------------------------------
// VisualOdometryThreaded (1 worker thread)
// ------------------------------
class VisualOdometryThreaded
{
public:
    VisualOdometryThreaded(float fx, float fy, float cx, float cy,
                           int width, int height,
                           bool debug_log = true,
                           bool debug_img = false)
        : width_(width),
          height_(height),
          debug_log_(debug_log),
          debug_img_(debug_img)
    {
        frameId_.store(0);
        cam_ = Camera(fx, fy, cx, cy, width, height);

        running_.store(true);
        worker_ = std::thread(&VisualOdometryThreaded::workerLoop, this);

        if (debug_log_)
            std::cout << "VO init cam: " << fx << " " << fy << " " << cx << " " << cy << "\n";
    }

    ~VisualOdometryThreaded()
    {
        running_.store(false);
        iQueue_.close();

        {
            std::lock_guard<std::mutex> lk(init_mtx_);
        }
        init_cv_.notify_all();

        if (worker_.joinable())
            worker_.join();
    }

    // Push new image (still builds a Texture here; you can later queue pointer jobs instead)
    void locAndMap(const ImageType *image_data)
    {
        if (!image_data)
            return;

        Texture<ImageType> image_texture(width_, height_, ImageType(-1), image_data);
        image_texture.generate_mipmaps(0);

        iQueue_.push(std::move(image_texture));
    }

    // Wait until the first keyframe exists (optional helper for caller)
    bool waitUntilInitialized(std::chrono::milliseconds timeout = std::chrono::milliseconds(0)) const
    {
        std::unique_lock<std::mutex> lk(init_mtx_);
        if (timeout.count() == 0)
        {
            init_cv_.wait(lk, [&]
                          { return initialized_ || !running_.load(); });
            return initialized_;
        }
        return init_cv_.wait_for(lk, timeout, [&]
                                 { return initialized_ || !running_.load(); }) &&
               initialized_;
    }

    // ---- Keyframe access without copying ----
    // Runs 'fn(kf)' under a shared lock. fn must not store references beyond the call.
    template <class Fn>
    bool withKeyframe(Fn &&fn) const
    {
        // Wait until initialized
        {
            std::unique_lock<std::mutex> lk(init_mtx_);
            if (!initialized_)
                init_cv_.wait(lk, [&]
                              { return initialized_ || !running_.load(); });
        }
        if (!running_.load())
            return false;

        std::shared_lock<std::shared_mutex> lk(kf_mtx_);
        if (!kf_)
            return false;
        fn(*kf_);
        return true;
    }

private:
    void workerLoop()
    {
        if (debug_log_)
            std::cout << "workerLoop\n";

        // Setup renderers/estimators
        Mesh screen_mesh = CreateScreenQuad<Mesh>();

        DIDxyRenderer didxy_renderer;
        DepthRenderer depth_renderer;
        ImageRenderer image_renderer;
        NodataReducerCPU nodata_reducer;

        PoseEstimator poseEstimator(width_, height_, false);
        PoseDepthEstimator poseDepthEstimator(width_, height_, debug_log_);

        Texture<ImageType> render_tmp(width_, height_, ImageType(-1)); // scratch for debug renders
        Texture<float> depth_tmp(width_, height_, -1.0f);

        FrameWindow frameWindow(width_, height_);
        int img_id = 0;

        // --------------------------
        // Initialize from first frame
        // --------------------------
        {
            // Frame &kf_frame = frameWindow.keyframe_frame();

            // create initial mesh
            Mesh mesh = CreateFlatMesh<Mesh>(0.5f, 1.5f, cam_, mesh_vo::mesh_width);

            // build initial keyframe (ASSUMPTION: KeyFrame can be built from a Frame + Mesh)
            // If your KeyFrame ctor differs, adjust this line accordingly.
            auto kf = std::make_unique<KeyFrame>(width_, height_, std::move(mesh));

            // block until first image arrives (or shutdown)
            if (!iQueue_.pop(kf->image()))
                return;

            kf->global_pose() = SE3f();
            kf->global_scale() = 1.0;
            kf->id() = img_id;

            float md = mean_depth(kf->mesh());
            kf->scale_mesh(md / mesh_vo::mapping_mean_depth);

            {
                std::unique_lock<std::shared_mutex> lk(kf_mtx_);
                kf_ = std::move(kf);
            }

            {
                std::lock_guard<std::mutex> lk(init_mtx_);
                initialized_ = true;
            }
            init_cv_.notify_all();

            if (debug_log_)
                std::cout << "Initialized first keyframe\n";
        }

        tic_toc tt;
        cv::Mat image_mat, depth_mat, ref_mat, l2_mat;
        int opt_steps = 0;
        constexpr float PI = 3.14159265358979323846f;

        while (running_.load())
        {
            Frame &frame = frameWindow.latest();

            // Move a new image texture into the preallocated latest frame slot (no default ctor needed)
            const bool got_frame = iQueue_.wait_pop_for(frame.image(), std::chrono::milliseconds(2));

            if (got_frame)
            {
                img_id++;

                // Precompute gradients for this frame
                for (int lvl = 0; lvl < static_cast<int>(frame.didxy().levels()); ++lvl)
                    didxy_renderer.Render(screen_mesh, lvl, lvl, frame.image(), frame.didxy());

                frame.id() = img_id;

                // Tracking uses shared access to keyframe
                {
                    std::shared_lock<std::shared_mutex> lk(kf_mtx_);
                    if (!kf_)
                        continue;

                    poseEstimator.guess(frame, *kf_);

                    if (debug_img_)
                    {
                        image_renderer.Render(kf_->mesh(),
                                              kf_->global_pose_to_local(frame.global_pose()),
                                              frame.local_exposure(),
                                              cam_,
                                              1, 1,
                                              kf_->image(), render_tmp);

                        image_mat = DownloadTextureToMat(render_tmp, 1);
                        ref_mat = DownloadTextureToMat(frame.image(), 1);
                        l2_mat = ref_mat - image_mat;
                        SaveDebugImage(l2_mat, "loc_" + std::to_string(kf_->id()) + "_" + std::to_string(frame.id()) + "_init_l2.png");
                    }

                    tt.tic();
                    poseEstimator.estimate(frame, *kf_, cam_);

                    if (debug_log_)
                        std::cout << "localization time " << tt.toc() << "\n";

                    if (debug_img_)
                    {
                        image_renderer.Render(kf_->mesh(),
                                              kf_->global_pose_to_local(frame.global_pose()),
                                              frame.local_exposure(),
                                              cam_,
                                              1, 1,
                                              kf_->image(), render_tmp);

                        image_mat = DownloadTextureToMat(render_tmp, 1);
                        l2_mat = ref_mat - image_mat;
                        SaveDebugImage(l2_mat, "loc_" + std::to_string(kf_->id()) + "_" + std::to_string(frame.id()) + "_opt_l2.png");
                    }
                }

                // Decide whether to keep frame (also read-only on keyframe)
                float lastMinViewAngle = PI;
                {
                    std::shared_lock<std::shared_mutex> lk(kf_mtx_);
                    if (!kf_)
                        continue;

                    for (const Frame *f : frameWindow.window_span_mut())
                    {
                        float lastViewAngle = kf_->meanViewAngle(f->global_pose(), frame.global_pose(), cam_);
                        if (lastViewAngle < lastMinViewAngle)
                            lastMinViewAngle = lastViewAngle;
                    }

                    // float keyframeViewAngle = kf_->meanViewAngle(kf_->global_pose(), frame.global_pose(), cam_);

                    // image_renderer.Render(kf_->mesh(),
                    //                       kf_->global_pose_to_local(frame.global_pose()),
                    //                       frame.local_exposure(),
                    //                       cam_,
                    //                       1, 1,
                    //                       kf_->image(), render_tmp);

                    // Error nodata;
                    // nodata_reducer.reduce(1, render_tmp, nodata);
                    // float pnodata = nodata.getError() / float(render_tmp.width(1) * render_tmp.height(1));
                    // float viewPercent = 1.0f - pnodata;

                    if (lastMinViewAngle > mesh_vo::last_min_angle || kf_->id() == 0)
                        frameWindow.accept_latest();
                    else
                        continue;

                    if (!frameWindow.full())
                        continue;

                    // Evaluate keyframe switch
                    // if (kf_->id() != 0 &&
                    //    viewPercent > mesh_vo::min_view_perc &&
                    //    keyframeViewAngle < mesh_vo::key_max_angle)
                    //{
                    //    continue;
                    //}
                }

                // Keyframe promotion + mapping init needs exclusive access to keyframe
                {
                    std::unique_lock<std::shared_mutex> lk(kf_mtx_);
                    if (!kf_)
                        continue;

                    if (debug_log_)
                        std::cout << "Creating new keyframe\n";

                    // frameWindow.promote_middle_to_keyframe();
                    Frame &newkf_frame = frameWindow.middle();
                    std::span<Frame *const> frame_span = frameWindow.window_span_mut();

                    depth_renderer.Render(kf_->mesh(),
                                          kf_->global_pose_to_local(newkf_frame.global_pose()),
                                          cam_, 0, depth_tmp);

                    poseDepthEstimator.update_keyframe(newkf_frame, depth_tmp, *kf_, cam_);
                    poseDepthEstimator.init(frame_span, *kf_, cam_);
                    opt_steps = 0;

                    if (debug_img_)
                    {
                        depth_renderer.Render(kf_->mesh(), SE3f(), cam_, 1, depth_tmp);
                        depth_mat = DownloadTextureToMat(depth_tmp, 1);
                        SaveDebugImage(depth_mat, "map_" + std::to_string(kf_->id()) + "_init_depth.png");
                    }
                }
            }
            else
            {
                // No new frame -> mapping iteration if window full
                if (!frameWindow.full() || poseDepthEstimator.converged())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                std::span<Frame *const> frame_span = frameWindow.window_span_mut();

                std::unique_lock<std::shared_mutex> lk(kf_mtx_);
                if (!kf_)
                    continue;

                opt_steps++;
                poseDepthEstimator.step(frame_span, *kf_, cam_);

                if (debug_img_)
                {
                    depth_renderer.Render(kf_->mesh(), SE3f(), cam_, 1, depth_tmp);
                    depth_mat = DownloadTextureToMat(depth_tmp, 1);
                    SaveDebugImage(depth_mat, "map_" + std::to_string(kf_->id()) + "_opt_depth_" + std::to_string(opt_steps) + ".png");
                }
            }
        }
    }

private:
    // Worker
    std::atomic<bool> running_{false};
    std::thread worker_;

    // Init barrier
    mutable std::mutex init_mtx_;
    mutable std::condition_variable init_cv_;
    bool initialized_{false};

    // Keyframe (non-copyable) stored by pointer
    mutable std::shared_mutex kf_mtx_;
    std::unique_ptr<KeyFrame> kf_;

    // Input queue
    ThreadSafeQueue<Texture<ImageType>> iQueue_;

    // Camera + params
    Camera cam_;
    int width_{0};
    int height_{0};
    std::atomic<int> frameId_{0};

    bool debug_log_{true};
    bool debug_img_{false};
};