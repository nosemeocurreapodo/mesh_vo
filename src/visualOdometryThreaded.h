#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
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

    // Push by value (enables move from caller)
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

    // Non-blocking pop
    bool try_pop(T &out)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
            return false;
        out = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    // Blocking pop (returns false if closed and empty)
    bool pop(T &out)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]
                 { return closed_ || !queue_.empty(); });
        if (queue_.empty())
            return false; // closed + empty
        out = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    // Timed pop (returns false on timeout or closed+empty)
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

    // Peek (copies) - generally avoid for large T
    bool try_peek(T &out) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
            return false;
        out = queue_.front(); // copy
        return true;
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
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
// VisualOdometryThreaded (actually: 1 worker thread)
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
        // #ifdef COMPILE_GL
        //         InitEGL(); // assumes this creates/makes current a context on this thread; be careful with multi-thread GL
        // #endif
        frameId_.store(0);

        // FIX: you had Camera(fy, fy, ...) - use fx, fy
        cam_ = Camera(fx, fy, cx, cy, width, height);

        running_.store(true);
        worker_ = std::thread(&VisualOdometryThreaded::workerLoop, this);

        if (debug_log_)
            std::cout << "init vo, cam : " << fx << " " << fy << " " << cx << " " << cy << std::endl;
    }

    ~VisualOdometryThreaded()
    {
        // Signal stop
        running_.store(false);

        // Wake any waits
        iQueue_.close();
        {
            std::lock_guard<std::mutex> lk(init_mtx_);
            // nothing else
        }
        init_cv_.notify_all();

        if (worker_.joinable())
            worker_.join();
    }

    // Initializes first keyframe and releases worker thread from init wait
    void flatInit(const ImageType *image_data)
    {
        if (debug_log_)
            std::cout << "flatInit" << std::endl;

        Mesh screen_mesh = CreateScreenQuad<Mesh>();

        DIDxyRenderer didxy_renderer;

        Texture<ImageType> image_texture(width_, height_, ImageType(-1), image_data);
        image_texture.generate_mipmaps(0);

        Texture<Vec3f> didxy_texture(width_, height_, Vec3f(0.0f, 0.0f, 0.0f));
        for (int lvl = 0; lvl < static_cast<int>(didxy_texture.levels()); ++lvl)
            didxy_renderer.Render(screen_mesh, lvl, lvl, image_texture, didxy_texture);

        Mesh mesh = CreateFlatMesh<Mesh>(0.5f, 1.5f, cam_, mesh_vo::mesh_width);

        const int id = frameId_.fetch_add(1);
        KeyFrame kframe(image_texture, didxy_texture, SE3f(), mesh, 1.0f, id);

        // Publish latest keyframe (for ROS)
        {
            std::lock_guard<std::mutex> lk(latest_kf_mtx_);
            latest_kf_ = std::move(kframe);
            has_latest_kf_ = true;
        }

        // Wake worker
        {
            std::lock_guard<std::mutex> lk(init_mtx_);
            initialized_ = true;
        }
        init_cv_.notify_all();
    }

    // Push new frame image for processing (copies into Texture)
    void locAndMap(const ImageType *image_data)
    {
        if (debug_log_)
            std::cout << "locAndMap" << std::endl;

        if (!image_data)
        {
            if (debug_log_)
                std::cout << "no image_data" << std::endl;

            return;
        }

        Texture<ImageType> image_texture(width_, height_, ImageType(-1), image_data);
        image_texture.generate_mipmaps(0);

        // Move into queue (no heavy copies if Texture is movable)
        iQueue_.push(std::move(image_texture));
    }

    // Non-blocking "latest keyframe" fetch (recommended for ROS thread)
    bool tryGetLatestKeyframe(KeyFrame &out) const
    {
        std::lock_guard<std::mutex> lk(latest_kf_mtx_);
        if (!has_latest_kf_)
            return false;
        out = latest_kf_; // copy (KeyFrame likely owns textures; if heavy, consider shared_ptr snapshot)
        return true;
    }

    // Blocking get (waits until initialized at least once)
    KeyFrame getKeyframe() const
    {
        if (debug_log_)
            std::cout << "getKeyframe" << std::endl;

        // Wait for first init
        {
            std::unique_lock<std::mutex> lk(init_mtx_);
            init_cv_.wait(lk, [&]
                          { return initialized_ || !running_.load(); });
        }

        std::lock_guard<std::mutex> lk(latest_kf_mtx_);
        return latest_kf_;
    }

    // Best-effort idle indicator
    bool isIdle() const
    {
        return iQueue_.empty();
    }

private:
    void workerLoop()
    {
        if (debug_log_)
            std::cout << "workerLoop" << std::endl;

        // Wait for init
        {
            std::unique_lock<std::mutex> lk(init_mtx_);
            init_cv_.wait(lk, [&]
                          { return initialized_ || !running_.load(); });
        }
        if (!running_.load())
            return;

        // Setup renderers/estimators
        Mesh screen_mesh = CreateScreenQuad<Mesh>();

        DIDxyRenderer didxy_renderer;
        DepthRenderer depth_renderer;
        ImageRenderer image_renderer;
        NodataReducerCPU nodata_reducer;

        PoseEstimator poseEstimator(width_, height_, false);
        PoseDepthEstimator poseDepthEstimator(width_, height_, debug_log_);

        Texture<ImageType> image_texture(width_, height_, ImageType(-1));
        Texture<Vec3f> didxy_texture(width_, height_, Vec3f(0.0f, 0.0f, 0.0f));
        Texture<float> depth_texture(width_, height_, -1.0f);

        FrameWindow frameWindow(width_, height_);

        // Local current keyframe state lives here
        KeyFrame kframe = getKeyframe();

        tic_toc tt;
        cv::Mat image_mat, depth_mat, ref_mat, l2_mat;

        int opt_steps = 0;

        constexpr float PI = 3.14159265358979323846f;

        while (running_.load())
        {
            // Try to get a new image, but don't block forever.
            // If no new frame, we spend time mapping (idle optimization).
            Frame &frame = frameWindow.latest();
            const bool got_frame = iQueue_.wait_pop_for(frame.image(), std::chrono::milliseconds(2));

            if (got_frame)
            {
                if (debug_log_)
                    std::cout << "got_frame" << std::endl;

                // Precompute gradients
                for (int lvl = 0; lvl < static_cast<int>(didxy_texture.levels()); ++lvl)
                    didxy_renderer.Render(screen_mesh, lvl, lvl, frame.image(), frame.didxy());

                poseEstimator.guess(frame, kframe);

                if (debug_img_)
                {
                    image_renderer.Render(kframe.mesh(),
                                          frame.local_pose(),
                                          frame.local_exposure(),
                                          cam_,
                                          1, 1,
                                          kframe.image(), image_texture);

                    image_mat = DownloadTextureToMat(image_texture, 1);
                    ref_mat = DownloadTextureToMat(frame.image(), 1);
                    l2_mat = ref_mat - image_mat;
                    SaveDebugImage(l2_mat, "loc_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + "_init_l2.png");
                }

                // Localize
                tt.tic();
                poseEstimator.estimate(frame, kframe, cam_);

                if (debug_log_)
                    std::cout << "localization time " << tt.toc() << std::endl;

                if (debug_img_)
                {
                    image_renderer.Render(kframe.mesh(),
                                          frame.local_pose(),
                                          frame.local_exposure(),
                                          cam_,
                                          1, 1,
                                          kframe.image(), image_texture);

                    image_mat = DownloadTextureToMat(image_texture, 1);
                    l2_mat = ref_mat - image_mat;
                    SaveDebugImage(l2_mat, "loc_" + std::to_string(kframe.id()) + "_" + std::to_string(frame.id()) + "_opt_l2.png");
                }

                // Select whether to keep the frame
                float lastMinViewAngle = PI;
                for (const Frame *f : frameWindow.window_span_mut())
                {
                    float lastViewAngle = kframe.meanViewAngle(f->local_pose(), frame.local_pose(), cam_);
                    if (lastViewAngle < lastMinViewAngle)
                        lastMinViewAngle = lastViewAngle;
                }

                if (lastMinViewAngle > mesh_vo::last_min_angle || kframe.id() == 0)
                {
                    frameWindow.accept_latest();
                }
                else
                {
                    // Drop frame
                    continue;
                }

                // Need enough frames before considering new keyframe / mapping init
                if (!frameWindow.full())
                    continue;

                // Evaluate whether to create a new keyframe
                float keyframeViewAngle = kframe.meanViewAngle(SE3f(), frame.local_pose(), cam_);

                Error nodata;
                nodata_reducer.reduce(1, image_texture, nodata);
                float pnodata = nodata.getError() / float(image_texture.width(1) * image_texture.height(1));
                float viewPercent = 1.0f - pnodata;

                // Keep current keyframe if still good enough
                if (kframe.id() != 0 &&
                    viewPercent > mesh_vo::min_view_perc &&
                    keyframeViewAngle < mesh_vo::key_max_angle)
                {
                    continue;
                }

                if (debug_log_)
                    std::cout << "Creating new keyframe because viewPercent=" << viewPercent
                              << " keyframeViewAngle=" << keyframeViewAngle << std::endl;

                // "Finalize" current keyframe (history). If you want, store it somewhere; currently we just overwrite latest.
                // Select new keyframe as middle frame
                frameWindow.promote_middle_to_keyframe();
                Frame &newkframe = frameWindow.keyframe_frame();
                std::span<Frame *const> frame_span = frameWindow.window_span_mut();

                poseDepthEstimator.changeKeyframe(newkframe, frame_span, kframe, cam_);
                poseDepthEstimator.init(frame_span, kframe, cam_);
                opt_steps = 0;

                if (debug_img_)
                {
                    /////////////////// Debug ///////////////////
                    depth_renderer.Render(kframe.mesh(),
                                          SE3f(), cam_, 1, depth_texture);
                    depth_mat = DownloadTextureToMat(depth_texture, 1);
                    SaveDebugImage(depth_mat, "map_" + std::to_string(kframe.id()) + "_init_depth.png");

                    for (Frame *f : frame_span)
                    {
                        image_renderer.Render(kframe.mesh(),
                                              f->local_pose(),
                                              f->local_exposure(),
                                              cam_,
                                              1, 1,
                                              kframe.image(), image_texture);

                        image_mat = DownloadTextureToMat(image_texture, 1);
                        ref_mat = DownloadTextureToMat(f->image(), 1);
                        l2_mat = ref_mat - image_mat;
                        SaveDebugImage(l2_mat, "map_" + std::to_string(kframe.id()) + "_" + std::to_string(f->id()) + "_ini_l2.png");
                    }
                    ///////////////////////////
                }

                // Publish latest keyframe snapshot for ROS consumers
                {
                    std::lock_guard<std::mutex> lk(latest_kf_mtx_);
                    latest_kf_ = kframe;
                    has_latest_kf_ = true;
                }
            }
            else
            {
                // No new frame
                // if (!doMapping_)
                //{
                //    // Avoid busy-spin
                //    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                //    continue;
                //}

                if (!frameWindow.full())
                {
                    // Still warming up; avoid burning CPU
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                std::span<Frame* const> frame_span = frameWindow.window_span_mut();

                // One mapping iteration per idle cycle (your proposed scheduler)
                opt_steps++;
                poseDepthEstimator.step(frame_span, kframe, cam_);

                // Publish latest keyframe snapshot (mesh updated inside kframe)
                {
                    std::lock_guard<std::mutex> lk(latest_kf_mtx_);
                    latest_kf_ = kframe;
                    has_latest_kf_ = true;
                }

                if (debug_img_)
                {
                    /////////////////// Debug ///////////////////
                    depth_renderer.Render(kframe.mesh(),
                                          SE3f(), cam_, 1, depth_texture);
                    depth_mat = DownloadTextureToMat(depth_texture, 1);
                    SaveDebugImage(depth_mat, "map_" + std::to_string(kframe.id()) + "_opt_depth_" + std::to_string(opt_steps) + ".png");

                    for (Frame* f : frame_span)
                    {
                        image_renderer.Render(kframe.mesh(),
                                              f->local_pose(),
                                              f->local_exposure(),
                                              cam_,
                                              1, 1,
                                              kframe.image(), image_texture);

                        image_mat = DownloadTextureToMat(image_texture, 1);
                        ref_mat = DownloadTextureToMat(f->image(), 1);
                        l2_mat = ref_mat - image_mat;
                        SaveDebugImage(l2_mat, "map_" + std::to_string(kframe.id()) + "_" + std::to_string(f->id()) + "_opt_l2_" + std::to_string(opt_steps) + ".png");
                    }
                    ///////////////////////////
                }
            }
        }
    }

private:
    // Worker thread control
    std::atomic<bool> running_{false};
    std::thread worker_;

    // Init barrier (worker waits until first keyframe exists)
    mutable std::mutex init_mtx_;
    mutable std::condition_variable init_cv_;
    bool initialized_{false};

    // Latest keyframe snapshot (for ROS thread)
    mutable std::mutex latest_kf_mtx_;
    KeyFrame latest_kf_;
    bool has_latest_kf_{false};

    // Input queue of textures
    ThreadSafeQueue<Texture<ImageType>> iQueue_;

    // Camera + params
    Camera cam_;
    int width_{0};
    int height_{0};
    std::atomic<int> frameId_{0};

    bool debug_log_{true};
    bool debug_img_{false};
};