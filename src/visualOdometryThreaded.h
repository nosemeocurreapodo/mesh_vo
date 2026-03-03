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
// ThreadSafeQueue
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
        cv_.wait(lock, [&] { return closed_ || !queue_.empty(); });
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
        cv_.wait_for(lock, dur, [&] { return closed_ || !queue_.empty(); });
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
// VisualOdometryThreaded
// ------------------------------
class VisualOdometryThreaded
{
public:
    struct PoseUpdate
    {
        SE3f global_pose = SE3f();
        int frame_id = -1;
    };

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
        cam_ = Camera(fx, fy, cx, cy, width, height);
        frameId_.store(0);

        running_.store(true);
        worker_ = std::thread(&VisualOdometryThreaded::workerLoop, this);

        if (debug_log_)
            std::cout << "VO init cam: " << fx << " " << fy << " " << cx << " " << cy << "\n";
    }

    ~VisualOdometryThreaded()
    {
        running_.store(false);
        iQueue_.close();

        init_cv_.notify_all();
        pose_cv_.notify_all();
        kf_sel_cv_.notify_all();
        kf_upd_cv_.notify_all();

        if (worker_.joinable())
            worker_.join();
    }

    void locAndMap(const ImageType *image_data)
    {
        if (!image_data)
            return;

        Texture<ImageType> image_texture(width_, height_, ImageType(-1), image_data);
        image_texture.generate_mipmaps(0);

        iQueue_.push(std::move(image_texture));
    }

    bool waitUntilInitialized(std::chrono::milliseconds timeout = std::chrono::milliseconds(0)) const
    {
        std::unique_lock<std::mutex> lk(init_mtx_);
        if (timeout.count() == 0)
        {
            init_cv_.wait(lk, [&] { return initialized_ || !running_.load(); });
            return initialized_;
        }

        return init_cv_.wait_for(lk, timeout, [&] { return initialized_ || !running_.load(); }) &&
               initialized_;
    }

    // Read-only access to current keyframe without copying
    template <class Fn>
    bool withKeyframe(Fn &&fn) const
    {
        {
            std::unique_lock<std::mutex> lk(init_mtx_);
            if (!initialized_)
                init_cv_.wait(lk, [&] { return initialized_ || !running_.load(); });
        }

        if (!running_.load())
            return false;

        std::shared_lock<std::shared_mutex> lk(kf_mtx_);
        if (!kf_)
            return false;

        fn(*kf_);
        return true;
    }

    // ---- Notifications ----

    bool waitForNextPose(uint64_t &last_seen_seq,
                         PoseUpdate &out,
                         std::chrono::milliseconds timeout = std::chrono::milliseconds(0)) const
    {
        std::unique_lock<std::mutex> lk(pose_mtx_);

        auto pred = [&] {
            return !running_.load() || pose_seq_ > last_seen_seq;
        };

        if (timeout.count() == 0)
        {
            pose_cv_.wait(lk, pred);
        }
        else
        {
            if (!pose_cv_.wait_for(lk, timeout, pred))
                return false;
        }

        if (!running_.load())
            return false;

        out = latest_pose_;
        last_seen_seq = pose_seq_;
        return true;
    }

    bool waitForNextKeyframeSelected(uint64_t &last_seen_seq,
                                     std::chrono::milliseconds timeout = std::chrono::milliseconds(0)) const
    {
        std::unique_lock<std::mutex> lk(kf_sel_mtx_);

        auto pred = [&] {
            return !running_.load() || kf_selected_seq_ > last_seen_seq;
        };

        if (timeout.count() == 0)
        {
            kf_sel_cv_.wait(lk, pred);
        }
        else
        {
            if (!kf_sel_cv_.wait_for(lk, timeout, pred))
                return false;
        }

        if (!running_.load())
            return false;

        last_seen_seq = kf_selected_seq_;
        return true;
    }

    bool waitForNextKeyframeUpdate(uint64_t &last_seen_seq,
                                   std::chrono::milliseconds timeout = std::chrono::milliseconds(0)) const
    {
        std::unique_lock<std::mutex> lk(kf_upd_mtx_);

        auto pred = [&] {
            return !running_.load() || kf_updated_seq_ > last_seen_seq;
        };

        if (timeout.count() == 0)
        {
            kf_upd_cv_.wait(lk, pred);
        }
        else
        {
            if (!kf_upd_cv_.wait_for(lk, timeout, pred))
                return false;
        }

        if (!running_.load())
            return false;

        last_seen_seq = kf_updated_seq_;
        return true;
    }

private:
    void notifyPoseUpdated(const SE3f &pose, int frame_id)
    {
        {
            std::lock_guard<std::mutex> lk(pose_mtx_);
            latest_pose_.global_pose = pose;
            latest_pose_.frame_id = frame_id;
            ++pose_seq_;
        }
        pose_cv_.notify_all();
    }

    void notifyKeyframeSelected()
    {
        {
            std::lock_guard<std::mutex> lk(kf_sel_mtx_);
            ++kf_selected_seq_;
        }
        kf_sel_cv_.notify_all();
    }

    void notifyKeyframeUpdated()
    {
        {
            std::lock_guard<std::mutex> lk(kf_upd_mtx_);
            ++kf_updated_seq_;
        }
        kf_upd_cv_.notify_all();
    }

    void workerLoop()
    {
        if (debug_log_)
            std::cout << "workerLoop\n";

        Mesh screen_mesh = CreateScreenQuad<Mesh>();

        DIDxyRenderer didxy_renderer;
        DepthRenderer depth_renderer;
        ImageRenderer image_renderer;
        NodataReducerCPU nodata_reducer;

        PoseEstimator poseEstimator(width_, height_, false);
        PoseDepthEstimator poseDepthEstimator(width_, height_, debug_log_);

        Texture<ImageType> render_tmp(width_, height_, ImageType(-1));
        Texture<float> depth_tmp(width_, height_, -1.0f);

        FrameWindow frameWindow(width_, height_);

        // --------------------------
        // Initialize from first frame
        // --------------------------
        {
            Mesh mesh = CreateFlatMesh<Mesh>(0.5f, 1.5f, cam_, mesh_vo::mesh_width);
            auto kf = std::make_unique<KeyFrame>(width_, height_, std::move(mesh));

            if (!iQueue_.pop(kf->image()))
                return;

            const int first_id = frameId_.fetch_add(1);
            kf->global_pose() = SE3f();
            kf->global_scale() = 1.0f;
            kf->id() = first_id;

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

            notifyKeyframeSelected();
            notifyKeyframeUpdated();
            notifyPoseUpdated(SE3f(), first_id);

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

            const bool got_frame = iQueue_.wait_pop_for(frame.image(), std::chrono::milliseconds(2));

            if (got_frame)
            {
                const int img_id = frameId_.fetch_add(1);

                for (int lvl = 0; lvl < static_cast<int>(frame.didxy().levels()); ++lvl)
                    didxy_renderer.Render(screen_mesh, lvl, lvl, frame.image(), frame.didxy());

                frame.id() = img_id;

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

                    notifyPoseUpdated(frame.global_pose(), frame.id());

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

                    if (lastMinViewAngle > mesh_vo::last_min_angle || kf_->id() == 0)
                        frameWindow.accept_latest();
                    else
                        continue;

                    if (!frameWindow.full())
                        continue;
                }

                {
                    std::unique_lock<std::shared_mutex> lk(kf_mtx_);
                    if (!kf_)
                        continue;

                    if (debug_log_)
                        std::cout << "Creating/updating keyframe\n";

                    Frame &newkf_frame = frameWindow.middle();
                    std::span<Frame *const> frame_span = frameWindow.window_span_mut();

                    depth_renderer.Render(kf_->mesh(),
                                          kf_->global_pose_to_local(newkf_frame.global_pose()),
                                          cam_, 0, depth_tmp);

                    poseDepthEstimator.update_keyframe(newkf_frame, depth_tmp, *kf_, cam_);
                    poseDepthEstimator.init(frame_span, *kf_, cam_);
                    opt_steps = 0;
                }

                notifyKeyframeSelected();
                notifyKeyframeUpdated();

                if (debug_img_)
                {
                    std::shared_lock<std::shared_mutex> lk(kf_mtx_);
                    if (kf_)
                    {
                        depth_renderer.Render(kf_->mesh(), SE3f(), cam_, 1, depth_tmp);
                        depth_mat = DownloadTextureToMat(depth_tmp, 1);
                        SaveDebugImage(depth_mat, "map_" + std::to_string(kf_->id()) + "_init_depth.png");
                    }
                }
            }
            else
            {
                if (!frameWindow.full() || poseDepthEstimator.converged())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                std::span<Frame *const> frame_span = frameWindow.window_span_mut();

                {
                    std::unique_lock<std::shared_mutex> lk(kf_mtx_);
                    if (!kf_)
                        continue;

                    ++opt_steps;
                    poseDepthEstimator.step(frame_span, *kf_, cam_);
                }

                notifyKeyframeUpdated();

                if (debug_img_)
                {
                    std::shared_lock<std::shared_mutex> lk(kf_mtx_);
                    if (kf_)
                    {
                        depth_renderer.Render(kf_->mesh(), SE3f(), cam_, 1, depth_tmp);
                        depth_mat = DownloadTextureToMat(depth_tmp, 1);
                        SaveDebugImage(depth_mat, "map_" + std::to_string(kf_->id()) + "_opt_depth_" + std::to_string(opt_steps) + ".png");
                    }
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

    // Keyframe
    mutable std::shared_mutex kf_mtx_;
    std::unique_ptr<KeyFrame> kf_;

    // Input queue
    ThreadSafeQueue<Texture<ImageType>> iQueue_;

    // Camera + params
    Camera cam_;
    int width_{0};
    int height_{0};
    std::atomic<int> frameId_{0};

    // Pose event
    mutable std::mutex pose_mtx_;
    mutable std::condition_variable pose_cv_;
    PoseUpdate latest_pose_;
    uint64_t pose_seq_{0};

    // Keyframe selected event
    mutable std::mutex kf_sel_mtx_;
    mutable std::condition_variable kf_sel_cv_;
    uint64_t kf_selected_seq_{0};

    // Keyframe updated event
    mutable std::mutex kf_upd_mtx_;
    mutable std::condition_variable kf_upd_cv_;
    uint64_t kf_updated_seq_{0};

    bool debug_log_{true};
    bool debug_img_{false};
};