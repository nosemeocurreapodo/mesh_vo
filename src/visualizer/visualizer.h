// Visualizer.hpp
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include <atomic>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include <optional>

#include "common/types.h"
#include "common/frame.h"
#include "common/keyframe.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

class Visualizer
{
public:
  struct Options
  {
    bool show_trajectory = true;
    bool show_frame_image = true;
    bool show_keyframe_image = false;
    bool show_keyframe_depth = true;

    bool use_viz3d = true;            // 3D window via cv::viz
    bool show_frusta = true;          // show camera frustum widgets
    bool show_depth_as_cloud = false; // optional: depth->point cloud in 3D

    size_t window_width = 1280;
    size_t window_height = 720;

    size_t max_keyframes = 200; // keep bounded
    size_t max_traj_points = 5000;
  };

  explicit VisualizerOpenCV(Options opt = {});
  ~VisualizerOpenCV();

  void start();
  void stop();

  void pushFrame(const Frame &f);
  void pushKeyframe(const KeyFrame &kf);

  void setOptions(const Options &opt);
  Options options() const;

private:
  // --- Thread loop
  void viewerLoop();

  // --- Rendering helpers
  void render2D(const std::optional<Frame> &f,
                const std::optional<KeyFrame> &kf);
  void render3D(const std::optional<Frame> &f);

  // --- Mouse control in 3D viewer
  struct ViewState
  {
    // Orbit-style camera around a target point
    cv::Point3d target = {0, 0, 0};
    double distance = 5.0;
    double yaw = 0.0;     // radians
    double pitch = -0.35; // radians
    bool dragging_left = false;
    bool dragging_right = false;
    cv::Point last_mouse = {0, 0};
  };

  static void mouseCallback(const cv::viz::MouseEvent &ev, void *cookie);
  void onMouse(const cv::viz::MouseEvent &ev);

  // --- Pose conversion
  static cv::Affine3d eigenToCvAffine(const Eigen::Isometry3d &T);

  // --- Depth visualization
  static cv::Mat colorizeDepth(const cv::Mat &depth);

  // --- Internal state
  mutable std::mutex mtx_;
  Options opt_;

  std::optional<Frame> latest_frame_;
  std::optional<KeyFrame> latest_keyframe_;

  std::deque<KeyFrame> keyframes_;       // bounded
  std::vector<cv::Point3d> traj_points_; // bounded

  std::atomic<bool> running_{false};
  std::thread th_;

  // 3D viewer
  std::unique_ptr<cv::viz::Viz3d> win_;
  ViewState view_;

  // Widgets bookkeeping (avoid re-adding)
  bool widgets_initialized_ = false;

  ImageRenderer image_renderer_;
};
