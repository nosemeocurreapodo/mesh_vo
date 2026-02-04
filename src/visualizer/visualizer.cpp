// Visualizer.cpp
#include "visualizer.h"
#include <cmath>

Visualizer::Visualizer(Options opt)
    : opt_(std::move(opt))
{
}

Visualizer::~Visualizer() { stop(); }

void Visualizer::star()
{
    if (running_.exchange(true))
        return;

    if (opt_.use_viz3d)
    {
        win_ = std::make_unique<cv::viz::Viz3d>("VSLAM Viewer");
        win_->registerMouseCallback(&VisualizerOpenCV::mouseCallback, this);
        win_->showWidget("axes", cv::viz::WCoordinateSystem(0.5));
        widgets_initialized_ = false;
    }

    th_ = std::thread(&VisualizerOpenCV::viewerLoop, this);
}

void Visualizer::stop()
{
    if (!running_.exchange(false))
        return;
    if (th_.joinable())
        th_.join();
    if (win_)
        win_->close();
    win_.reset();
}

void Visualizer::setOptions(const Options &opt)
{
    std::lock_guard<std::mutex> lk(mtx_);
    opt_ = opt;
}

Visualizer::Options Visualizer::options() const
{
    std::lock_guard<std::mutex> lk(mtx_);
    return opt_;
}

void Visualizer::pushFrame(const Frame &f)
{
    std::lock_guard<std::mutex> lk(mtx_);
    latest_frame_ = f;

    // trajectory point from pose translation
    Vec3f t = f.local_pose().translation();
    traj_points_.emplace_back(t(0), t(1), t(2));
    if (traj_points_.size() > opt_.max_traj_points)
    {
        traj_points_.erase(traj_points_.begin(),
                           traj_points_.begin() + (traj_points_.size() - opt_.max_traj_points));
    }
}

void Visualizer::pushKeyframe(const Keyframe &kf)
{
    std::lock_guard<std::mutex> lk(mtx_);
    latest_keyframe_ = kf;
    keyframes_.push_back(kf);
    while (keyframes_.size() > opt_.max_keyframes)
        keyframes_.pop_front();
}

void Visualizer::viewerLoop()
{
    // Simple ~30-60Hz loop
    while (running_)
    {
        Options opt;
        std::optional<Frame> f;
        std::optional<KeyFrame> kf;
        std::vector<cv::Point3d> traj_copy;
        std::deque<KeyFrame> kfs_copy;

        {
            std::lock_guard<std::mutex> lk(mtx_);
            opt = opt_;
            f = latest_frame_;
            kf = latest_keyframe_;
            traj_copy = traj_points_;
            kfs_copy = keyframes_;
        }

        if (opt.show_frame_image || opt.show_keyframe_image || opt.show_keyframe_depth)
        {
            render2D(f, kf);
            // Hotkeys for 2D windows
            int key = cv::waitKey(1);
            if (key == 27)
                running_ = false; // ESC
            if (key == 't' || key == 'T')
            {
                opt.show_trajectory = !opt.show_trajectory;
                setOptions(opt);
            }
            if (key == 'i' || key == 'I')
            {
                opt.show_frame_image = !opt.show_frame_image;
                setOptions(opt);
            }
            if (key == 'k' || key == 'K')
            {
                opt.show_keyframe_image = !opt.show_keyframe_image;
                setOptions(opt);
            }
            if (key == 'd' || key == 'D')
            {
                opt.show_keyframe_depth = !opt.show_keyframe_depth;
                setOptions(opt);
            }
        }

        if (opt.use_viz3d && win_)
        {
            // Initialize widgets once
            if (!widgets_initialized_)
            {
                win_->setBackgroundColor(cv::viz::Color::black());
                widgets_initialized_ = true;
            }

            // Trajectory widget
            if (opt.show_trajectory && traj_copy.size() >= 2)
            {
                cv::Mat pts((int)traj_copy.size(), 1, CV_64FC3);
                for (int i = 0; i < (int)traj_copy.size(); ++i)
                {
                    pts.at<cv::Vec3d>(i, 0) = cv::Vec3d(traj_copy[i].x, traj_copy[i].y, traj_copy[i].z);
                }
                cv::viz::WPolyLine line(pts, cv::viz::Color::green());
                win_->showWidget("traj", line);
            }
            else
            {
                win_->removeWidget("traj");
            }

            // Current camera frustum
            if (opt.show_frusta && f.has_value())
            {
                cv::Affine3d Twc = eigenToCvAffine(f->T_w_c);
                cv::viz::WCameraPosition frustum(0.3); // scale
                win_->showWidget("cam", frustum, Twc);
            }
            else
            {
                if (win_->isWidgetShown("cam"))
                    win_->removeWidget("cam");
            }

            // Apply viewer camera pose from ViewState (mouse-driven)
            render3D(f);

            win_->spinOnce(1, true);
        }

        // tiny sleep to avoid pegging CPU (optional)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Cleanup 2D windows
    cv::destroyWindow("Frame");
    cv::destroyWindow("Keyframe");
    cv::destroyWindow("Depth");
}

void Visualizer::render2D(const std::optional<Frame> &f,
                          const std::optional<KeyFrame> &kf)
{
    Options opt = options();

    if (opt.show_frame_image && f.has_value() && !f->image.empty())
    {
        cv::imshow("Frame", f->image());
    }

    if (opt.show_keyframe_image && kf.has_value() && !kf->image.empty())
    {
        cv::imshow("Keyframe", kf->image());
    }

    if (opt.show_keyframe_depth && kf.has_value() && !kf->depth.empty())
    {
        cv::Mat depth_vis = colorizeDepth(kf->depth);
        cv::imshow("Depth", depth_vis);
    }
}

void Visualizer::render3D(const std::optional<Frame> & /*f*/)
{
    if (!win_)
        return;

    // Orbit camera around target based on yaw/pitch/distance
    const double cy = std::cos(view_.yaw), sy = std::sin(view_.yaw);
    const double cp = std::cos(view_.pitch), sp = std::sin(view_.pitch);

    cv::Point3d dir(cp * cy, sp, cp * sy); // forward direction-ish
    cv::Point3d cam_pos = view_.target - dir * view_.distance;

    // Build a lookAt pose (viewer pose is camera-in-world)
    cv::Vec3d eye(cam_pos.x, cam_pos.y, cam_pos.z);
    cv::Vec3d center(view_.target.x, view_.target.y, view_.target.z);
    cv::Vec3d up(0.0, -1.0, 0.0); // depending on your convention, you may want (0,1,0)

    cv::Affine3d viewer = cv::viz::makeCameraPose(eye, center, up);
    win_->setViewerPose(viewer);
}

void Visualizer::mouseCallback(const cv::viz::MouseEvent &ev, void *cookie)
{
    static_cast<VisualizerOpenCV *>(cookie)->onMouse(ev);
}

void Visualizer::onMouse(const cv::viz::MouseEvent &ev)
{
    // Left-drag: orbit (yaw/pitch)
    // Right-drag: pan target
    // Wheel: zoom
    if (ev.type == cv::viz::MouseEvent::Type::MouseButtonPress)
    {
        if (ev.button == cv::viz::MouseEvent::Button::LeftButton)
            view_.dragging_left = true;
        if (ev.button == cv::viz::MouseEvent::Button::RightButton)
            view_.dragging_right = true;
        view_.last_mouse = ev.pointer;
    }
    if (ev.type == cv::viz::MouseEvent::Type::MouseButtonRelease)
    {
        if (ev.button == cv::viz::MouseEvent::Button::LeftButton)
            view_.dragging_left = false;
        if (ev.button == cv::viz::MouseEvent::Button::RightButton)
            view_.dragging_right = false;
    }
    if (ev.type == cv::viz::MouseEvent::Type::MouseMove)
    {
        cv::Point cur = ev.pointer;
        cv::Point d = cur - view_.last_mouse;
        view_.last_mouse = cur;

        const double rot_speed = 0.005;
        const double pan_speed = 0.002 * view_.distance;

        if (view_.dragging_left)
        {
            view_.yaw += d.x * rot_speed;
            view_.pitch += d.y * rot_speed;
            // clamp pitch to avoid flipping
            const double lim = 1.55;
            view_.pitch = std::max(-lim, std::min(lim, view_.pitch));
        }
        if (view_.dragging_right)
        {
            // Pan in viewer plane (approx): move target in X/Z and Y
            view_.target.x -= d.x * pan_speed;
            view_.target.y += d.y * pan_speed;
        }
    }
    if (ev.type == cv::viz::MouseEvent::Type::MouseScroll)
    {
        // ev.wheelRotation: positive/negative depending on OS
        view_.distance *= (ev.wheelRotation > 0 ? 0.9 : 1.1);
        view_.distance = std::max(0.1, view_.distance);
    }
}

cv::Affine3d Visualizer::eigenToCvAffine(const Eigen::Isometry3d &T)
{
    Eigen::Matrix3d R = T.rotation();
    Eigen::Vector3d t = T.translation();
    cv::Matx33d Rc(
        R(0, 0), R(0, 1), R(0, 2),
        R(1, 0), R(1, 1), R(1, 2),
        R(2, 0), R(2, 1), R(2, 2));
    cv::Vec3d tc(t.x(), t.y(), t.z());
    return cv::Affine3d(Rc, tc);
}

cv::Mat Visualizer::colorizeDepth(const cv::Mat &depth)
{
    cv::Mat d32;
    if (depth.type() == CV_32FC1)
    {
        d32 = depth;
    }
    else if (depth.type() == CV_16UC1)
    {
        depth.convertTo(d32, CV_32F, 1.0 / 1000.0); // mm -> m
    }
    else
    {
        cv::Mat tmp;
        depth.convertTo(tmp, CV_32F);
        d32 = tmp;
    }

    // mask invalid (<=0 or NaN)
    cv::Mat valid = (d32 > 0) & (d32 == d32);

    double minv = 0, maxv = 0;
    cv::minMaxLoc(d32, &minv, &maxv, nullptr, nullptr, valid);
    if (!(maxv > minv))
    {
        maxv = minv + 1.0;
    }

    cv::Mat norm, u8, color;
    cv::normalize(d32, norm, 0, 255, cv::NORM_MINMAX, CV_8U, valid);
    cv::applyColorMap(norm, color, cv::COLORMAP_TURBO);

    // paint invalid as black
    color.setTo(cv::Scalar(0, 0, 0), ~valid);
    return color;
}
