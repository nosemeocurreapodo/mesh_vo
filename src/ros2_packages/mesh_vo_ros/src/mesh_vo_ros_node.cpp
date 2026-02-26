#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "cv_bridge/cv_bridge.hpp"

// Your VO
#include "visualOdometryThreaded.h"

class MeshVoRosNode : public rclcpp::Node
{
public:
  MeshVoRosNode() : Node("mesh_vo_ros_node")
  {
    // ---- Params ----
    image_topic_  = declare_parameter<std::string>("image_topic", "/camera/image_raw");
    map_frame_    = declare_parameter<std::string>("map_frame", "map");
    camera_frame_ = declare_parameter<std::string>("camera_frame", "camera");
    publish_ms_   = declare_parameter<int>("publish_period_ms", 100);

    fx_ = declare_parameter<double>("fx", 525.0);
    fy_ = declare_parameter<double>("fy", 525.0);
    cx_ = declare_parameter<double>("cx", 319.5);
    cy_ = declare_parameter<double>("cy", 239.5);
    width_  = declare_parameter<int>("width", 640);
    height_ = declare_parameter<int>("height", 480);

    // Optional flags you had hardcoded (keep as params)
    use_something_ = declare_parameter<bool>("flag0", true);
    use_something2_ = declare_parameter<bool>("flag1", false);

    // ---- Create VO ----
    vo_ = std::make_unique<VisualOdometryThreaded>(
      static_cast<float>(fx_), static_cast<float>(fy_),
      static_cast<float>(cx_), static_cast<float>(cy_),
      width_, height_,
      use_something_, use_something2_);

    // ---- ROS I/O ----
    image_subscriber_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&MeshVoRosNode::image_callback, this, std::placeholders::_1));

    pose_publisher_ = create_publisher<geometry_msgs::msg::PoseStamped>("~/pose", 10);

    // Map-like data: transient_local so RViz gets last mesh even if it connects later
    auto map_qos = rclcpp::QoS(1).reliable().transient_local();
    mesh_publisher_ = create_publisher<visualization_msgs::msg::Marker>("~/mesh", map_qos);

    // ---- Publisher thread ----
    running_.store(true);
    publish_thread_ = std::thread(&MeshVoRosNode::publish_keyframes, this);

    RCLCPP_INFO(get_logger(),
      "MeshVoRosNode started. image_topic=%s intrinsics=[fx=%.3f fy=%.3f cx=%.3f cy=%.3f] size=%dx%d",
      image_topic_.c_str(), fx_, fy_, cx_, cy_, width_, height_);
  }

  ~MeshVoRosNode() override
  {
    running_.store(false);
    if (publish_thread_.joinable())
      publish_thread_.join();
  }

private:
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    // Keep callback small but thread-safe: VO is accessed by callback + publisher thread.
    // If VisualOdometryThreaded is internally thread-safe you can remove this mutex.
    std::lock_guard<std::mutex> lk(vo_mtx_);

    // Convert image (note: toCvCopy will copy; required if you need float / specific type)
    cv_bridge::CvImageConstPtr mono8;
    try
    {
      mono8 = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (const cv_bridge::Exception &e)
    {
      RCLCPP_ERROR(get_logger(), "cv_bridge toCvShare MONO8 failed: %s", e.what());
      return;
    }

    // Convert to your ImageType OpenCV format (e.g., CV_32F) as needed by your VO
    cv::Mat img_typed;
    try
    {
      mono8->image.convertTo(img_typed, GetOpenCVFormat<ImageType>());
    }
    catch (const cv::Exception &e)
    {
      RCLCPP_ERROR(get_logger(), "OpenCV convertTo failed: %s", e.what());
      return;
    }

    // Safety: ensure contiguous data if your VO expects a flat pointer
    if (!img_typed.isContinuous())
      img_typed = img_typed.clone();

    ImageType *ptr = reinterpret_cast<ImageType *>(img_typed.data);

    if (is_first_image_)
    {
      vo_->flatInit(ptr);
      is_first_image_ = false;
    }
    else
    {
      vo_->locAndMap(ptr);
    }

    // Keep last stamp so published pose aligns with image time (more useful than now())
    last_stamp_ = msg->header.stamp;
  }

  void publish_keyframes()
  {
    using namespace std::chrono_literals;

    while (rclcpp::ok() && running_.load())
    {
      KeyFrame kf;
      builtin_interfaces::msg::Time stamp;

      {
        // Protect VO access (getKeyframe) vs callback updates
        std::lock_guard<std::mutex> lk(vo_mtx_);
        kf = vo_->getKeyframe();
        stamp = last_stamp_;
      }

      // Publish pose
      geometry_msgs::msg::PoseStamped pose_msg;
      pose_msg.header.stamp = stamp;
      pose_msg.header.frame_id = map_frame_;

      const auto &T = kf.global_pose();
      pose_msg.pose.position.x = T.translation()(0);
      pose_msg.pose.position.y = T.translation()(1);
      pose_msg.pose.position.z = T.translation()(2);

      auto q = T.so3().unit_quaternion();
      pose_msg.pose.orientation.x = q.x();
      pose_msg.pose.orientation.y = q.y();
      pose_msg.pose.orientation.z = q.z();
      pose_msg.pose.orientation.w = q.w();

      pose_publisher_->publish(pose_msg);

      // Publish mesh marker
      visualization_msgs::msg::Marker mesh_msg;
      mesh_msg.header.stamp = stamp;
      mesh_msg.header.frame_id = map_frame_;
      mesh_msg.ns = "mesh";
      mesh_msg.id = 0;
      mesh_msg.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
      mesh_msg.action = visualization_msgs::msg::Marker::ADD;

      // IMPORTANT:
      // If vertices are already in map/world coordinates, keep identity pose.
      // If vertices are in camera/keyframe local coordinates, set mesh_msg.pose = pose_msg.pose.
      // Here we assume vertices are in map coordinates (safer default for "map mesh").
      mesh_msg.pose.orientation.w = 1.0;

      mesh_msg.scale.x = mesh_msg.scale.y = mesh_msg.scale.z = 1.0;
      mesh_msg.color.a = 1.0;
      mesh_msg.color.r = 1.0;
      mesh_msg.color.g = 1.0;
      mesh_msg.color.b = 1.0;

      auto vertices = kf.mesh().vertex_buffer_.MapRead();
      auto indices  = kf.mesh().ebo_buffer_.MapRead();

      const size_t vcount = vertices.size() / 3;
      const size_t icount = indices.size();

      if (icount >= 3 && (icount % 3) == 0)
      {
        mesh_msg.points.reserve(icount); // 1 point per index

        for (size_t i = 0; i < icount; i += 3)
        {
          const auto i0 = static_cast<size_t>(indices[i + 0]);
          const auto i1 = static_cast<size_t>(indices[i + 1]);
          const auto i2 = static_cast<size_t>(indices[i + 2]);

          if (i0 >= vcount || i1 >= vcount || i2 >= vcount) continue;

          geometry_msgs::msg::Point p0, p1, p2;
          p0.x = vertices[i0 * 3 + 0]; p0.y = vertices[i0 * 3 + 1]; p0.z = vertices[i0 * 3 + 2];
          p1.x = vertices[i1 * 3 + 0]; p1.y = vertices[i1 * 3 + 1]; p1.z = vertices[i1 * 3 + 2];
          p2.x = vertices[i2 * 3 + 0]; p2.y = vertices[i2 * 3 + 1]; p2.z = vertices[i2 * 3 + 2];

          mesh_msg.points.push_back(p0);
          mesh_msg.points.push_back(p1);
          mesh_msg.points.push_back(p2);
        }
      }

      // Avoid publishing empty TRIANGLE_LIST markers (RViz can be cranky)
      if (!mesh_msg.points.empty())
        mesh_publisher_->publish(mesh_msg);

      std::this_thread::sleep_for(std::chrono::milliseconds(std::max(1, publish_ms_)));
    }
  }

private:
  // Params
  std::string image_topic_;
  std::string map_frame_;
  std::string camera_frame_;
  int publish_ms_{100};

  double fx_{525.0}, fy_{525.0}, cx_{319.5}, cy_{239.5};
  int width_{640}, height_{480};
  bool use_something_{true}, use_something2_{false};

  // VO + synchronization
  std::unique_ptr<VisualOdometryThreaded> vo_;
  std::mutex vo_mtx_;
  bool is_first_image_{true};
  builtin_interfaces::msg::Time last_stamp_{};

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mesh_publisher_;

  // Thread
  std::atomic<bool> running_{false};
  std::thread publish_thread_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MeshVoRosNode>());
  rclcpp::shutdown();
  return 0;
}