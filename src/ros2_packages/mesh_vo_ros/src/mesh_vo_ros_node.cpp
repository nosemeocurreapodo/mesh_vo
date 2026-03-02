#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <optional>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "cv_bridge/cv_bridge.hpp"

#include "visualOdometryThreaded.h"

class MeshVoRosNode : public rclcpp::Node
{
public:
    MeshVoRosNode() : Node("mesh_vo_ros_node")
    {
        // ---- Params ----
        image_topic_  = declare_parameter<std::string>("image_topic", "/camera/image_raw");
        map_frame_    = declare_parameter<std::string>("map_frame", "map");
        camera_frame_ = declare_parameter<std::string>("camera_frame", "camera"); // currently unused
        publish_ms_   = declare_parameter<int>("publish_period_ms", 100);

        fx_ = declare_parameter<double>("fx", 525.0);
        fy_ = declare_parameter<double>("fy", 525.0);
        cx_ = declare_parameter<double>("cx", 319.5);
        cy_ = declare_parameter<double>("cy", 239.5);
        width_  = declare_parameter<int>("width", 640);
        height_ = declare_parameter<int>("height", 480);

        debug_log_ = declare_parameter<bool>("debug_log", true);
        debug_img_ = declare_parameter<bool>("debug_img", false);

        // ---- Create VO ----
        vo_ = std::make_unique<VisualOdometryThreaded>(
            static_cast<float>(fx_), static_cast<float>(fy_),
            static_cast<float>(cx_), static_cast<float>(cy_),
            width_, height_,
            debug_log_, debug_img_);

        // ---- ROS I/O ----
        image_subscriber_ = create_subscription<sensor_msgs::msg::Image>(
            image_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(&MeshVoRosNode::image_callback, this, std::placeholders::_1));

        pose_publisher_ = create_publisher<geometry_msgs::msg::PoseStamped>("~/pose", 10);

        auto map_qos = rclcpp::QoS(1).reliable().transient_local();
        mesh_publisher_ = create_publisher<visualization_msgs::msg::Marker>("~/mesh", map_qos);

        // ---- Publisher thread ----
        running_.store(true);
        publish_thread_ = std::thread(&MeshVoRosNode::publish_keyframes, this);

        RCLCPP_INFO(
            get_logger(),
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

        if (!img_typed.isContinuous())
            img_typed = img_typed.clone();

        const ImageType *ptr = reinterpret_cast<const ImageType *>(img_typed.data);
        vo_->locAndMap(ptr);

        {
            std::lock_guard<std::mutex> lk(stamp_mtx_);
            last_stamp_ = msg->header.stamp;
        }
    }

    void publish_keyframes()
    {
        while (rclcpp::ok() && running_.load())
        {
            geometry_msgs::msg::PoseStamped pose_msg;
            visualization_msgs::msg::Marker mesh_msg;
            bool have_pose = false;
            bool have_mesh = false;

            builtin_interfaces::msg::Time stamp;
            {
                std::lock_guard<std::mutex> lk(stamp_mtx_);
                stamp = last_stamp_;
            }

            // Access current keyframe without copying it
            const bool got_kf = vo_->withKeyframe([&](const KeyFrame& kf)
            {
                // ---- Pose ----
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

                have_pose = true;

                // ---- Mesh ----
                mesh_msg.header.stamp = stamp;
                mesh_msg.header.frame_id = map_frame_;
                mesh_msg.ns = "mesh";
                mesh_msg.id = 0;
                mesh_msg.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
                mesh_msg.action = visualization_msgs::msg::Marker::ADD;

                // Assumes vertices are already in map/world coordinates
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
                    mesh_msg.points.reserve(icount);

                    for (size_t i = 0; i < icount; i += 3)
                    {
                        const auto i0 = static_cast<size_t>(indices[i + 0]);
                        const auto i1 = static_cast<size_t>(indices[i + 1]);
                        const auto i2 = static_cast<size_t>(indices[i + 2]);

                        if (i0 >= vcount || i1 >= vcount || i2 >= vcount)
                            continue;

                        geometry_msgs::msg::Point p0, p1, p2;
                        p0.x = vertices[i0 * 3 + 0];
                        p0.y = vertices[i0 * 3 + 1];
                        p0.z = vertices[i0 * 3 + 2];

                        p1.x = vertices[i1 * 3 + 0];
                        p1.y = vertices[i1 * 3 + 1];
                        p1.z = vertices[i1 * 3 + 2];

                        p2.x = vertices[i2 * 3 + 0];
                        p2.y = vertices[i2 * 3 + 1];
                        p2.z = vertices[i2 * 3 + 2];

                        mesh_msg.points.push_back(p0);
                        mesh_msg.points.push_back(p1);
                        mesh_msg.points.push_back(p2);
                    }
                }

                have_mesh = !mesh_msg.points.empty();
            });

            if (got_kf && have_pose)
                pose_publisher_->publish(pose_msg);

            if (got_kf && have_mesh)
                mesh_publisher_->publish(mesh_msg);

            std::this_thread::sleep_for(std::chrono::milliseconds(std::max(1, publish_ms_)));
        }
    }

private:
    // Params
    std::string image_topic_;
    std::string map_frame_;
    std::string camera_frame_; // reserved for future TF publishing
    int publish_ms_{100};

    double fx_{525.0}, fy_{525.0}, cx_{319.5}, cy_{239.5};
    int width_{640}, height_{480};
    bool debug_log_{true};
    bool debug_img_{false};

    // VO
    std::unique_ptr<VisualOdometryThreaded> vo_;

    // Timestamp synchronization
    std::mutex stamp_mtx_;
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