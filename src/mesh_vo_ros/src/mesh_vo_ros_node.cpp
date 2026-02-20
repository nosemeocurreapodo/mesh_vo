#include "sensor_msgs/image_encodings.hpp"
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "cv_bridge/cv_bridge.hpp"

#include "visualOdometryThreaded.h"

class MeshVoRosNode : public rclcpp::Node
{
public:
    MeshVoRosNode() : Node("mesh_vo_ros_node")
    {
        // Create a visual odometry object
        vo_ = std::make_unique<VisualOdometryThreaded>(
            525.0f, 525.0f, 319.5f, 239.5f, // fx, fy, cx, cy
            640, 480,                       // width, height
            true, false);

        // Create a subscriber to the image topic
        image_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw",
            rclcpp::SensorDataQoS(),
            std::bind(&MeshVoRosNode::image_callback, this, std::placeholders::_1));

        // Create a publisher for the pose
        pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("~/pose", 10);

        // Create a publisher for the mesh
        mesh_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("~/mesh", 10);

        // Create a thread to publish the keyframes
        running_.store(true);
        publish_thread_ = std::thread(&MeshVoRosNode::publish_keyframes, this);
    }

    ~MeshVoRosNode() override
    {
        running_.store(false);
        if (publish_thread_.joinable())
            publish_thread_.join();
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Convert the ROS image to a CV Mat
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
            cv_ptr->image.convertTo(cv_ptr->image, GetOpenCVFormat<ImageType>());
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Pass the image to the visual odometry
        if (is_first_image_)
        {
            vo_->flatInit((ImageType *)cv_ptr->image.data);
            is_first_image_ = false;
        }
        else
        {
            vo_->locAndMap((ImageType *)cv_ptr->image.data);
        }
    }

    void publish_keyframes()
    {
        while (rclcpp::ok() && running_.load())
        {
            // Get the keyframe
            KeyFrame kf = vo_->getKeyframe();

            // Publish the pose
            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header.stamp = this->now();
            pose_msg.header.frame_id = "map";
            pose_msg.pose.position.x = kf.global_pose().translation()(0);
            pose_msg.pose.position.y = kf.global_pose().translation()(1);
            pose_msg.pose.position.z = kf.global_pose().translation()(2);
            pose_msg.pose.orientation.x = kf.global_pose().so3().unit_quaternion().x();
            pose_msg.pose.orientation.y = kf.global_pose().so3().unit_quaternion().y();
            pose_msg.pose.orientation.z = kf.global_pose().so3().unit_quaternion().z();
            pose_msg.pose.orientation.w = kf.global_pose().so3().unit_quaternion().w();
            pose_publisher_->publish(pose_msg);

            // Publish the mesh
            visualization_msgs::msg::Marker mesh_msg;
            mesh_msg.header.stamp = this->now();
            mesh_msg.header.frame_id = "map";
            mesh_msg.ns = "mesh";
            mesh_msg.id = 0;
            mesh_msg.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
            mesh_msg.action = visualization_msgs::msg::Marker::ADD;
            mesh_msg.pose = pose_msg.pose;
            mesh_msg.scale.x = 1.0;
            mesh_msg.scale.y = 1.0;
            mesh_msg.scale.z = 1.0;
            mesh_msg.color.a = 1.0;
            mesh_msg.color.r = 1.0;
            mesh_msg.color.g = 1.0;
            mesh_msg.color.b = 1.0;

            auto vertices = kf.mesh().vertex_buffer_.MapRead();
            auto indices = kf.mesh().ebo_buffer_.MapRead();

            for (size_t i = 0; i < indices.size(); i += 3)
            {
                geometry_msgs::msg::Point p1, p2, p3;
                p1.x = vertices[indices[i] * 3];
                p1.y = vertices[indices[i] * 3 + 1];
                p1.z = vertices[indices[i] * 3 + 2];

                p2.x = vertices[indices[i + 1] * 3];
                p2.y = vertices[indices[i + 1] * 3 + 1];
                p2.z = vertices[indices[i + 1] * 3 + 2];

                p3.x = vertices[indices[i + 2] * 3];
                p3.y = vertices[indices[i + 2] * 3 + 1];
                p3.z = vertices[indices[i + 2] * 3 + 2];

                mesh_msg.points.push_back(p1);
                mesh_msg.points.push_back(p2);
                mesh_msg.points.push_back(p3);
            }
            mesh_publisher_->publish(mesh_msg);

            // Sleep for a bit
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    std::unique_ptr<VisualOdometryThreaded> vo_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mesh_publisher_;
    std::atomic<bool> running_{false};
    std::thread publish_thread_;
    bool is_first_image_ = true;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MeshVoRosNode>());
    rclcpp::shutdown();
    return 0;
}
