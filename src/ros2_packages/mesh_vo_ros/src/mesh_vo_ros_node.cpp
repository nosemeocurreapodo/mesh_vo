#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "visualization_msgs/msg/marker.hpp"
#include "nav_msgs/msg/path.hpp"
#include "cv_bridge/cv_bridge.hpp"

#include "visualOdometryThreaded.h"

class MeshVoRosNode : public rclcpp::Node
{
public:
	MeshVoRosNode() : Node("mesh_vo_ros_node")
	{
		// ---- Params ----
		image_topic_ = declare_parameter<std::string>("image_topic", "/camera/image_raw");
		map_frame_ = declare_parameter<std::string>("map_frame", "map");
		camera_frame_ = declare_parameter<std::string>("camera_frame", "camera");

		fx_ = declare_parameter<double>("fx", 525.0);
		fy_ = declare_parameter<double>("fy", 525.0);
		cx_ = declare_parameter<double>("cx", 319.5);
		cy_ = declare_parameter<double>("cy", 239.5);
		width_ = declare_parameter<int>("width", 640);
		height_ = declare_parameter<int>("height", 480);

		cam_ = Cameraf(fx_, fy_, cx_, cy_, width_, height_);

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

		auto latched_qos = rclcpp::QoS(1).reliable().transient_local();
		path_publisher_ = create_publisher<nav_msgs::msg::Path>("~/keyframe_path", latched_qos);
		cloud_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/keyframe_cloud", latched_qos);
		mesh_publisher_ = create_publisher<visualization_msgs::msg::Marker>("~/mesh", latched_qos);

		tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

		path_msg_.header.frame_id = map_frame_;

		// ---- Worker threads ----
		running_.store(true);
		pose_thread_ = std::thread(&MeshVoRosNode::pose_publish_loop, this);
		keyframe_thread_ = std::thread(&MeshVoRosNode::keyframe_publish_loop, this);

		RCLCPP_INFO(
			get_logger(),
			"MeshVoRosNode started. image_topic=%s intrinsics=[fx=%.3f fy=%.3f cx=%.3f cy=%.3f] size=%dx%d",
			image_topic_.c_str(), fx_, fy_, cx_, cy_, width_, height_);
	}

	~MeshVoRosNode() override
	{
		running_.store(false);

		if (pose_thread_.joinable())
			pose_thread_.join();
		if (keyframe_thread_.joinable())
			keyframe_thread_.join();
	}

private:
	static uint8_t gray_to_u8(float v)
	{
		if (v <= 1.0f)
			v *= 255.0f;
		v = std::clamp(v, 0.0f, 255.0f);
		return static_cast<uint8_t>(v);
	}

	// static SE3f convert_cam_cv_to_cam_ros(const SE3f &T_map_cam_cv)
	//{
	//     SE3f T_map_cam_ros = T_map_cam_cv.inverse();
	//     return T_map_cam_ros;
	// }

	static SE3d convert_cam_cv_to_cam_ros(const SE3d &T_map_cam_cv)
	{
		// OpenCV camera frame:
		//   x right, y down, z forward
		//
		// ROS-style camera/body frame:
		//   x forward, y left, z up
		//
		// This matches the same convention as your previous matrix-based version.

		constexpr double half_pi = 1.57079632679f;

		// Pure rotations, no translation
		const SE3d Rx90(SO3<double>::exp(Vec3<double>(-half_pi, 0.0f, 0.0f)), Vec3<double>(0.0f, 0.0f, 0.0f));
		// const SE3d Rz90(SO3<double>::exp(Vec3f(0.0f, 0.0f, half_pi)), Vec3f(0.0f, 0.0f, 0.0f));

		// Change only the camera-frame convention; camera center stays the same
		return T_map_cam_cv.inverse(); // * Rx90;
	}

	void publish_tf(const SE3d &T_map_cam,
					const builtin_interfaces::msg::Time &stamp)
	{
		geometry_msgs::msg::TransformStamped tf_msg;
		tf_msg.header.stamp = stamp;
		tf_msg.header.frame_id = map_frame_;
		tf_msg.child_frame_id = camera_frame_;

		tf_msg.transform.translation.x = T_map_cam.translation()(0);
		tf_msg.transform.translation.y = T_map_cam.translation()(1);
		tf_msg.transform.translation.z = T_map_cam.translation()(2);

		auto q = T_map_cam.so3().unit_quaternion();
		tf_msg.transform.rotation.x = q.x();
		tf_msg.transform.rotation.y = q.y();
		tf_msg.transform.rotation.z = q.z();
		tf_msg.transform.rotation.w = q.w();

		tf_broadcaster_->sendTransform(tf_msg);
	}

	geometry_msgs::msg::Pose to_pose_msg(const SE3d &T) const
	{
		geometry_msgs::msg::Pose pose;
		pose.position.x = T.translation()(0);
		pose.position.y = T.translation()(1);
		pose.position.z = T.translation()(2);

		auto q = T.so3().unit_quaternion();
		pose.orientation.x = q.x();
		pose.orientation.y = q.y();
		pose.orientation.z = q.z();
		pose.orientation.w = q.w();
		return pose;
	}

	builtin_interfaces::msg::Time current_stamp() const
	{
		std::lock_guard<std::mutex> lk(stamp_mtx_);
		return last_stamp_;
	}

	sensor_msgs::msg::PointCloud2 make_colorized_cloud(const KeyFrame &kf,
													   const Cameraf &cam,
													   const builtin_interfaces::msg::Time &stamp) const
	{
		sensor_msgs::msg::PointCloud2 cloud;
		cloud.header.frame_id = map_frame_;
		cloud.header.stamp = stamp;

		auto vertices = kf.mesh().vertex_buffer_.MapRead();
		auto img = kf.image().MapRead(0);

		const int img_w = static_cast<int>(kf.image().width(0));
		const int img_h = static_cast<int>(kf.image().height(0));

		const size_t vcount = vertices.size() / 3;

		sensor_msgs::PointCloud2Modifier modifier(cloud);
		modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
		modifier.resize(vcount);

		sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
		sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
		sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
		sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(cloud, "r");
		sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(cloud, "g");
		sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(cloud, "b");

		const float s = kf.global_scale();
		const SE3d &T = convert_cam_cv_to_cam_ros(kf.global_pose());

		constexpr float eps = 1e-8f;

		for (size_t vi = 0; vi < vcount; ++vi, ++iter_x, ++iter_y, ++iter_z, ++iter_r, ++iter_g, ++iter_b)
		{
			const float xl = vertices[vi * 3 + 0] * s;
			const float yl = vertices[vi * 3 + 1] * s;
			const float zl = vertices[vi * 3 + 2] * s;

			Vec3f p_local(xl, yl, zl);
			Vec3<double> pd_local(xl, yl, zl);
			Vec3<double> p_world = T * pd_local;

			*iter_x = float(p_world(0));
			*iter_y = float(p_world(1));
			*iter_z = float(p_world(2));

			Vec2f pix = cam.pointToPix(p_local);

			uint8_t gray = 0;

			if (cam.IsPixVisible(pix))
			{
				const int ix = std::clamp(static_cast<int>(std::round(pix(0) * img_w)), 0, img_w - 1);
				const int iy = std::clamp(static_cast<int>(std::round(pix(1) * img_h)), 0, img_h - 1);
				gray = gray_to_u8(static_cast<float>(img[iy * img_w + ix]));
			}

			*iter_r = gray;
			*iter_g = gray;
			*iter_b = gray;
		}

		return cloud;
	}

	visualization_msgs::msg::Marker make_mesh_marker(const KeyFrame &kf,
													 const builtin_interfaces::msg::Time &stamp) const
	{
		visualization_msgs::msg::Marker marker;

		// Publish the mesh
		marker.header.stamp = stamp;
		marker.header.frame_id = "map";
		marker.ns = "mesh";
		marker.id = 0;
		marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
		marker.action = visualization_msgs::msg::Marker::ADD;
		marker.pose = to_pose_msg(convert_cam_cv_to_cam_ros(kf.global_pose()));
		marker.scale.x = kf.global_scale();
		marker.scale.y = kf.global_scale();
		marker.scale.z = kf.global_scale();
		marker.color.a = 1.0;
		marker.color.r = 1.0;
		marker.color.g = 1.0;
		marker.color.b = 1.0;

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

			marker.points.push_back(p1);
			marker.points.push_back(p2);
			marker.points.push_back(p3);
		}

		return marker;
	}

	visualization_msgs::msg::Marker make_mesh_marker2(const KeyFrame &kf,
													  const Cameraf &cam,
													  const builtin_interfaces::msg::Time &stamp) const
	{
		visualization_msgs::msg::Marker marker;

		marker.header.stamp = stamp;
		marker.header.frame_id = "map";
		marker.ns = "mesh";
		marker.id = 0;
		marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
		marker.action = visualization_msgs::msg::Marker::ADD;

		// Keep this if your global_pose() is map->camera and you want marker pose as camera->map.
		// If global_pose() is already camera->map, remove the inverse().
		marker.pose = to_pose_msg(convert_cam_cv_to_cam_ros(kf.global_pose()));

		// Mesh is locally scaled
		marker.scale.x = kf.global_scale();
		marker.scale.y = kf.global_scale();
		marker.scale.z = kf.global_scale();

		// Base color should stay white so it does not tint per-vertex colors
		marker.color.a = 1.0f;
		marker.color.r = 1.0f;
		marker.color.g = 1.0f;
		marker.color.b = 1.0f;

		auto vertices = kf.mesh().vertex_buffer_.MapRead();
		auto indices = kf.mesh().ebo_buffer_.MapRead();
		auto image = kf.image().MapRead(0);

		const int img_w = static_cast<int>(kf.image().width(0));
		const int img_h = static_cast<int>(kf.image().height(0));

		if (indices.size() < 3 || (indices.size() % 3) != 0 || img_w <= 0 || img_h <= 0)
			return marker;

		// Current code assumes xyz-packed vertices
		const size_t stride = 3;
		const size_t vcount = vertices.size() / stride;

		marker.points.reserve(indices.size());
		marker.colors.reserve(indices.size());

		auto sample_gray = [&](const geometry_msgs::msg::Point &p_local) -> std_msgs::msg::ColorRGBA
		{
			std_msgs::msg::ColorRGBA c;
			c.a = 1.0f;

			Vec3f p_cam(static_cast<float>(p_local.x),
						static_cast<float>(p_local.y),
						static_cast<float>(p_local.z));

			// Reject points behind the camera
			if (p_cam(2) <= 1e-8f)
			{
				c.r = c.g = c.b = 0.5f;
				return c;
			}

			Vec2f pix = cam.pointToPix(p_cam);

			const int px = std::clamp(static_cast<int>(std::round(pix(0) * img_w)), 0, img_w - 1);
			const int py = std::clamp(static_cast<int>(std::round(pix(1) * img_h)), 0, img_h - 1);

			float g = 0.0f;
			if (cam.IsPixVisible(pix))
			{
				g = static_cast<float>(image[py * img_w + px]);

				// If ImageType is float in [0,1], map to [0,1] color.
				// If it is already [0,255], normalize.
				if (g > 1.0f)
					g /= 255.0f;

				g = std::clamp(g, 0.0f, 1.0f);
			}
			else
			{
				g = 0.0f;
			}

			c.r = g;
			c.g = g;
			c.b = g;
			return c;
		};

		for (size_t i = 0; i < indices.size(); i += 3)
		{
			const size_t i0 = static_cast<size_t>(indices[i + 0]);
			const size_t i1 = static_cast<size_t>(indices[i + 1]);
			const size_t i2 = static_cast<size_t>(indices[i + 2]);

			if (i0 >= vcount || i1 >= vcount || i2 >= vcount)
				continue;

			geometry_msgs::msg::Point p0, p1, p2;

			p0.x = vertices[i0 * stride + 0];
			p0.y = vertices[i0 * stride + 1];
			p0.z = vertices[i0 * stride + 2];

			p1.x = vertices[i1 * stride + 0];
			p1.y = vertices[i1 * stride + 1];
			p1.z = vertices[i1 * stride + 2];

			p2.x = vertices[i2 * stride + 0];
			p2.y = vertices[i2 * stride + 1];
			p2.z = vertices[i2 * stride + 2];

			marker.points.push_back(p0);
			marker.points.push_back(p1);
			marker.points.push_back(p2);

			marker.colors.push_back(sample_gray(p0));
			marker.colors.push_back(sample_gray(p1));
			marker.colors.push_back(sample_gray(p2));
		}

		return marker;
	}

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

	void pose_publish_loop()
	{
		uint64_t pose_seq = 0;
		VisualOdometryThreaded::PoseUpdate upd;

		while (rclcpp::ok() && running_.load())
		{
			if (!vo_->waitForNextPose(pose_seq, upd, std::chrono::milliseconds(100)))
				continue;

			geometry_msgs::msg::PoseStamped pose_msg;
			pose_msg.header.stamp = current_stamp();
			pose_msg.header.frame_id = map_frame_;
			pose_msg.pose = to_pose_msg(convert_cam_cv_to_cam_ros(upd.global_pose));

			pose_publisher_->publish(pose_msg);

			publish_tf(convert_cam_cv_to_cam_ros(upd.global_pose), current_stamp());
		}
	}

	void keyframe_publish_loop()
	{
		uint64_t kf_upd_seq = 0;

		while (rclcpp::ok() && running_.load())
		{
			if (!vo_->waitForNextKeyframeUpdate(kf_upd_seq, std::chrono::milliseconds(100)))
				continue;

			const builtin_interfaces::msg::Time stamp = current_stamp();

			nav_msgs::msg::Path path_to_publish;
			sensor_msgs::msg::PointCloud2 cloud_to_publish;
			visualization_msgs::msg::Marker mesh_to_publish;

			bool have_path = false;
			bool have_cloud = false;
			bool have_mesh = false;

			vo_->withKeyframe([&](const KeyFrame &kf)
							  {
                                  geometry_msgs::msg::PoseStamped kf_pose;
                                  kf_pose.header.stamp = stamp;
                                  kf_pose.header.frame_id = map_frame_;
                                  kf_pose.pose = to_pose_msg(convert_cam_cv_to_cam_ros(kf.global_pose()));

                                  // Append path only when the keyframe id changes; otherwise update last pose
                                  if (path_msg_.poses.empty() || kf.id() != last_path_kf_id_)
                                  {
                                      path_msg_.poses.push_back(kf_pose);
                                      last_path_kf_id_ = kf.id();
                                  }
                                  else
                                  {
                                      path_msg_.poses.back() = kf_pose;
                                  }

                                  path_msg_.header.stamp = stamp;
                                  path_msg_.header.frame_id = map_frame_;
                                  path_to_publish = path_msg_;
                                  have_path = true;

                                  cloud_to_publish = make_colorized_cloud(kf, cam_, stamp);
                                  have_cloud = (cloud_to_publish.width > 0);

                                  //mesh_to_publish = make_mesh_marker(kf, stamp);
                                  mesh_to_publish = make_mesh_marker2(kf, cam_, stamp);
                                  have_mesh = true; });

			if (have_path)
				path_publisher_->publish(path_to_publish);

			if (have_cloud)
				cloud_publisher_->publish(cloud_to_publish);

			if (have_mesh)
				mesh_publisher_->publish(mesh_to_publish);
		}
	}

private:
	// Params
	std::string image_topic_;
	std::string map_frame_;
	std::string camera_frame_;

	double fx_{525.0}, fy_{525.0}, cx_{319.5}, cy_{239.5};
	int width_{640}, height_{480};
	bool debug_log_{true};
	bool debug_img_{false};

	// VO
	std::unique_ptr<VisualOdometryThreaded> vo_;

	// Timestamp synchronization
	mutable std::mutex stamp_mtx_;
	builtin_interfaces::msg::Time last_stamp_{};

	// Path state
	nav_msgs::msg::Path path_msg_;
	int last_path_kf_id_{-1};

	// ROS
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
	rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
	rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_publisher_;
	rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mesh_publisher_;

	std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

	Cameraf cam_;

	// Threads
	std::atomic<bool> running_{false};
	std::thread pose_thread_;
	std::thread keyframe_thread_;
};

int main(int argc, char **argv)
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<MeshVoRosNode>());
	rclcpp::shutdown();
	return 0;
}