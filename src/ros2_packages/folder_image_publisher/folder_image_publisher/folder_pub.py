import os
import glob
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class FolderImagePublisher(Node):
    def __init__(self):
        super().__init__("folder_image_publisher")

        self.declare_parameter("folder", "")
        self.declare_parameter("topic", "/camera/image_raw")
        self.declare_parameter("fps", 1.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("encoding", "bgr8")  # "mono8" if grayscale

        folder = self.get_parameter("folder").get_parameter_value().string_value
        self.topic = self.get_parameter("topic").get_parameter_value().string_value
        fps = float(self.get_parameter("fps").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.encoding = self.get_parameter("encoding").get_parameter_value().string_value
        self.dir = 1

        if not folder or not os.path.isdir(folder):
            raise RuntimeError(f"Parameter 'folder' must be a valid directory. Got: {folder}")

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(folder, e)))
        self.files = sorted(files)

        if not self.files:
            raise RuntimeError(f"No images found in {folder} with extensions {exts}")

        self.get_logger().info(f"Found {len(self.files)} images in: {folder}")
        self.get_logger().info(f"Publishing to: {self.topic} at {fps} FPS (loop={self.loop})")

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, self.topic, qos_profile_sensor_data)

        self.idx = 0
        period = 1.0 / max(1e-6, fps)
        self.timer = self.create_timer(period, self.timer_cb)

    def timer_cb(self):
        if self.idx >= len(self.files):
            if self.loop:
                self.idx = len(self.files) - 1
                self.dir = -1
            else:
                self.get_logger().info("Done publishing all images (loop=false). Shutting down.")
                rclpy.shutdown()
                return
        if self.idx < 0:
            self.idx = 1
            self.dir = 1

        path = self.files[self.idx]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.get_logger().warning(f"Failed to read image: {path} (skipping)")
            self.idx += self.dir
            return

        # If encoding is mono8 but image is color, convert
        if self.encoding == "mono8" and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        msg = self.bridge.cv2_to_imgmsg(img, encoding=self.encoding)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"

        self.pub.publish(msg)
        self.idx += self.dir


def main():
    rclpy.init()
    node = FolderImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()