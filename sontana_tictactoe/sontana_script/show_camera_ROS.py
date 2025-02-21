import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from realsense2_camera_msgs.msg import Extrinsics
import numpy as np
import cv2
from cv_bridge import CvBridge

class SHOWCAM(Node):
    def __init__(self):
        super().__init__('show_camera_feed')
        self.bridge = CvBridge()
        self.camera_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.camera_callback,
            10)
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10)
        self.extrinsics_subscription = self.create_subscription(
            Extrinsics,
            '/camera/camera/extrinsics/depth_to_color',
            self.extrinsics_callback,
            10)
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera/aligned_depth_to_color/camera_info',
            self.camera_info_callback,
            10)
        self.camera_subscription  # prevent unused variable warning
        self.depth_subscription  # prevent unused variable warning
        self.extrinsics_subscription  # prevent unused variable warning
        self.camera_info_subscription  # prevent unused variable warning
        self.extrinsics_rotation = None
        self.extrinsics_translation = None
        self.intrinsics = None
        self.depth_image = None
        self.color_image = None
        self.timer = self.create_timer(0.1, self.detect_and_print_coordinates)

    def camera_callback(self, msg):
        #self.get_logger().info('Received camera image data')
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        #self.get_logger().info('Received depth image data')
        self.depth_image = msg

    def extrinsics_callback(self, msg):
        #self.get_logger().info('Received extrinsics data')
        self.extrinsics_rotation = np.array(msg.rotation).reshape(3, 3)
        self.extrinsics_translation = np.array(msg.translation)

    def camera_info_callback(self, msg):
        #self.get_logger().info('Received camera info data')
        self.intrinsics = msg

    def calculate_real_world_coordinates(self, u, v, depth):
        if self.intrinsics is None:
            self.get_logger().warn('Intrinsics not received yet')
            return None

        # Get camera intrinsics
        fx = self.intrinsics.k[0]
        fy = self.intrinsics.k[4]
        cx = self.intrinsics.k[2]
        cy = self.intrinsics.k[5]

        # Calculate real-world coordinates
        z = depth / 1000.0  # Convert depth from mm to meters if necessary
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Apply extrinsics transformation if available
        if self.extrinsics_rotation is not None and self.extrinsics_translation is not None:
            point = np.array([x, y, z])
            transformed_point = self.extrinsics_rotation @ point + self.extrinsics_translation
            return transformed_point

        return np.array([x, y, z])

    def detect_and_print_coordinates(self):
        if self.color_image is None or self.depth_image is None:
            self.get_logger().warn('Color or depth image not received yet')
            return

        # Example: Detect the center of the image
        height, width, _ = self.color_image.shape
        center_u = width // 2
        center_v = height // 2

        # Get the depth value at the center of the image
        depth_array = np.frombuffer(self.depth_image.data, dtype=np.uint16).reshape(self.depth_image.height, self.depth_image.width)
        depth = depth_array[center_v, center_u]

        # Calculate real-world coordinates
        coordinates = self.calculate_real_world_coordinates(center_u, center_v, depth)
        if coordinates is not None:
            self.get_logger().info(f'Center coordinates: x={coordinates[0]*1000.0} mm, y={coordinates[1]*1000.0} mm, z={coordinates[2]*1000.0} mm')

            # Draw the coordinates on the image
            text = f"x={coordinates[0]*1000.0 :.2f} mm, y={coordinates[1]*1000.0:.2f} mm , z={coordinates[2]*1000.0:.2f} mm"
            cv2.putText(self.color_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the image
        cv2.imshow('Camera Feed', self.color_image)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = SHOWCAM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()