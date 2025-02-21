import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from realsense2_camera_msgs.msg import Extrinsics
import numpy as np
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO

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
        self.mask = None
        self.timer = self.create_timer(0.1, self.detect_and_print_coordinates)

        # Load YOLO model
        model_path = "/home/stu/tic-tac-toe_test_yolo_20.2.2025/tic-tac-toe_test_yolo/datasets_tic_tac_toe_v1/best_tictactoe_v1.pt"
        self.model = YOLO(model_path)
        self.class_names = ['None', 'o', 'table', 'x']
        self.target_class = 'table'

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

        # Perform object detection
        results = self.model(source=self.color_image, conf=0.8, verbose=True)

        depth_array = np.frombuffer(self.depth_image.data, dtype=np.uint16).reshape(self.depth_image.height, self.depth_image.width)

        for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        #print("No boxes found in result.boxes, checking obb object")
                        
                        obb = result.obb
                        if obb is not None:
                            
                            class_ids = obb.cls.int().tolist()
                            confidences = obb.conf.tolist()
                            coordinates = obb.xyxy.int().tolist()
                        else:
                            continue
                    else:
                        class_ids = boxes.cls.int().tolist()
                        confidences = boxes.conf.tolist()
                        coordinates = boxes.xyxy.int().tolist()
                    for class_id, confidence, (x1, y1, x2, y2) in zip(class_ids, confidences, coordinates):
                        class_name = self.class_names[class_id]
                        #print(f"Class: {class_name}, Confidence: {confidence}, Coordinates: {x1, y1, x2, y2}")
                        if class_name == self.target_class:
                            self.get_logger().info('Found target class')
                            x1-= 20
                            y1-= 20
                            x2+= 20
                            y2+= 20
                            roi = self.color_image[y1:y2, x1:x2]
                            self.mask = np.zeros_like(self.color_image)
                            self.mask[y1:y2, x1:x2] = roi
                            
                            roi_results = self.model(source=self.mask, conf=0.75, verbose=False, show=False)
                        
                            for roi_result in roi_results:
                                roi_boxes = roi_result.boxes
                                if roi_boxes is None:
                                    roi_obb = roi_result.obb
                                    if roi_obb is not None:
                                        # remove class named "table" from roi_obb
                                        roi_class_ids = roi_obb.cls.int().tolist()
                                        roi_confidences = roi_obb.conf.tolist()
                                        roi_coordinates = roi_obb.xyxy.int().tolist()
                                        roi_center = roi_obb.xywhr.int().tolist()
                                    else:
                                        continue
                                else:    
                                    # remove class named "table" from roi_boxes
                                    roi_boxes = [box for box in roi_boxes if self.class_names[box.cls.int().item()] != 'table']
                                    roi_class_ids = roi_boxes.cls.int().tolist()
                                    roi_confidences = roi_boxes.conf.tolist()
                                    roi_coordinates = roi_boxes.xyxy.int().tolist()
                                    roi_center = roi_boxes.xywhr.int().tolist()
                        
                        else:
                            self.get_logger().info('target class not found')

        # Display the image
        # if self.mask is not None:
        #     #cv2.imshow('Camera Feed', self.mask)
        #     cv2.waitKey(1)

def main():
    rclpy.init()
    node = SHOWCAM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()