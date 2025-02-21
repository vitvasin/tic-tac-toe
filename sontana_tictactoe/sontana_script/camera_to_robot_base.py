import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
import numpy as np

class CAM2BASE(Node):
    def __init__(self):
        super().__init__('camera_to_robot_base')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.get_transform)

    def get_transform(self):
        from_frame = 'head_camera'
        to_frame = 'gen3lite_base_link'
        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform(to_frame, from_frame, rclpy.time.Time())

            tx, ty, tz = trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z
            qx, qy, qz, qw = trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w

            T = self.to_transformation_matrix(tx, ty, tz, qx, qy, qz, qw)
            self.get_logger().info(f'Transformation Matrix from {from_frame} to {to_frame}:\n{T}')
            #transform camera coordinates to robot base coordinates
            
            # camera_coordinates = [0, 0, 0.5, 1]
            # robot_coordinates = np.dot(T, camera_coordinates)
            # self.get_logger().info(f'Camera coordinates: {camera_coordinates}')
            # self.get_logger().info(f'Robot coordinates: {robot_coordinates}')

        except Exception as e:
            self.get_logger().warn(f'Could not get transform: {str(e)}')

    def to_transformation_matrix(self, tx, ty, tz, qx, qy, qz, qw):
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat([qx, qy, qz, qw])
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = [tx, ty, tz]
        return T
    

            

def main():
    rclpy.init()
    node = CAM2BASE()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
