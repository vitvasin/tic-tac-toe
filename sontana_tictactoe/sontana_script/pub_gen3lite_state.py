import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from kortex_api.RouterClient import RouterClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.TCPTransport import TCPTransport
import numpy as np
import os,sys
import serial,time



class Gen3l_JointStatePublisher(Node):
    def __init__(self, base_kin):
        super().__init__('joint_state_publisher')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.01, self.publish_joint_states)  # 10 Hz
        self.joint_names = [ 'left_wheel_joint', 'right_wheel_joint', 'body_joint', 'neck_joint', 'head_joint', 'gen3lite_joint_1', 'gen3lite_joint_2', 'gen3lite_joint_3', 'gen3lite_joint_4', 'gen3lite_joint_5', 'gen3lite_joint_6', 'right_finger_bottom_joint']
        

        self.get_logger().info('init')
        # self.joint_names = [
        #     'gen3lite_joint_1', 'gen3lite_joint_2', 'gen3lite_joint_3', 'gen3lite_joint_4', 'gen3lite_joint_5', 'gen3lite_joint_6'
        # ]
        
        self.base = base_kin
        #initial position
        self.left_wheel_joint = 0.0
        self.right_wheel_joint = 0.0
        self.body_joint = 0.0
        self.neck_joint = 0.0
        self.head_joint = 0.0
        self.gen3_lite_joint_1 = 0.0
        self.gen3_lite_joint_2 = 0.0
        self.gen3_lite_joint_3 = 0.0
        self.gen3_lite_joint_4 = 0.0
        self.gen3_lite_joint_5 = 0.0
        self.gen3_lite_joint_6 = 0.0
        self.gripper_joint = 0.0
        self.ser = serial.Serial('/dev/ttyACM0', 57600, timeout=1)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
    def send_angles(self,yaw_angle, pitch_angle):
        command = f"{yaw_angle},{pitch_angle}\n"
        self.ser.write(command.encode())
        print(f"Sent: {command.strip()}")

    def receive_head_neck_status(self):
        if self.ser.in_waiting > 0:
            status = self.ser.readline().decode()
            

#            print(f"Received: {status}")
            #time.sleep(1)
            #self.ser.reset_input_buffer()
            # Convert status from string "0.00,0.00" into self.neck_joint and self.head_joint
            try:
                neck_joint_str, head_joint_str = status.split(',')
                self.neck_joint = np.deg2rad(float(neck_joint_str))
                self.head_joint = np.deg2rad(float(head_joint_str))
            except ValueError as e:
                self.get_logger().error(f"Failed to convert status to joint values: {e}")
                self.neck_joint = 0.0
                self.head_joint = 0.0
            # Flush the input buffer after reading
            self.ser.reset_input_buffer()
            return print(f"Received: {status}")
        
        # return None
    



    def get_joint_angles(self):
        try:
            response = self.base.GetMeasuredJointAngles()
            #joint_positions = [joint.value for joint in response.joint_angles]
            joint_angle_rad = []
            for joint_angle in response.joint_angles:
                joint_angle_rad.append(np.deg2rad(joint_angle.value))
                #print(joint_angle.joint_identifier, " : ", joint_angle.value)
                
            #print(joint_angle_rad)                
            return joint_angle_rad
        except Exception as e:
            self.get_logger().error(f'Failed to get joint angles: {e}')
            return [0.0] * 6

    def publish_joint_states(self):
        
        self.receive_head_neck_status()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        self.gen3_lite_joint_1, self.gen3_lite_joint_2, self.gen3_lite_joint_3, self.gen3_lite_joint_4, self.gen3_lite_joint_5, self.gen3_lite_joint_6,  = self.get_joint_angles()
        #self.receive_head_neck_status()
        msg.position = [self.left_wheel_joint, self.right_wheel_joint, 
                        self.body_joint, self.neck_joint, self.head_joint, 
                        self.gen3_lite_joint_1, 
                        self.gen3_lite_joint_2, 
                        self.gen3_lite_joint_3, 
                        self.gen3_lite_joint_4,  
                        self.gen3_lite_joint_5, 
                        self.gen3_lite_joint_6, 
                        self.gripper_joint]
        
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published Joint States: {msg.position}')


def main(args=None):
    rclpy.init(args=args)
    #sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import sontana_script.utilities as utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        # Create required services
        base = BaseClient(router)

        # Example core
        success = True
        #success &= example_forward_kinematics(base)
        #success &= example_inverse_kinematics(base)
        
        #return 0 if success else 1
        node = Gen3l_JointStatePublisher(base)
        
        #joint_state = node.get_joint_angles()
       # node.publish_joint_states()
        # node.receive_head_neck_status()
        # if node.ser.in_waiting > 0:
        #     status = node.ser.readline().decode().strip()
        #     print(f"Received: {status}")
        
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
