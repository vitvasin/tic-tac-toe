#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###
import math
import sys

from collections.abc import MutableMapping
from collections.abc import MutableSequence
import collections
collections.MutableMapping = collections.abc.MutableMapping
collections.MutableSequence = collections.abc.MutableSequence
import os
import time
import threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check
 
def example_move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Sontana":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished

def example_angular_action_movement(base):
    
    print("Starting angular action movement ...")
    action = Base_pb2.Action()
    action.name = "Example angular action movement"
    action.application_data = ""

    actuator_count = base.GetActuatorCount()

    # Place arm straight up
    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = 0

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Angular movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def example_angular_action_test(base):
    
    print("Starting angular action movement ...")
    action = Base_pb2.Action()
    action.name = "Example angular action movement"
    action.application_data = ""

    actuator_count = base.GetActuatorCount()

    joint_values = [92,288,210,98,347,269] # Example values for each joint

    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = joint_values[joint_id]
        
        
    # Place arm on home
    

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Angular movement completed")
    else:
        print("Timeout on action notification wait")
    return finished
    

def example_cartesian_action_movement(base, base_cyclic):
    
    print("Starting Cartesian action movement ...")
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = feedback.base.tool_pose_x          # (meters)
    cartesian_pose.y = feedback.base.tool_pose_y - 0.1    # (meters)
    cartesian_pose.z = feedback.base.tool_pose_z - 0.2    # (meters)
    cartesian_pose.theta_x = feedback.base.tool_pose_theta_x # (degrees)
    cartesian_pose.theta_y = feedback.base.tool_pose_theta_y # (degrees)
    cartesian_pose.theta_z = feedback.base.tool_pose_theta_z # (degrees)

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished


    
    
    
    
    
def draw_circle_on_xz_plane(radius, center_y, center_z, num_points):
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        y = center_y + radius * math.cos(angle)
        z = center_z + radius * math.sin(angle)
        points.append((y, z))
    return points

def move_robot_arm_to(base, base_cyclic, x, y, z, theta_x, theta_y, theta_z):
    action = Base_pb2.Action()
    action.name = "Move to point"
    action.application_data = ""
    
    base.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE

    feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = x
    cartesian_pose.y = y
    cartesian_pose.z = z
    cartesian_pose.theta_x = theta_x
    cartesian_pose.theta_y = theta_y
    cartesian_pose.theta_z = theta_z

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteAction(action)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    return finished


def draw_circle_XZ(base, base_cyclic, radius, num_points):
    print("Starting Cartesian action movement ...")
    
    feedback = base_cyclic.RefreshFeedback()
    center_x = feedback.base.tool_pose_x
    center_y = feedback.base.tool_pose_y 
    center_z = feedback.base.tool_pose_z 
    theta_x = feedback.base.tool_pose_theta_x
    theta_y = feedback.base.tool_pose_theta_y
    theta_z = feedback.base.tool_pose_theta_z

    move_robot_arm_to(base, base_cyclic, center_x+0.05, center_y, center_z, theta_x, theta_y, theta_z)
    time.sleep(0.5)
    
    # radius = 0.1  # Example radius
    # num_points = 100  # Number of points to draw the circle
    circle_points = draw_circle_on_xz_plane(radius, center_y, center_z, num_points)

    for point in circle_points:
        y, z = point
        success = move_robot_arm_to(base, base_cyclic, center_x, y, z, theta_x, theta_y, theta_z)
        if not success:
            print("Failed to move to point:", point)
            break
        #time.sleep(0.01)

    print("Finished drawing circle on XZ plane")
    move_robot_arm_to(base, base_cyclic, center_x+0.02, center_y, center_z, theta_x, theta_y, theta_z)
    time.sleep(0.5)
    move_robot_arm_to(base, base_cyclic, center_x, center_y, center_z, theta_x, theta_y, theta_z)

def draw_cross_XZ(base, base_cyclic, radius, num_points):
    print("Starting Cartesian action movement to draw cross ...")
    
    feedback = base_cyclic.RefreshFeedback()
    center_x = feedback.base.tool_pose_x
    center_y = feedback.base.tool_pose_y 
    center_z = feedback.base.tool_pose_z 
    theta_x = feedback.base.tool_pose_theta_x
    theta_y = feedback.base.tool_pose_theta_y
    theta_z = feedback.base.tool_pose_theta_z
    
    move_robot_arm_to(base, base_cyclic, center_x, center_y+0.03, center_z, theta_x, theta_y, theta_z)
    time.sleep(1)

    # Calculate points for the cross
    cross_points = []
    step = radius / (num_points // 2)
    
    # Diagonal line 1
    for i in range(-num_points // 2, num_points // 2 + 1):
        if i == 0:
            continue
        cross_points.append((center_x + i * step, center_z + i * step))
    
    for point in cross_points:
        x, z = point
        success = move_robot_arm_to(base, base_cyclic, x, center_y, z, theta_x, theta_y, theta_z)
        if not success:
            print("Failed to move to point:", point)
            break
        #time.sleep(0.01)

    # Move back to center before drawing the second diagonal line
    #success = move_robot_arm_to(base, base_cyclic, center_x+0.05, center_y, center_z, theta_x, theta_y, theta_z)
    if not success:
        print("Failed to move back to center")
        return
    #time.sleep(1)

    # Diagonal line 2
    cross_points = []
    for i in range(-num_points // 2, num_points // 2 + 1):
        if i == 0:
            continue
        cross_points.append((center_x + i * step, center_z - i * step))

    for point in cross_points:
        x, z = point
        success = move_robot_arm_to(base, base_cyclic, x, center_y, z, theta_x, theta_y, theta_z)
        if not success:
            print("Failed to move to point:", point)
            break
        #time.sleep(0.01)

    print("Finished drawing cross on XZ plane")
    #move_robot_arm_to(base, base_cyclic, center_x+0.05, center_y, center_z, theta_x, theta_y, theta_z)

def draw_circle_with_twist_command(base,base_cyclic, radius, duration):
    
    print("Starting Cartesian action movement to draw circle ...")
    
    feedback = base_cyclic.RefreshFeedback()
    center_x = feedback.base.tool_pose_x
    center_y = feedback.base.tool_pose_y 
    center_z = feedback.base.tool_pose_z 
    theta_x = feedback.base.tool_pose_theta_x
    theta_y = feedback.base.tool_pose_theta_y
    theta_z = feedback.base.tool_pose_theta_z
    
    # move_robot_arm_to(base, base_cyclic, center_x+0.05, center_y, center_z, theta_x, theta_y, theta_z)
    # time.sleep(0.5)
    # move_robot_arm_to(base, base_cyclic, center_x, center_y+0.02, center_z, theta_x, theta_y, theta_z)
    # time.sleep(0.5)
    
    command = Base_pb2.TwistCommand()
    command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
    command.duration = 0

    twist = command.twist
    #twist.linear_y = 0
    twist.linear_z = 0
    twist.angular_x = 0
    twist.angular_y = 0
    twist.angular_z = 0
    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time
        twist.linear_x = radius * math.cos(2 * math.pi * elapsed_time / duration)
        twist.linear_y = radius * math.sin(2 * math.pi * elapsed_time / duration)
        #twist.angular_x = 2 * math.pi / duration

        base.SendTwistCommand(command)
        #time.sleep(0.1)  # Adjust the sleep time as needed

    print("Stopping the robot...")
    base.Stop()
    time.sleep(1)
    
    # move_robot_arm_to(base, base_cyclic, center_x+0.05, center_y, center_z, theta_x, theta_y, theta_z)
    # time.sleep(0.5)
    # move_robot_arm_to(base, base_cyclic, center_x, center_y, center_z, theta_x, theta_y, theta_z)

    return True

def main():
    radius = 0.1  # Example radius
    num_points = 5  # Number of points to draw the circle
    
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        # Create required services
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        # Example core
        success = True

        success &= example_move_to_home_position(base)
       # time.sleep(1)
       # move_robot_arm_to(base, base_cyclic, 0.10, -0.30,0.30, 100, 100, 20)
       # time.sleep(2)
       # move_robot_arm_to(base, base_cyclic, 0.10, -0.5,0.30, 100, 100, 20)
       # time.sleep(2)
       # move_robot_arm_to(base, base_cyclic, 0.10, -0.30,0.30, 100, 100, 20)
        # time.sleep(1)
        # draw_cross_XZ(base, base_cyclic, 0.01, 2)
        # time.sleep(2)
        # draw_circle_with_twist_command(base,base_cyclic, 0.05, 2)

        return 0 if success else 1

if __name__ == "__main__":
    exit(main())
