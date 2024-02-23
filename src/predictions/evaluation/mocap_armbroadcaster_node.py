"""This code broadcasts poses received from mocap """
import math

from geometry_msgs.msg import TransformStamped,PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
import time

class FramePublisher(Node):
    def __init__(self):
        super().__init__('mocap_armbroadcaster_node')

        # Initialize the transform broadcasters
        # self.tf_broadcaster_rbttf = TransformBroadcaster(self)
        # self.tf_broadcaster_cuptf = TransformBroadcaster(self)
        self.tf_broadcaster_armtf = TransformBroadcaster(self)

        # Subscribe to a  topic and call handle_turtle_pose
        # callback function on each message
        # self.subscription_rbt = self.create_subscription(
        #     PoseStamped,
        #     '/vrpn_mocap/RobotBase/pose',
        #     self.handle_robot_pose,
        #     10)
        
        # self.subscription_cup = self.create_subscription(
        #     PoseStamped,
        #     '/vrpn_mocap/cup/pose',
        #     self.handle_cup_pose,
        #     10)
        
        self.subscription_arm = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/Arm/pose',
            self.handle_arm_pose,
            10)
        
        self.get_logger().info("Broadcasting arm mocap frames")
        #self.subscription  # prevent unused variable warning

    
    # def handle_robot_pose(self, msg):
    #     t = TransformStamped()

    #     # Read message content and assign it to corresponding tf variables
    #     t.header.stamp = self.get_clock().now().to_msg()
    #     t.header.frame_id = 'world'
    #     t.child_frame_id = 'base_link'

    #     t.transform.translation.x = msg.pose.position.x
    #     t.transform.translation.y = msg.pose.position.y
    #     t.transform.translation.z = msg.pose.position.z
    #     t.transform.rotation.x = msg.pose.orientation.x
    #     t.transform.rotation.y = msg.pose.orientation.y
    #     t.transform.rotation.z = msg.pose.orientation.z
    #     t.transform.rotation.w = msg.pose.orientation.w
    #     # Send the transformation
    #     #Transform is fixed for now
    #     #self.tf_broadcaster_rbttf.sendTransform(t)
    
    # def handle_cup_pose(self, msg):
    #     t = TransformStamped()

    #     # Read message content and assign it to corresponding tf variables
    #     t.header.stamp = self.get_clock().now().to_msg()
    #     t.header.frame_id = 'world'
    #     t.child_frame_id = 'cup_link'

    #     t.transform.translation.x = msg.pose.position.x
    #     t.transform.translation.y = msg.pose.position.y
    #     t.transform.translation.z = msg.pose.position.z
    #     t.transform.rotation.x = msg.pose.orientation.x
    #     t.transform.rotation.y = msg.pose.orientation.y
    #     t.transform.rotation.z = msg.pose.orientation.z
    #     t.transform.rotation.w = msg.pose.orientation.w
    #     self.get_logger().info("Broadcasting cup link frames")
    #     # Send the transformation
    #     time.sleep(0.01)
    #     self.tf_broadcaster_cuptf.sendTransform(t)
    
    def handle_arm_pose(self, msg):
        t = TransformStamped()

        # Read message content and assign it to corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'arm_link'

        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation.x = msg.pose.orientation.x
        t.transform.rotation.y = msg.pose.orientation.y
        t.transform.rotation.z = msg.pose.orientation.z
        t.transform.rotation.w = msg.pose.orientation.w
        self.get_logger().info("Broadcasting arm link frames")
        # Send the transformation
        self.tf_broadcaster_armtf.sendTransform(t)
        


def main():
    rclpy.init()
    node = FramePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()