import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, Vector3
from ros2_aruco_interfaces.msg import ArucoMarkers
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer,TransformException
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformListener  # Add this import
import tf2_geometry_msgs
from builtin_interfaces.msg import Time, Duration
from geometry_msgs.msg import Twist
import math


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Vector3, TwistStamped
from ros2_aruco_interfaces.msg import ArucoMarkers
from tf2_ros import TransformListener
import time
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Vector3, TwistStamped
from ros2_aruco_interfaces.msg import ArucoMarkers

class Dangernode(Node):

    def __init__(self):
        super().__init__('dangerindex_node')

        self.subscription = self.create_subscription(
            ArucoMarkers,
            '/aruco_markers',
            self.aruco_callback,
            10  # QoS profile
        )

        self.velocity_publishers = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.prev_positions = {}  # Store the previous positions for each marker


        self.prev_position = {}  # Dictionary to store previous positions for each marker
        self.velocity_publishers = {}  # Dictionary to store velocity publishers for each identified marker

    def aruco_callback(self, msg):
        for i in range(len(msg.poses)):
            marker_id = msg.marker_ids[i]
            position = msg.poses[i].position
            orientation = msg.poses[i].orientation

            try:
                transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
                tool0_pose = transform.transform
            except TransformException as ex:
                self.get_logger().warning("Failed to get tool0 pose.")
                tool0_pose = None

            if tool0_pose is not None:
                # Calculate relative velocity as change in position relative to tool0
                relative_velocity = self.calculate_relative_velocity(marker_id, position, tool0_pose)

                # Publish the relative velocity
                self.publish_relative_velocity(relative_velocity, marker_id)

                self.get_logger().info(f"Published Relative Velocity for Marker {marker_id}: {relative_velocity}\n")

            # Create a PoseStamped message to publish the information
            pose_stamped = PoseStamped()
            pose_stamped.header = msg.header
            pose_stamped.pose = Pose(position=position, orientation=orientation)

            # Publish the information to the 'aruco_info' topic
            aruco_info_publisher = self.create_publisher(PoseStamped, f'aruco_info_{marker_id}', 10)
            aruco_info_publisher.publish(pose_stamped)

            self.get_logger().info(f"Published Marker {marker_id} information to 'aruco_info' topic.")
            self.get_logger().info(f"Position: x={position.x}, y={position.y}, z={position.z}")
            self.get_logger().info(f"Orientation: x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w}")

    def publish_relative_velocity(self, relative_velocity, marker_id):
        if marker_id not in self.velocity_publishers:
            self.velocity_publishers[marker_id] = self.create_publisher(Vector3, f'aruco_relative_velocity_{marker_id}', 10)
        
        velocity_msg = Vector3()
        velocity_msg.x = relative_velocity.x
        velocity_msg.y = relative_velocity.y
        velocity_msg.z = relative_velocity.z
        self.velocity_publishers[marker_id].publish(velocity_msg)

    def calculate_relative_velocity(self, marker_id, position, tool0_pose):
        if marker_id not in self.prev_position:
            self.prev_position[marker_id] = position
            return Vector3()  # No previous data for this marker

            #  Calculate the relative position
        # Calculate the relative position
        relative_position = Vector3()
        relative_position.x = position.x - tool0_pose.translation.x
        relative_position.y = position.y - tool0_pose.translation.y
        relative_position.z = position.z - tool0_pose.translation.z

        # Calculate the relative velocity as change in relative position
        relative_velocity = Vector3()
        relative_velocity.x = relative_position.x - self.prev_position[marker_id].x
        relative_velocity.y = relative_position.y - self.prev_position[marker_id].y
        relative_velocity.z = relative_position.z - self.prev_position[marker_id].z

        # Update the previous position for this marker
        self.prev_position[marker_id] = relative_position
        return relative_velocity

def main(args=None):
    rclpy.init(args=args)
    aruco_subscriber_publisher = Dangernode()
    rclpy.spin(aruco_subscriber_publisher)
    aruco_subscriber_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


# class Dangernode(Node):

#     def __init__(self):
#         super().__init__('dangerindex_node')

#         self.subscription = self.create_subscription(
#             ArucoMarkers,
#             '/aruco_markers',
#             self.aruco_callback,
#             10  # QoS profile
#         )
#         self.subscription

#         self.velocity_publishers = {}
#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)

#         self.prev_positions = {}  # Store the previous positions for each marker

#     def aruco_callback(self, msg):
#         for i in range(len(msg.poses)):
#             marker_id = msg.marker_ids[i]
#             position = msg.poses[i].position
#             orientation = msg.poses[i].orientation

#             # Calculate relative velocity with respect to a transform
#             relative_velocity = self.calculate_relative_velocity(marker_id, position)

#             # Create a PoseStamped message to publish the information
#             pose_stamped = PoseStamped()
#             pose_stamped.header = msg.header
#             pose_stamped.pose = Pose(position=position, orientation=orientation)

#             # Publish the information to the 'aruco_info' topic
#             aruco_info_publisher = self.create_publisher(PoseStamped, f'aruco_info_{marker_id}', 10)
#             aruco_info_publisher.publish(pose_stamped)

#             # Publish relative velocity information
#             if relative_velocity is not None:
#                 self.publish_velocity(relative_velocity, marker_id)

#             self.get_logger().info(f"Published Marker {marker_id} information to 'aruco_info' topic.")
#             self.get_logger().info(f"Position: x={position.x}, y={position.y}, z={position.z}")
#             self.get_logger().info(f"Orientation: x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w}")
#             self.get_logger().info(f"Relative Velocity: {relative_velocity}\n")

#     def publish_velocity(self, velocity, marker_id):
#         if marker_id not in self.velocity_publishers:
#             self.velocity_publishers[marker_id] = self.create_publisher(Vector3, f'aruco_velocity_{marker_id}', 10)
        
#         self.velocity_publishers[marker_id].publish(velocity)
       

#     def calculate_relative_velocity(self, marker_id, position):
#         try:
#             #transform = self.tf_buffer.lookup_transform('world_frame', f'aruco_frame_{marker_id}', rclpy.time.Time())
#             transform = self.tf_buffer.lookup_transform('world', 'camera_color_optical_frame', rclpy.time.Time())
#         except TransformException as ex:
#             # Handle transform lookup exceptions
#             self.get_logger().info(f"Failed to lookup transform for marker {marker_id}")
#             return None

#         # Calculate the relative velocity using the transform
#         #velocity = tf2_geometry_msgs.do_transform_vector3(position, transform)

#         # msg=Twist()
#         # msg.linear.x = math.sqrt(transform.transform.translation.x**2)
#         # msg.linear.y = math.sqrt(transform.transform.translation.y**2)
#         # msg.linear.z = math.sqrt(transform.transform.translation.z**2)

#         # velocity = Vector3(msg.linear)

#         velocity = Vector3()
#         velocity.x = abs(transform.transform.translation.x  - position.x)
#         velocity.y = abs(transform.transform.translation.y  - position.y)
#         velocity.z = abs(transform.transform.translation.z  - position.z)
#         return velocity 

# def main(args=None):
#     rclpy.init(args=args)
#     aruco_subscriber_publisher = Dangernode()
#     rclpy.spin(aruco_subscriber_publisher)
#     aruco_subscriber_publisher.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
