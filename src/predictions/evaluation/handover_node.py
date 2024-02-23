"""
What Is this? This node depends on the mocap vrpn and the robot driver.
What does it do ? It receives Robot base pose and cup pose ,
tranform the poses into base link pose , calculates the handover pose
and send publishes it.

"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Pose, Vector3,PoseStamped
from tf2_ros import Buffer,TransformException
from tf2_ros import TransformListener,TransformBroadcaster  # Add this import
from geometry_msgs.msg import Vector3
import tf2_geometry_msgs
import numpy as np

class Handovernode(Node):

    def __init__(self):
        super().__init__('handover_node')

        self.subscription = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/RobotBase/pose',
            self.robotbase_tf_pub_callback,
            10  # QoS profile
        )

        self.subscription = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/cup/pose',
            self.hv_callback,
            10  # QoS profile
        )
        self.handover_pose_publisher = self.create_publisher(PoseStamped, 'handover_dir', 10)
        self.cup_pose_publisher = self.create_publisher(TransformStamped, 'cup_t_pose', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.stepsize = 0.1
        self.handover_threshold=0.05
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_broadcaster2 = TransformBroadcaster(self)
        self.subscription

    def robotbase_tf_pub_callback(self,msg):
        """ 
        This function subscribes to the robot base in mocap and broadcasts the transform
        """
        t = TransformStamped()
        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'

        # Turtle only exists in 2D, thus we get x and y translation
        # coordinates from the message and set the z coordinate to 0
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z

        # For the same reason, turtle can only rotate around one axis
        # and this why we set rotation in x and y to 0 and obtain
        # rotation in z axis from the message
        t.transform.rotation.x = msg.pose.orientation.x
        t.transform.rotation.y = msg.pose.orientation.y
        t.transform.rotation.z = msg.pose.orientation.z
        t.transform.rotation.w = msg.pose.orientation.w

        # Send the transformation
        #self.tf_broadcaster.sendTransform(t)

    def hv_callback(self, msg):
        #Transform lookup for cup pose.
        cup_t = TransformStamped()
        # Read message content and assign it to corresponding tf variables
        cup_t.header.stamp = self.get_clock().now().to_msg()
        cup_t.header.frame_id = 'world'
        cup_t.child_frame_id = 'cup_link'

        # Turtle only exists in 2D, thus we get x and y translation
        # coordinates from the message and set the z coordinate to 0
        cup_t.transform.translation.x = msg.pose.position.x
        cup_t.transform.translation.y = msg.pose.position.y
        cup_t.transform.translation.z = msg.pose.position.z

        # For the same reason, turtle can only rotate around one axis
        # and this why we set rotation in x and y to 0 and obtain
        # rotation in z axis from the message
        cup_t.transform.rotation.x = msg.pose.orientation.x
        cup_t.transform.rotation.y = msg.pose.orientation.y
        cup_t.transform.rotation.z = msg.pose.orientation.z
        cup_t.transform.rotation.w = msg.pose.orientation.w
        self.tf_broadcaster2.sendTransform(cup_t)
        self.cup_pose_publisher.publish(cup_t)

        try:
            transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            transform2 = self.tf_buffer.lookup_transform('base_link', 'cup_link', rclpy.time.Time())
            tool0_pose = transform.transform
            cup_pose = transform2.transform
            self.get_logger().info(f"Cup Position: x={cup_pose.translation.x}, y={cup_pose.translation.y}, z={cup_pose.translation.z}")

        except TransformException as ex:
            self.get_logger().warning("Failed to get tool0 pose or cup transform.")
            tool0_pose = None
            cup_pose = None

        if tool0_pose is not None:
            # Determine the handover pose.
            # Calculate relative velocity as change in position relative to tool0
            dist, normalised_direction = self.vector_distance_and_normalized_direction(cup_pose.translation,tool0_pose.translation)
            # if (dist < self.handover_threshold):
            #     handover_location = cup_pose.translation
            # else :
            #     dir_3d = self.stepsize * normalised_direction
            #     handover_location = Vector3()
            #     handover_location.x = cup_pose.translation.x + dir_3d[0]
            #     handover_location.y = cup_pose.translation.y + dir_3d[1] 
            #     handover_location.z = cup_pose.translation.z + dir_3d[2] 
            #     #handover_location = handover_location + cup_pose.translation

            
            # Publish the relative velocity
            #self.publish_pose(handover_location)
            # Create a PoseStamped message to publish the information
            pose_pub = PoseStamped()
            pose_pub.header = msg.header
            #pose_stamped.frame_id = 'base_link'
            #pose_stamped.pose = Pose(position=(handover_location[0],handover_location[1],handover_location[2]), orientation=cup_pose.rotation)
            pose_pub.pose.position.x = normalised_direction.x
            pose_pub.pose.position.y = normalised_direction.y
            pose_pub.pose.position.z = normalised_direction.z
            #pose_pub.pose.orientation = cup_pose.rotation

            # Publish the information to the 'handoverpose' topic

            self.get_logger().warn(f"handover direction {pose_pub}")
            self.handover_pose_publisher.publish(pose_pub)

    def vector_distance_and_normalized_direction(self, cup_pose_position, tool0_pose_position):
        # Calculate the Euclidean distance between the two vectors
        cup_pose = np.array([cup_pose_position.x, cup_pose_position.y, cup_pose_position.z]) 
        tool0_pose = np.array([tool0_pose_position.x, tool0_pose_position.y, tool0_pose_position.z])

        #using cup_pose as reference
        distance = np.linalg.norm(tool0_pose - cup_pose )
        # Calculate the direction vector
        direction_vector = (tool0_pose-cup_pose)
        # Normalize the direction vector
        normalized_direction = direction_vector / np.linalg.norm(direction_vector)
        
        return distance, normalized_direction

def main(args=None):
    rclpy.init(args=args)
    aruco_subscriber_publisher = Handovernode()
    rclpy.spin(aruco_subscriber_publisher)
    aruco_subscriber_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
