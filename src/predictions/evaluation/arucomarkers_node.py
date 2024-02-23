import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from ros2_aruco_interfaces.msg import ArucoMarkers
from geometry_msgs.msg import PoseStamped, Vector3
import time
import math

class ArucoSubscriberPublisher(Node):

    def __init__(self):
        super().__init__('arucomarkers_node')

        self.subscription = self.create_subscription(
            ArucoMarkers,
            '/aruco_markers',
            self.aruco_callback,
            10  # QoS profile
        )
        self.subscription

        self.prev_position = {}  # Dictionary to store previous positions for each marker
        self.prev_time = {}  # Dictionary to store previous timestamps for each marker
        self.velocity_publishers = {}  # Dictionary to store velocity publishers for each identified marker

    def aruco_callback(self, msg):
        for i in range(len(msg.poses)):
            marker_id = msg.marker_ids[i]
            position = msg.poses[i].position
            orientation = msg.poses[i].orientation

            # Calculate velocity as change in position over time
            velocity = self.calculate_velocity(marker_id, position, msg.header.stamp)

            # Create a PoseStamped message to publish the information
            pose_stamped = PoseStamped()
            pose_stamped.header = msg.header
            pose_stamped.pose = Pose(position=position, orientation=orientation)

            # Publish the information to the 'aruco_info' topic
            aruco_info_publisher = self.create_publisher(PoseStamped, f'aruco_info_{marker_id}', 10)
            aruco_info_publisher.publish(pose_stamped)

            # Publish velocity information only for identified markers
            if velocity is not None:
                self.publish_velocity(velocity, marker_id)

            self.get_logger().info(f"Published Marker {marker_id} information to 'aruco_info' topic.")
            self.get_logger().info(f"Position: x={position.x}, y={position.y}, z={position.z}")
            self.get_logger().info(f"Orientation: x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w}")
            self.get_logger().info(f"Velocity: {velocity}\n")

    def publish_velocity(self, velocity, marker_id):
        if marker_id not in self.velocity_publishers:
            self.velocity_publishers[marker_id] = self.create_publisher(Vector3, f'aruco_velocity_{marker_id}', 10)
        
        velocity_msg = Vector3()
        velocity_msg.x = velocity
        self.velocity_publishers[marker_id].publish(velocity_msg)

    def calculate_velocity(self, marker_id, position, current_time):
        if marker_id not in self.prev_position:
            self.prev_position[marker_id] = position
            self.prev_time[marker_id] = current_time
            return None  # No previous data for this marker

        # Calculate displacement
        displacement = math.sqrt(
            (position.x - self.prev_position[marker_id].x) ** 2 +
            (position.y - self.prev_position[marker_id].y) ** 2 +
            (position.z - self.prev_position[marker_id].z) ** 2
        )

        # Calculate time difference
        time_diff = (current_time.sec - self.prev_time[marker_id].sec) + (current_time.nanosec - self.prev_time[marker_id].nanosec) / 1e9

        # Calculate velocity as displacement over time
        velocity = displacement / time_diff if time_diff > 0 else 0.0

        # Update previous position and time for this marker
        self.prev_position[marker_id] = position
        self.prev_time[marker_id] = current_time

        return velocity

def main(args=None):
    rclpy.init(args=args)
    aruco_subscriber_publisher = ArucoSubscriberPublisher()
    rclpy.spin(aruco_subscriber_publisher)
    aruco_subscriber_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
