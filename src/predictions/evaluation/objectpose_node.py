import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped,PoseStamped

class YellowBoxTracker(Node):
    def __init__(self):
        super().__init__('objectpose_node')
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.image_subscription = self.create_subscription(
            Image,
            'camera/color/image_raw',
            self.image_callback,
            10)
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            'camera/color/camera_info',
            self.camera_info_callback,
            10)
        self.depth_image_subscription = self.create_subscription(
            Image,
            'camera/depth/image_rect_raw',
            self.depth_image_callback,
            10)
        self.depth_image = None
        self.camera_info = None

         # Set up the publisher for the object poses
        #self.pose_publisher = self.create_publisher(PoseStamped, 'yellow_poses', 10)
        #self.transform_publisher = self.create_publisher(TransformStamped, 'yellow_transform',10)


    def camera_info_callback(self, msg):
        self.camera_info = msg

    def depth_image_callback(self, msg):
        self.depth_image = msg

    def image_callback(self, msg):
        if self.camera_info is not None and self.depth_image is not None:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                self.get_logger().error('CvBridgeError: %s' % str(e))
                return

            yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
            yellow_upper = np.array([30, 255, 255], dtype=np.uint8)

            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Assuming there's only one yellow box, take the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Calculate the position in meters using the depth image
                position_x, position_y, position_z = self.calculate_position(x, y, w, h)

                # Publish the transform with the updated position
                self.publish_transform(position_x, position_y, position_z)

    def calculate_position(self, x, y, w, h):
        if self.depth_image is not None and self.camera_info is not None:
            depth_image = self.bridge.imgmsg_to_cv2(self.depth_image)
            depth_scaling = 0.001  # RealSense typically scales depth in millimeters

            # Get camera intrinsic parameters
            fx = self.camera_info.k[0]
            fy = self.camera_info.k[4]
            cx = self.camera_info.k[2]
            cy = self.camera_info.k[5]

            # Calculate the depth value at the center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            #if center_x >= 0 and center_x < depth_image.shape[1] and center_y >= 0 and center_y < depth_image.shape[0]:
            # Check if the center coordinates are within the depth image boundaries
            if 0 <= center_x < depth_image.shape[1] and 0 <= center_y < depth_image.shape[0]:
                depth = depth_image[center_y, center_x] * depth_scaling
            else:
                # Handle the case where the center coordinates are out of bounds
                depth = 0.0

            # Convert pixel coordinates to metric coordinates
            x_m = (center_x - cx) * depth / fx
            y_m = (center_y - cy) * depth / fy
            z_m = depth

           # self.get_logger().info(f'  x: {x_m}  y: {y_m}  z: {z_m}')

            return x_m, y_m, z_m
        else:
            return 0.0, 0.0, 0.0

    def publish_transform(self, x, y, z):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'camera_link'  # Change to your desired frame
        transform.child_frame_id = 'yellow_box'

        # Set the translation and rotation
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z

        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0

                # Create a PoseStamped message and populate it with the pose data
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z

        # Publish the pose message
        #self.pose_publisher.publish(pose_msg)
        #self.transform_publisher.publish(transform)

        self.tf_broadcaster.sendTransform(transform)
        #self.get_logger().warn(f"Object pose {transform.transform.translation}")

def main(args=None):
    rclpy.init(args=args)
    node = YellowBoxTracker()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
