import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped

class YellowBoxTracker(Node):
    def __init__(self):
        super().__init__('objectpose_node')
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.image_subscription = self.create_subscription(
            Image,
            'camera/color/image_raw',  # Adjust topic name based on your setup
            self.image_callback,
            10)
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            'camera/color/camera_info',  # Adjust topic name based on your setup
            self.camera_info_callback,
            10)
        self.camera_info = None

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def image_callback(self, msg):
        if self.camera_info is None:
            self.get_logger().warn("No camera info received yet. Waiting...")
            return

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

            # Convert pixel coordinates to metric coordinates
            position_x, position_y, position_z = self.calculate_position(x, y, w, h)

            # Publish the transform with the updated position
            self.publish_transform(position_x, position_y, position_z)

    def calculate_position(self, x, y, w, h):
        # Get camera intrinsic parameters
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        # Assuming depth is not available in this example
        # You will need to retrieve depth information from the RealSense camera

        # Convert pixel coordinates to metric coordinates (Z is not used in this example)
        x_m = (x + w / 2 - cx) / fx
        y_m = (y + h / 2 - cy) / fy
        z_m = 0.0

        return x_m, y_m, z_m

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

        self.tf_broadcaster.sendTransform(transform)

def main(args=None):
    rclpy.init(args=args)
    node = YellowBoxTracker()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

