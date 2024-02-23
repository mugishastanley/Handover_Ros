import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from evaluate.msg import Poselandmarks
from geometry_msgs.msg import PointStamped, Point,TransformStamped
from cv_bridge import CvBridge
import cv2
import std_msgs
import numpy as np
import pyrealsense2

class LandmarkProcessorNode(Node):
    def __init__(self):
        super().__init__('armpose_node')

        self.bridge = CvBridge()
        self.landmarks = None
        self.camera_info = None
        self.intrinsics = None

        self.landmarks_subscription = self.create_subscription(Poselandmarks, '/landmarks', self.landmarks_callback, 10)
        self.camera_info_subscription = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)
        #self.camera_info_subscription = self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.camera_info_callback, 10)
        self.depth_image_subscription = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_image_callback, 10)
        #self.sub = self.create_subscription(Image,'/camera/depth/image_rect_raw', self.imageDepthCallback,10)

        self.publisher = self.create_publisher(Poselandmarks, '/landmark_3d_poses', 10)
        self.publisher2 = self.create_publisher(TransformStamped, '/hand_poses', 10)
        self.get_logger().info('Landmark Processor Node is ready')

    def landmarks_callback(self, msg):
        # Store landmarks when received
        self.landmarks = msg

    def camera_info_callback(self, msg):
        # Store camera information when received
        self.camera_info = msg


    def depth_image_callback(self, msg):
        if self.landmarks is not None and self.camera_info is not None:
            # Define the specific indices you want to select
            specific_indices = [12, 14, 16]  # Replace with the indices you want to process
            #Hand_indices = [16]  # Replace with the indices you want to process

            # Process depth image and landmarks for the specific indices
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            K = np.array(self.camera_info.k).reshape(3, 3)

            selected_indices = []
            selected_points = []

            for i, index in enumerate(self.landmarks.indices):
                if index in specific_indices:
                    x, y = self.landmarks.points[i].x * self.camera_info.width, self.landmarks.points[i].y * self.camera_info.height
                    #x,y = 640,360

                     # Ensure x and y are within the valid image dimensions
                    if x >= 0 and x < depth_image.shape[1] and y >= 0 and y < depth_image.shape[0]:
                        depth = depth_image[int(y), int(x)] / 1000.0  # Convert depth from mm to meters

                        if not np.isnan(depth):
                            # Convert (x, y, depth) to 3D camera frame pose
                            uv_point = np.array([x, y, 1.0])
                            uv_point_normalized = np.linalg.inv(K).dot(uv_point)
                            camera_frame_pose = Point()
                            camera_frame_pose.x = uv_point_normalized[0] * depth
                            camera_frame_pose.y = uv_point_normalized[1] * depth
                            camera_frame_pose.z = depth

                            selected_indices.append(index)
                            selected_points.append(camera_frame_pose)
                 
                            if index==16:
                                t = TransformStamped()
                                # Read message content and assign it to
                                # corresponding tf variables
                                t.header.stamp = self.get_clock().now().to_msg()
                                t.header.frame_id = 'camera_link'
                                t.child_frame_id = 'hand_link'

                                # Turtle only exists in 2D, thus we get x and y translation
                                # coordinates from the message and set the z coordinate to 0
                                t.transform.translation.x = camera_frame_pose.x
                                t.transform.translation.y = camera_frame_pose.y
                                t.transform.translation.z = camera_frame_pose.z

                                # For the same reason, turtle can only rotate around one axis
                                # and this why we set rotation in x and y to 0 and obtain
                                # rotation in z axis from the message
                                # = quaternion_from_euler(0, 0, 0)
                                t.transform.rotation.x = 0.0
                                t.transform.rotation.y = 0.0
                                t.transform.rotation.z = 0.0
                                t.transform.rotation.w = 1.0

                                # Send the transformation
                                #self.tf_broadcaster.sendTransform(t)
                                self.publisher2.publish(t)
                            self.get_logger().info(f"camera_frame_pose: {camera_frame_pose})") 


            # Create a new Poselandmarks message with selected indices and points
            #selected_landmarks_msg.header.stamp = self.get_clock().now().to_msg()
            selected_landmarks_msg = Poselandmarks()
            selected_landmarks_msg.header.stamp = self.get_clock().now().to_msg()
            selected_landmarks_msg.indices = selected_indices
            selected_landmarks_msg.points = selected_points

            # Publish the selected 3D poses in the camera frame
            self.publisher.publish(selected_landmarks_msg)

    def imageDepthCallback(self, data):
        if self.landmarks is not None and self.camera_info is not None:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            pix = (data.width/2, data.height/2)
            self.get_logger().info(f" Depth at center{pix[0], pix[1],cv_image[int(pix[1]), int(pix[0])]}")


            if self.intrinsics:
                depth = cv_image[pix[1], pix[0]]
                result = pyrealsense2.rs2_deproject_pixel_to_point(self.intrinsics, [pix[0], pix[1]], depth)
                
                self.get_logger().info(f" result{result}")
                # sys.stdout.write('%s: Depth at center(%d, %d): %f(mm)\r' % (self.topic, pix[0], pix[1], cv_image[pix[1], pix[0]]))
                # sys.stdout.flush()
                
            #     camera_frame_pose = Point()
            #     camera_frame_pose.x
            #     camera_frame_pose.y
            #     depth = depth_image[int(x), int(y)]
            #     camera_frame_pose.x,camera_frame_pose.y,camera_frame_pose.z = 
            #     selected_indices.append(index)
            #     selected_points.append(camera_frame_pose)
                
                

            # self.get_logger().info(f"camera_frame_pose: ({camera_frame_pose.x})")        

            # # Create a new Poselandmarks message with selected indices and points
            # selected_landmarks_msg = Poselandmarks()
            # selected_landmarks_msg.indices = selected_indices
            # selected_landmarks_msg.points = selected_points

            # # Publish the selected 3D poses in the camera frame
            # self.publisher.publish(selected_landmarks_msg)


 

def main(args=None):
    rclpy.init(args=args)
    node = LandmarkProcessorNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
