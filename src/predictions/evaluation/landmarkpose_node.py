#This node detects humn poses and exports the landmarks
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from evaluate.msg import Poselandmarks  # Create a custom message type for landmarks
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import cv2
import mediapipe as mp
import time
from std_msgs.msg import Header


class PoseDet():
    def __init__(self, detectionCon=0.5, trackCon=0.5):
        self.detectionCon = detectionCon
        self.trackCon= trackCon

        self.mpPose = mp.solutions.pose
        self.mp_pose = self.mpPose.Pose(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def findLandmarks(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.mp_pose.process(imgRGB)
        self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, 
            self.mpPose.POSE_CONNECTIONS, 
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        return img
    
    def findLandmarks2(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.mp_pose.process(imgRGB)

        landmarks_3d = []  # List to store 3D pose landmarks
        landmarks_results = [] 
        if self.results.pose_landmarks:
            for index, landmark in enumerate(self.results.pose_landmarks.landmark):
                    x, y, z = landmark.x, landmark.y, landmark.z
                    landmarks_3d.append((index, (x, y, z)))
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                    self.mpPose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        #return landmarks_3d
        return self.results,landmarks_3d

    
class mediapipeNode(Node):
    def __init__(self):
        super().__init__('mediapipe_demo')

        topic_name = "/camera/color/image_raw"
        topic_name2 = "/landmarks"
        self.get_logger().info('initializing mediapipe..')
        self.detector = PoseDet(detectionCon=0.3, trackCon=0.3)

        self.publisher_ = self.create_publisher(Image, topic_name , 10)
        self.publisher2_ = self.create_publisher(Poselandmarks, topic_name2 , 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.cap = cv2.VideoCapture(0)
        self.br = CvBridge()

        self.subscription = self.create_subscription(Image, topic_name, self.img_callback, 1)
        self.subscription 
        self.br = CvBridge()


    def timer_callback(self):
        ret, frame = self.cap.read()     
        if ret == True:
            self.publisher_.publish(self.br.cv2_to_imgmsg(frame))
            self.get_logger().info('Publishing frame for process')


    def img_callback(self, data):
        current_frame = self.br.imgmsg_to_cv2(data)
        # Create a header for the message

        ##dectect
        processed_img = self.detector.findLandmarks(current_frame, draw=True)
        res,processed_img2 = self.detector.findLandmarks2(current_frame, draw=False)
        self.get_logger().info('mediapipe processed img')

        msg = Poselandmarks()
        #msg.landmarks = []  # Initialize the landmarks array
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.indices = []  # Initialize the indices array
        msg.points = []  # Initialize the poses array

        for index, (x,y,z) in processed_img2:
            #msg.index = index  # Set the index field
            landmark = Point()
            landmark.x = x
            landmark.y = y
            landmark.z = z
            msg.indices.append(index)  # Set the index field
            msg.points.append(landmark)  # Append each landmark to the array

        self.publisher2_.publish(msg)
        self.get_logger().info(f'MediaPipe processed landmarks:x')
        for i in range(len(msg.indices)):
            self.get_logger().info(f'Landmark #{msg.indices[i]}:')
            self.get_logger().info(f'  x: {msg.points[i].x}\n  y: {msg.points[i].y}\n  z: {msg.points[i].z}')
            
        cv2.imshow("demo", processed_img)   
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = mediapipeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

  
if __name__ == '__main__':
  main()
