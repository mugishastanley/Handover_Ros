#!/usr/bin/env python3
#broadcast arm transform

import rospy
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_ros import TransformBroadcaster
import time

class FramePublisher_arm:
    def __init__(self):
        rospy.init_node('pred_cup_broadcaster')

        # Initialize the transform broadcasters
        self.tf_broadcaster_armtf = TransformBroadcaster()

        self.armpose_sub = rospy.Subscriber('/vrpn_client_node/cup/pose', PoseStamped, self.handle_arm_pose,queue_size=10)
        self.predarmpose_pub = rospy.Publisher('/pred_cup_tf', TransformStamped, queue_size=10)
        rospy.loginfo("Broadcasting arm mocap frames")
    
    def handle_arm_pose(self, msg):
        t = TransformStamped()

        # Read message content and assign it to corresponding tf variables
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = 'world'
        t.child_frame_id= 'pred_cup_link'

        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation.x = msg.pose.orientation.x
        t.transform.rotation.y = msg.pose.orientation.y
        t.transform.rotation.z = msg.pose.orientation.z
        t.transform.rotation.w = msg.pose.orientation.w
        rospy.loginfo("Broadcasting pred cup link frames")
        # Send the transformation
        rospy.sleep(0.04)
        self.tf_broadcaster_armtf.sendTransform(t)

def main():
    try:
        node = FramePublisher_arm()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
