#!/usr/bin/env python3
#broadcast arm transform

import rospy
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_ros import TransformBroadcaster
import time

class FramePublisher_cup:
    def __init__(self):
        rospy.init_node('mocap_cup_broadcaster')

        # Initialize the transform broadcasters
        self.tf_broadcaster_cuptf = TransformBroadcaster()

        # callback function on each message
        self.subscription_arm = rospy.Subscriber(
            '/vrpn_client_node/cup/pose',
            PoseStamped,
            self.handle_cup_pose
        )

        rospy.loginfo("Broadcasting cup mocap frames")
    
    def handle_cup_pose(self, msg):
        t = TransformStamped()

        # Read message content and assign it to corresponding tf variables
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = 'world'
        t.child_frame_id = 'cup_link'

        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation.x = msg.pose.orientation.x
        t.transform.rotation.y = msg.pose.orientation.y
        t.transform.rotation.z = msg.pose.orientation.z
        t.transform.rotation.w = msg.pose.orientation.w
        rospy.loginfo("Broadcasting cup link frames")
        # Send the transformation
        rospy.sleep(0.005)
        self.tf_broadcaster_cuptf.sendTransform(t)

def main():
    try:
        node = FramePublisher_cup()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
