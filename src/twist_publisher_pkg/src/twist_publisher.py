#!/usr/bin/env python3
# publisher node

import rospy
from geometry_msgs.msg import TwistStamped

def twist_publisher():
    # Initialize the ROS node
    rospy.init_node('twist_publisher', anonymous=True)

    # Create a publisher for the TwistStamped message on the specified topic
    pub = rospy.Publisher('/servo_server/delta_twist_cmds', TwistStamped, queue_size=10)

        # Set the rate at which to publish the message (100 Hz in this case)
    rate = rospy.Rate(100)

    # # Create a TwistStamped message
    # twist_msg = TwistStamped()
    # twist_msg.header.stamp = rospy.Time.now()
    # twist_msg.header.frame_id = "base_link"
    # twist_msg.twist.linear.x = 0.0
    # twist_msg.twist.linear.y = 0.01
    # twist_msg.twist.linear.z = -0.01
    # twist_msg.twist.angular.x = 0.0
    # twist_msg.twist.angular.y = 0.0
    # twist_msg.twist.angular.z = 0.0

    def twist_callback(msg):
        # Update the timestamp before republishing
        msg.header.stamp = rospy.Time.now()
        # Publish the received TwistStamped message on the new topic
        pub.publish(msg)
    rospy.Subscriber('/robot_twist', TwistStamped, twist_callback)

    while not rospy.is_shutdown():
        # Sleep to maintain the specified publishing rate
        rate.sleep()

if __name__ == '__main__':
    try:
        twist_publisher()
    except rospy.ROSInterruptException:
        pass
