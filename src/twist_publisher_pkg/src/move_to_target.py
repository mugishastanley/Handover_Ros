#!/usr/bin/env python3

import sys
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3

from moveit_msgs.msg import CollisionObject
from moveit_msgs.srv import ApplyPlanningScene
# from moveit_python import MoveGroupInterface, PlanningSceneInterface
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
from robotiq_hande_ros_driver.srv import gripper_service

class ObstaclesNode:

    def __init__(self):
        rospy.init_node('move_robot_node')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.publisher = rospy.Publisher('robot_pose', PoseStamped, queue_size=10)
        self.publisher2 = rospy.Publisher('robot_twist', TwistStamped, queue_size=10)

        # self.arm_group = MoveGroupInterface("arm")
        # self.planning_scene = PlanningSceneInterface("base_link")
        # self.apply_planning_scene_srv = rospy.ServiceProxy('/apply_planning_scene', ApplyPlanningScene)
                ## gripper messages
        # Initialize the gripper service
        self.gripper_srv = rospy.ServiceProxy('gripper_service', gripper_service)
        rospy.loginfo("gripper initilised")
        self.isgripper_open  = True
        self.open_gripper()
        #moveit stuff
        moveit_commander.roscpp_initialize(sys.argv)
        robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

         # Set the desired loop rate to 100 Hz
        self.rate = rospy.Rate(100)
        # rospy.Timer(rospy.Duration(10), self.on_timer)
            # Run the loop continuously
        while not rospy.is_shutdown():
            self.on_timer(None)
            self.rate.sleep()

    def on_timer(self, event):
        from_frame_rel = "base_link"
        to_cup_frame = "cup_link"
        to_arm_frame = "arm_link"
        to_ee_frame = "wrist_3_link"
        arm_radius = 0.15
        d_max = 0.10
        v_min = 0.1
        k_d = 1.0
        k_v = 1.0
        rospy.loginfo("Timer called")

        try:
            t = self.tf_buffer.lookup_transform(from_frame_rel, to_cup_frame, rospy.Time())
            t2 = self.tf_buffer.lookup_transform(from_frame_rel, to_arm_frame, rospy.Time())
            t3 = self.tf_buffer.lookup_transform(from_frame_rel, to_ee_frame, rospy.Time())
        except tf2_ros.LookupException as ex:
            rospy.loginfo("Could not transform: %s", str(ex))
            return

        cup_pose = t.transform
        arm_pose = t2.transform
        ee_pose = t3.transform

        result_to_cup_dist, result_to_cup_dir = self.vector_distance_and_normalized_direction(ee_pose, cup_pose)
        result_danger_dist, result_danger_dir = self.vector_distance_and_normalized_direction(arm_pose, ee_pose)

        # rospy.loginfo("cup_pose %f , %f, %f ",cup_pose.translation.x,cup_pose.translation.y,cup_pose.translation.z)


        # self.move_collision_box_to_pose(arm_pose)

        # if self.isgripper_open:
        #     rospy.loginfo("Gripper is %f" , self.isgripper_open)
        # if result_danger_dist < d_max:
        #     self.move_robot(result_danger_dist, result_danger_dir, ee_pose, arm_pose)
        # else:
        self.move_robot(result_to_cup_dist, result_to_cup_dir, cup_pose, ee_pose)
        self.move_box(cup_pose)
        self.move_box(arm_pose)

    def vector_distance_and_normalized_direction(self, vector1, vector2):
        distance = np.linalg.norm(np.array([vector2.translation.x, vector2.translation.y, vector2.translation.z]) -
                                  np.array([vector1.translation.x, vector1.translation.y, vector1.translation.z]))
        
        direction_vector = np.array([vector2.translation.x - vector1.translation.x,
                                    vector2.translation.y - vector1.translation.y,
                                    vector2.translation.z - vector1.translation.z])
        
        normalized_direction = direction_vector / np.linalg.norm(direction_vector)
        
        rospy.loginfo("direction norm %f , %f, %f ",normalized_direction[0],normalized_direction[1],normalized_direction[2])
        # rospy.loginfo("vector 1 %f , %f, %f ",vector2.translation.x,vector2.translation.y,vector2.translation.z)
        rospy.loginfo("vector 1 distance %f  ",distance)
        return distance, normalized_direction

    # def move_collision_box_to_pose(self, target_pose):
    #     rospy.loginfo("Moving arm to new pose")

    #     collision_object = CollisionObject()
    #     collision_object.header.frame_id = "base_link"
    #     collision_object.id = "box1"
    #     collision_object.operation = collision_object.REMOVE

    #     self.apply_planning_scene_srv(collision_object)

    #     collision_object.operation = collision_object.ADD
    #     collision_object.primitives.append(CollisionObject.PRIMITIVE_BOX)
    #     collision_object.primitive_poses.append(target_pose)
    #     self.planning_scene.applyCollisionObject(collision_object)

    def move_robot(self, distance, direction_norm, cup_pose, ee_pose):
        #rospy.loginfo("Moving robot")
        gripper_offset = 0.2
        scaler = 0.03
        distance_threshold = 0.17

        handover_pose = PoseStamped()
        handover_pose.header.stamp = rospy.Time.now()
        handover_pose.header.frame_id = "base_link"
        handover_pose.pose.orientation.x = 0.0
        handover_pose.pose.orientation.y = 0.0
        handover_pose.pose.orientation.z = 0.0
        handover_pose.pose.orientation.w = 1.0

        if distance < distance_threshold:
            handover_pose.pose.position.x = cup_pose.translation.x
            handover_pose.pose.position.y = cup_pose.translation.y - gripper_offset
            handover_pose.pose.position.z = cup_pose.translation.z + 0.02
            self.close_gripper()
            
        else:
            # delta_pose = np.array(scaler * np.array([direction_norm.x, direction_norm.y, direction_norm.z]))
            delta_pose = scaler * direction_norm
            handover_pose.pose.position.x = ee_pose.translation.x + delta_pose[0]
            handover_pose.pose.position.y = ee_pose.translation.y + delta_pose[1] - gripper_offset
            handover_pose.pose.position.z = ee_pose.translation.z + delta_pose[2] 
            #move robot to handover location

            robot_twist_msg = TwistStamped()
            robot_twist_msg.header.stamp = rospy.Time.now()
            robot_twist_msg.header.frame_id = "base_link"
            # robot_twist_msg.twist.linear.x = result_danger[0] if result_danger[0] < d_max else result_to_cup[0]
            
            linear_velocity = Vector3()
            linear_velocity.x = delta_pose[0] * 4.6
            linear_velocity.y = delta_pose[1] * 4.6
            linear_velocity.z = delta_pose[2] * 4.6
            robot_twist_msg.twist.linear = linear_velocity
            #self.open_gripper()
            
            self.publisher2.publish(robot_twist_msg)

    #     # Publish robot pose
    #     robot_pose_msg = PoseStamped()
    #     robot_pose_msg.header.stamp = rospy.Time.now()
    #     robot_pose_msg.header.frame_id = "base_link"
    #     robot_pose_msg.pose = handover_pose.pose
    #     self.publisher.publish(robot_pose_msg)

    #     # self.moveToPose(handover_pose, "arm_link", max_velocity_scaling_factor=0.1)
    #     # rospy.loginfo("Grasping...")
    #     # rospy.sleep(2.0)
    #     # self.gripper_action(grasp)
    #     # rospy.sleep(2.0)
            
    def move_box(self,position):
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position = position #wrt to the base
        box_name = "cup"
        self.scene.add_box(box_name, box_pose, size=(0.075, 0.075, 0.075))


    def open_gripper(self):
                # open gripper small speed and force (0-255)
        rospy.loginfo("Gripper Opening...")
        rospy.sleep(1.0)
        response = self.gripper_srv(position=5, speed=5, force=5)
        if(response):
            self.isgripper_open = True
        
    def close_gripper(self):
        # close gripper small speed and force (0-255)
        rospy.loginfo("Grasping...")
        rospy.sleep(1.0)
        # close gripper small speed and force
        response = self.gripper_srv(position=150, speed=5, force=5)
        if(response):
            self.isgripper_open = False

if __name__ == '__main__':
    try:
        ObstaclesNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
