#!/usr/bin/env python3
#GP Prediction alone
import rospy
import os
import sys
from geometry_msgs.msg import TransformStamped,PoseStamped
import numpy as np
import george
from george import kernels
from scipy.optimize import minimize
import csv

class GPRNode:
    def __init__(self):
        rospy.init_node('gaussian_process_node')
        # self.pose_sub = self.create_subscription(
        #     TransformStamped,
        #     'hand_poses',
        #     self.pose_callback,
        #     10
        # )
        # self.positions_pub2 = self.create_publisher(PoseStamped,'pred_pos',10)

                # ROS Subscribers and Publishers
        # self.pose_sub = rospy.Subscriber('/vrpn_client_node/cup/pose', TransformStamped, self.pose_callback,queue_size=10)
        self.pose_sub = rospy.Subscriber('/vrpn_mocap/cup/pose', PoseStamped, self.pose_callback,queue_size=10)
        
        self.predpose_pub = rospy.Publisher('/pred_cup_pose', PoseStamped, queue_size=10)
        self.refpose_pub = rospy.Publisher('/ref_cup_pose', PoseStamped, queue_size=10)
        rospy.loginfo("Node started")
        # Initialize variables for position, time, and velocity data
        self.position_data = []  # Stores XYZ position data
        self.time_data = []  # Stores time datas
        self.num_future_time_points = 10  # Number of future time points
        self.horizon = 10 #horizon in seconds. 
        """what's the difference between the 2 above"""
        #self.future_time_points = np.linspace(0.0, self.horizon, num=self.num_future_time_points)
        self.tol = 0.01  # tolerance value for position
        self.veltol = 0.001  # Set your velocity tolerance value
        self.prev_pred = []    
        
        self.header_written = False  # Add this attribute to track whether the header has been written


    def pose_callback(self, msg):
        # Handle TransformStamped data and calculate XYZ velocity
        position = msg.pose.position
        current_time = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9

        self.position_data.append([position.x, position.y, position.z])
        self.time_data.append(current_time)

        # Maintain a constant buffer size (remove old data)
        buffer_size = 10
        if len(self.position_data) > buffer_size:
            self.position_data.pop(0)
            self.time_data.pop(0)

        if len(self.position_data) >= 2:
            unfiltered_position = np.array(self.position_data)
            # self.get_logger().warn(f"prev_pred{self.prev_pred}")
            #rospy.loginfo(f"prev_pred: {self.prev_pred}")

            # Calculate XYZ velocity as the change in position over time
            dx = np.diff(unfiltered_position, axis=0)
            dt = np.diff(self.time_data)

            # Ensure that dt has one less element than the number of rows in dx
            if len(dt) < len(dx):
                dt = np.append(dt, dt[-1])  # Append the last element to match the length (Doesn't this affect the final result) ---------- Need to investigate

            vel = dx / dt[:, None]
            vel_dt = self.num_future_time_points/self.horizon #rate

            # Train the GP model on time and dx/dt and generate preditions
            pred_velx, pred_vely, pred_velz = self.Holrd_predict(self.time_data[:-1], vel)

            #rospy.loginfo(f"in time:{self.time_data[:-1]} in vel{vel} pred_velx size:{len(pred_velx)},vel x data {pred_velx}")

            pred_x= self.pos_at_zero_vel2(unfiltered_position[-1:,0],pred_velx, vel_dt)
            pred_y= self.pos_at_zero_vel2(unfiltered_position[-1:,1],pred_vely, vel_dt)
            pred_z= self.pos_at_zero_vel2(unfiltered_position[-1:,2],pred_velz, vel_dt)

            rospy.loginfo(f"pred_x size:{len(pred_x)},pred_x {pred_x}")

            posx, posy, posz = self.is_within_tolerance(self.prev_pred, pred_x, pred_y, pred_z)
            
                   # Publish calculated position as a TransformStamped message
            pred_position_msg = PoseStamped()
            ref_position_msg = PoseStamped()
            #position_msg.header.stamp = rospy.Time.now()
            pred_position_msg.header.stamp = msg.header.stamp
            ref_position_msg.header.stamp = msg.header.stamp
            pred_position_msg.header.frame_id = 'world'
            ref_position_msg.header.frame_id = 'world'
            #position_msg.child_frame_id = 'pred_hand'

            # Assuming position_at_zero_velocity is a 3D vector
            pred_position_msg.pose.position.x = posx.astype(float)
            pred_position_msg.pose.position.y = posy.astype(float)
            pred_position_msg.pose.position.z = posz.astype(float)

            ref_position_msg.pose.position = msg.pose.position

            self.predpose_pub.publish(pred_position_msg)
            # self.get_logger().warn(f'published msg: {position_msg.pose.position}')
            self.refpose_pub.publish(ref_position_msg)
            #save of pred and the unpred results to csv
            self.prev_pred = [posx,posy,posz]

            print("Current working directory:", os.getcwd())
            self.write_to_csv(ref_position_msg, pred_position_msg)

    def write_to_csv(self, ref_msg, pred_msg):
        with open('/home/starde20/ur_driver/src/results/comparison_res.csv', mode='a') as file:
            writer = csv.writer(file)

            # Write the header only if it hasn't been written before
            if not self.header_written:
                writer.writerow(["Time","Reference X", "Reference Y", "Reference Z", "Predicted X", "Predicted Y", "Predicted Z"])
                self.header_written = True

            # Write the data row
            writer.writerow([ rospy.Time.now(),
                ref_msg.pose.position.x, ref_msg.pose.position.y, ref_msg.pose.position.z,
                pred_msg.pose.position.x, pred_msg.pose.position.y, pred_msg.pose.position.z
            ])

            

    def is_within_tolerance(self, prevpred, posx, posy, posz):
        # Check if the difference between the previously published position and the current position is less than the tolerance
        if len(prevpred) ==0:
            prevpred = [posx,posy,posz]
      

        def calculate_difference(prev, new):
            # Helper function to calculate the Euclidean distance between two 3D points
            #rospy.loginfo(f"prev {prev},new {new}")
            return np.linalg.norm(np.array(prev) - np.array(new))

        # Check if any of posx, posy, posz is greater than 10 """To filter out abnormal values"""
        # if any(coord > 5.0 for coord in [posx, posy, posz]):
        #     # Level 1: Within workspace
        #     #return prevpred
        #     newpred = prevpred
        # else:
        newpred = [posx, posy, posz]

        # Calculate the difference between previous prediction and new prediction
        diff = calculate_difference(prevpred, newpred)

        # Level 2: Consider a new GP if the difference in predictions is greater than tolerance
        if diff > self.tol : # You can adjust this threshold as needed
            # Update published position if the difference is greater than the tolerance
            pubpos = newpred     
        else:
            # Keep the previous position if within tolerance
            pubpos = prevpred
            
        prev_pred = pubpos
        return pubpos


    def Holrd_predict(self, time_data, velocity):
        """
        future_time : A list containing  the prediction horizon time
        time_data : A list conatining training time data
        velocity : 2d array of xyz vel 

        """

        time_data = np.array(time_data).reshape(-1,1)
        velocity = np.array(velocity).reshape(3, -1)


        last_time = time_data[-1]
        future_time = np.linspace(last_time + 0.0, last_time + self.horizon, self.num_future_time_points)
        #rospy.loginfo(f"future time:{len(future_time)},future time data {future_time}")

        t_train = time_data    
        vel_x = velocity[0,:]
        vel_y = velocity[1,:]
        vel_z = velocity[2,:]

                       
        #setup the gp
        kernel = 0.5 * kernels.Matern32Kernel(1.0, ndim=1, axes=0)
        # gp1 = george.GP(kernel, solver=george.HODLRSolver,white_noise=np.log(0.1**2))
        # gp2 = george.GP(kernel, solver=george.HODLRSolver,white_noise=np.log(0.1**2))
        # gp3 = george.GP(kernel, solver=george.HODLRSolver,white_noise=np.log(0.1**2))    

        gp1 = george.GP(kernel, solver=george.HODLRSolver,mean=np.mean(vel_x),white_noise=np.log(0.002**2))
        gp2 = george.GP(kernel, solver=george.HODLRSolver,mean=np.mean(vel_y),white_noise=np.log(0.002**2))
        gp3 = george.GP(kernel, solver=george.HODLRSolver,mean=np.mean(vel_z),white_noise=np.log(0.002**2))   
        
        gp1.compute(t_train)
        gp2.compute(t_train)
        gp3.compute(t_train)
        
        "--------------------------------------------------------------"     
        def nll(p):
            gp1.set_parameter_vector(p)
            ll = gp1.lnlikelihood(vel_x,quiet=True)
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the objective function.
        def grad_nll(p):
            gp1.set_parameter_vector(p)
            return -gp1.grad_lnlikelihood(vel_x, quiet=True)
        
        result1 = minimize(nll, gp1.get_parameter_vector(), 
                        jac=grad_nll, method="L-BFGS-B")
        gp1.set_parameter_vector(result1.x)
        predx , _  = gp1.predict(vel_x,future_time,return_var=True)
        #self.get_logger().warn(f"predx:{predx} varx : {varx}")
        #self.get_logger().warn(f"predx size:{len(predx)},train time data {len(t_train)} pedictions {predx}" )
        
        "--End GPx------------------------------------------------------------"
        def neg_ln_like2(p):
            gp2.set_parameter_vector(p)
            return -gp2.log_likelihood(vel_y)
        
        def grad_neg_ln_like2(p):
            gp2.set_parameter_vector(p)
            return -gp2.grad_log_likelihood(vel_y,quiet=True)
        
        result2 = minimize(neg_ln_like2, gp2.get_parameter_vector(), 
                        jac=grad_neg_ln_like2, method="L-BFGS-B")
        gp2.set_parameter_vector(result2.x)
        predy ,_ = gp2.predict(vel_y,future_time,return_var=True)
        
        "--End GPy------------------------------------------------------------"
        
        def neg_ln_like(p):
            gp3.set_parameter_vector(p)
            return -gp3.log_likelihood(vel_z)
        
        def grad_neg_ln_like(p):
            gp3.set_parameter_vector(p)
            return -gp3.grad_log_likelihood(vel_z,quiet=True)
        
        result3 = minimize(neg_ln_like, gp3.get_parameter_vector(), 
                        jac=grad_neg_ln_like, method="L-BFGS-B")
        gp3.set_parameter_vector(result3.x)
        
        predz ,_ = gp3.predict(vel_z,future_time,return_var=True)
        "--End GPz------------------------------------------------------------"

        return predx, predy, predz
    
    def pos_at_zero_vel(self, old_pos, predictions, dt):
        """
        inputs: predictions, delta_t, old_pos, veltol
        output: new position

        If there exists a value in predictions less than veltol,
        create a sub array containing prediction values to that point.
        Let x_t = old pos
        current_pos = old_pos + prediction value * dt
        old_pos = current pos
        return current_pos
        """
        vel_sub=[]
        if (any(abs(value) < self.veltol for value in predictions)):
            for prediction in predictions:
                if prediction >= self.veltol :
                    vel_sub.append(prediction)
                    rospy.loginfo(f"zero position found predsize: {prediction.shape}from :{prediction}")
            #old_pos= current_pos

            for pred in vel_sub:
                current_pos = old_pos + pred * dt
                old_pos = current_pos
                rospy.loginfo(f"old_pos {old_pos} current pos{current_pos}")
                return current_pos
        else :
            rospy.loginfo(f" Nooo zero position found from :{old_pos}")
            return old_pos
        
    def pos_at_zero_vel2(self, old_pos, predictions, dt):
            # Check if velocities list is empty
        if len(predictions) < 2:
            return old_pos
        
        # Find the index where velocity first crosses zero
        zero_velocity_index = np.argmax(np.abs(predictions) < self.veltol)
        # If zero velocity is not found, return current position
        if zero_velocity_index >= len(predictions):
            return old_pos

        # Integrate velocities from the current position up to the zero velocity point
        positions = np.cumsum(predictions[:zero_velocity_index]) * dt

        print(f"positions at zero velocity index {positions}")
        # Check if positions array is empty
        if not positions:
            return old_pos
        
        # Calculate the position at zero velocity
        position_at_zero_velocity = old_pos + positions[-1]

        return position_at_zero_velocity

if __name__ == '__main__':
    try:
        gp_node = GPRNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
