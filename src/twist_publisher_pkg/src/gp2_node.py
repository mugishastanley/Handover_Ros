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
import bagpy
from bagpy import bagreader

class GPRNode:
    def __init__(self):
        rospy.init_node('gaussian_process_node')
        # self.pose_sub = rospy.Subscriber('/vrpn_client_node/cup/pose', TransformStamped, self.pose_callback,queue_size=10)
        self.pose_sub = rospy.Subscriber('/vrpn_mocap/cup/pose', PoseStamped, self.pose_callback,queue_size=10)
        
        self.predpose_pub = rospy.Publisher('/pred_cup_pose', PoseStamped, queue_size=10)
        self.refpose_pub = rospy.Publisher('/ref_cup_pose', PoseStamped, queue_size=10)
        rospy.loginfo("Node started")
        # Initialize variables for position, time, and velocity data
        self.position_data = []  # Stores XYZ position data
        self.time_data = []  # Stores time datas
        self.num_future_time_points = 20  # Number of future time points
        self.horizon = 3.0 #horizon in seconds. 
        """what's the difference between the 2 above"""
        #self.future_time_points = np.linspace(0.0, self.horizon, num=self.num_future_time_points)
        self.tol = 0.01  # tolerance value for position
        self.veltol = 0.01  # Set your velocity tolerance value
        self.prev_pred = []            
        self.header_written = False  # Add this attribute to track whether the header has been written
        self.seconds1 = rospy.get_time()


    def pose_callback(self, msg):
        # Handle TransformStamped data and calculate XYZ velocity
        position = msg.pose.position
        current_time = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9

        self.position_data.append([position.x, position.y, position.z])
        self.time_data.append(current_time)

        # Maintain a constant buffer size (remove old data)
        buffer_size = 40
        if len(self.position_data) > buffer_size:
            self.position_data.pop(0)
            self.time_data.pop(0)

        if len(self.position_data) >= 2:
            unfiltered_position = np.array(self.position_data)
            # self.get_logger().warn(f"prev_pred{self.prev_pred}")
            #rospy.loginfo(f"prev_pred: {self.prev_pred}")

            # # Calculate XYZ velocity as the change in position over time
            # dx = np.diff(unfiltered_position, axis=0)
            # dt = np.diff(self.time_data)

            # # Ensure that dt has one less element than the number of rows in dx
            # if len(dt) < len(dx):
            #     dt = np.append(dt, dt[-1])  # Append the last element to match the length (Doesn't this affect the final result) ---------- Need to investigate

            # vel = dx / dt[:, None]
            # vel_dt = self.num_future_time_points/self.horizon #rate

            # Train the GP model on time and dx/dt and generate preditions


            #rospy.loginfo(f"in time:{len(self.time_data)} unfiltered pos in{unfiltered_position} pred_posx size:{unfiltered_position.shape}")
            vx,vy, vz,pred_posx, pred_posy, pred_posz = self.Holrd_predict(self.time_data, unfiltered_position)

            #rospy.loginfo(f"in time:{self.time_data} in posx size:{unfiltered_position[:,0]} pred pos_x{pred_posx} ")

            pred_x= pred_posx
            pred_y= pred_posy
            pred_z= pred_posz
            vx=vx[0]
            vy=vy[0]
            vz=vz[0]
            

            # rospy.loginfo(f"pred_x size:{len(pred_x)},pred_x {pred_x}")

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
            pred_position_msg.pose.position.x = pred_x.astype(float)
            pred_position_msg.pose.position.y = pred_y.astype(float)
            pred_position_msg.pose.position.z = pred_z.astype(float)

            ref_position_msg.pose.position = msg.pose.position

            self.predpose_pub.publish(pred_position_msg)
            # self.get_logger().warn(f'published msg: {position_msg.pose.position}')
            self.refpose_pub.publish(ref_position_msg)
            #save of pred and the unpred results to csv
            #self.prev_pred = [posx,posy,posz]

            # print("Current working directory:", os.getcwd())
            point = self.points(pred_x,pred_y,pred_z)

            self.write_to_csv(msg.header.stamp, ref_position_msg, pred_position_msg, vx, vy,vz, point)


    def write_to_csv(self, time_stamp,ref_msg, pred_msg, vx, vy,vz, point):
        with open('/home/starde20/ur_driver/src/results/comparison_res.csv', mode='a') as file:
            writer = csv.writer(file)

            # Write the header only if it hasn't been written before
            if not self.header_written:
                writer.writerow(["Time","Ref X", "Ref Y", "Ref Z", "Pred X", "Pred Y", "Pred Z", "vel_x", "vel_y", "vel_z" ,"point"])
                self.header_written = True

            #t = rospy.Time.from_sec(time_stamp)
            # Write the data row
            seconds = rospy.get_time()
            t= seconds-self.seconds1
            writer.writerow([t,
                ref_msg.pose.position.x, ref_msg.pose.position.y, ref_msg.pose.position.z,
                pred_msg.pose.position.x, pred_msg.pose.position.y, pred_msg.pose.position.z, vx, vy,vz, point
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


    def Holrd_predict(self, time_data, positions):
        """
        future_time : A list containing  the prediction horizon time
        time_data : A list conatining training time data
        positions : 3d array of xyz positions

        """

        time_data = np.array(time_data).reshape(-1, 1)

        #rospy.loginfo(f"gp positions in :{positions[:,0]}" )

        #positions = np.array(positions).reshape(3, -1)


        last_time = time_data[-1]
        future_time = np.linspace(last_time + 0.0, last_time + self.horizon, self.num_future_time_points)
        #rospy.loginfo(f"future time:{len(future_time)},future time data {future_time}")

        t_train = time_data    
        pos_x = positions[:,0]
        pos_y = positions[:,1]
        pos_z = positions[:,2]
        #rospy.loginfo(f"gp posx in :{len(pos_x)},train time data {t_train} posx_in values {pos_x}" )

                       
        #setup the gp
        kernel1 = 0.5 * kernels.Matern52Kernel(1.0, ndim=1, axes=0)
        kernel2 = 0.5 * kernels.Matern52Kernel(1.0, ndim=1, axes=0)
        kernel3 = 0.5 * kernels.Matern52Kernel(1.0, ndim=1, axes=0)

        gp1 = george.GP(kernel1, solver=george.HODLRSolver,mean=np.mean(pos_x),white_noise=np.log(0.02**2))
        gp2 = george.GP(kernel2, solver=george.HODLRSolver,mean=np.mean(pos_y),white_noise=np.log(0.02**2))
        gp3 = george.GP(kernel3, solver=george.HODLRSolver,mean=np.mean(pos_z),white_noise=np.log(0.02**2))   
        
        gp1.compute(t_train)
        gp2.compute(t_train)
        gp3.compute(t_train)
        
        "--------------------------------------------------------------"     
        def nll(p):
            gp1.set_parameter_vector(p)
            ll = gp1.lnlikelihood(pos_x,quiet=True)
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the objective function.
        def grad_nll(p):
            gp1.set_parameter_vector(p)
            return -gp1.grad_lnlikelihood(pos_x, quiet=True)
        
        result1 = minimize(nll, gp1.get_parameter_vector(), 
                        jac=grad_nll, method="L-BFGS-B")
        gp1.set_parameter_vector(result1.x)
        predx , _  = gp1.predict(pos_x,future_time,return_var=True)
        #rospy.loginfo(f"gp predx size:{len(predx)}, future time data {future_time} pedictions {predx}" )
        
        "--End GPx------------------------------------------------------------"
        def neg_ln_like2(p):
            gp2.set_parameter_vector(p)
            return -gp2.log_likelihood(pos_y)
        
        def grad_neg_ln_like2(p):
            gp2.set_parameter_vector(p)
            return -gp2.grad_log_likelihood(pos_y,quiet=True)
        
        result2 = minimize(neg_ln_like2, gp2.get_parameter_vector(), 
                        jac=grad_neg_ln_like2, method="L-BFGS-B")
        gp2.set_parameter_vector(result2.x)
        predy ,_ = gp2.predict(pos_y,future_time,return_var=True)
        
        "--End GPy------------------------------------------------------------"
        
        def neg_ln_like(p):
            gp3.set_parameter_vector(p)
            return -gp3.log_likelihood(pos_z)
        
        def grad_neg_ln_like(p):
            gp3.set_parameter_vector(p)
            return -gp3.grad_log_likelihood(pos_z,quiet=True)
        
        result3 = minimize(neg_ln_like, gp3.get_parameter_vector(), 
                        jac=grad_neg_ln_like, method="L-BFGS-B")
        gp3.set_parameter_vector(result3.x)
        
        predz ,_ = gp3.predict(pos_z,future_time,return_var=True)
        "--End GPz------------------------------------------------------------"

        """
        function zero vel point
        input: timesteps, prediction_list
        output: prediction, time
        1. calculate predicted vel by dividing prediction_list by hor_timesteps
        2. check if there is any velocity v in predicted vel less than a threshold vtol
        3. return hor_timestep and prediction from prediction_list at v
        """
        def zero_vel_point(timesteps, prediction_list, vtol):
            predicted_vel = []
            for i in range(1, len(timesteps)):
                delta_prediction = prediction_list[i] - prediction_list[i - 1]
                delta_time = timesteps[i] - timesteps[i - 1]
                if delta_time == 0:  # To avoid division by zero
                    predicted_vel.append(float('inf'))
                else:
                    predicted_vel.append(abs(delta_prediction / delta_time))

            #print(f"predicted vel{predicted_vel}")
            
            for i, vel in enumerate(predicted_vel):
                if vel < vtol:
                    return vel[0], timesteps[i+1], prediction_list[i+1]
            
            #return None, None  # If no velocity is below the threshold
            return vel[0], timesteps[-1], prediction_list[-1]  # If no velocity is below the threshold
            #return timesteps[0], prediction_list[0]  # If no velocity is below the threshold


        vel_x, _ , hv_x  = zero_vel_point(future_time, predx, self.veltol)
        vel_y, _ , hv_y   = zero_vel_point(future_time, predy, self.veltol)
        vel_z, _ , hv_z   = zero_vel_point(future_time, predz, self.veltol)

       # print(f"predicted pos_x{hv_x}")

        return vel_x,vel_y, vel_z, hv_x, hv_y, hv_z
    
    def points(self, x, y, z):
        """
        Returns a predicted point.
        Input: x, y, z
        Output: point 
        Algo:
            Define a pointset containing 3d points A, B, C, D, E
            Define a tol
            for all points in pointset
                if dist between a point and x, y, z is less than tol
                    return the point
            return None
        """
        # Define your pointset containing 3d points A, B, C, D, E
        pointset = [(2.260,	0.702,	0.838), 
                    (2.671,	0.704,	0.838), 
                    (2.245,	0.262,  0.838), 
                    (2.641,	0.311,	0.844), 
                    (3.04,	0.810,  0.844)]
        tol = 0.2  # Define your tolerance
        
        closest_point_index = None
        min_dist = float('inf')
        
        # Calculate distance and find the closest point
        for i, point in enumerate(pointset):
            dist = (abs(point[0] - x) + abs(point[1] - y)+ abs(point[2] - z))
            if dist < min_dist:
                min_dist = dist
                closest_point_index = i
        
        # Check if the closest point is within tolerance and return its index
        if min_dist < tol:
            return closest_point_index
        else:
            return -1.0

 
    

if __name__ == '__main__':
    try:
        gp_node = GPRNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
