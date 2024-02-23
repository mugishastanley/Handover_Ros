#GP Prediction alone
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped,PoseStamped
import numpy as np
import george
from george import kernels
from scipy.optimize import minimize

class GPRNode(Node):
    def __init__(self):
        super().__init__('gpr_node')
        self.pose_sub = self.create_subscription(
            TransformStamped,
            'hand_poses',
            self.pose_callback,
            10
        )
        self.positions_pub2 = self.create_publisher(PoseStamped,'pred_pos',10)
        # Initialize variables for position, time, and velocity data
        self.position_data = []  # Stores XYZ position data
        self.time_data = []  # Stores time datas
        self.num_future_time_points = 20  # Number of future time oints
        self.horizon = 2.0 #horizon in seconds
        #self.future_time_points = np.linspace(0.0, self.horizon, num=self.num_future_time_points)
        self.tol = 0.02  # tolerance value for position
        self.veltol = 0.005  # Set your velocity tolerance value
        self. prev_pred = []
   


    def pose_callback(self, msg):
        # Handle TransformStamped data and calculate XYZ velocity
        position = msg.transform.translation
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.position_data.append([position.x, position.y, position.z])
        self.time_data.append(current_time)

        # Maintain a constant buffer size (remove old data)
        buffer_size = 20
        if len(self.position_data) > buffer_size:
            self.position_data.pop(0)
            self.time_data.pop(0)

        if len(self.position_data) >= 2:
            unfiltered_position = np.array(self.position_data)
            self.get_logger().warn(f"prev_pred{self.prev_pred}")

            # Calculate XYZ velocity as the change in position over time
            dx = np.diff(unfiltered_position, axis=0)
            dt = np.diff(self.time_data)

            #self.get_logger().warn(f"dx shape :{dx.shape}")

            # Ensure that dt has one less element than the number of rows in dx
            if len(dt) < len(dx):
                dt = np.append(dt, dt[-1])  # Append the last element to match the length (Doesn't this affect the final result) ---------- Need to investigate

            vel = dx / dt[:, None]
            vel_dt = self.num_future_time_points/self.horizon #rate

            # Train the GP model on time and dx/dt
            pred_velx, pred_vely, pred_velz = self.Holrd_predict(self.time_data[:-1], vel)
            #self.get_logger().warn(f"input vel_x: {vel[:,0]} len dt{len(self.time_data[:-1])}  len pred_velx: {len(pred_velx)}")

            pred_x= self.pos_at_zero_vel(unfiltered_position[-1:,0],pred_velx, vel_dt)
            pred_y= self.pos_at_zero_vel(unfiltered_position[-1:,1],pred_vely, vel_dt)
            pred_z= self.pos_at_zero_vel(unfiltered_position[-1:,2],pred_velz, vel_dt)

            #self.get_logger().warn(f"pred poses x={pred_x}:y={pred_y}:z={pred_z}")
            #self.get_logger().warn(f"pred poses x={pred_x}: unfiltered last pose ={unfiltered_position[-1:,0]}:pred vel ={pred_velx}")  

            posx, posy, posz = self.is_within_tolerance(self.prev_pred, pred_x,pred_y,pred_z)
            
                   # Publish calculated position as a TransformStamped message
            position_msg = PoseStamped()
            position_msg.header.stamp = self.get_clock().now().to_msg()
            #position_msg.header.stamp = msg.header.stamp
            position_msg.header.frame_id = 'camera_link'
            #position_msg.child_frame_id = 'pred_hand'

            # Assuming position_at_zero_velocity is a 3D vector
            position_msg.pose.position.x = float(posx)
            position_msg.pose.position.y = float(posy)
            position_msg.pose.position.z = float(posz)

            self.positions_pub2.publish(position_msg)
            self.get_logger().warn(f'published msg: {position_msg.pose.position}')
            self.prev_pred = [posx,posy,posz]
            



    def is_within_tolerance(self, prevpred, posx, posy, posz):
        # Check if the difference between the previously published position and the current position is less than the tolerance
        if len(prevpred) ==0:
            prevpred = [posx,posy,posz]        

        def calculate_difference(prev, new):
            # Helper function to calculate the Euclidean distance between two 3D points
            return np.linalg.norm(np.array(prev) - np.array(new))

        # Check if any of posx, posy, posz is greater than 1.0
        if any(coord > 1.0 for coord in [posx, posy, posz]):
            # Level 1: Within workspace
            return prevpred
        else:
            newpred = [posx, posy, posz]

            # Calculate the difference between previous prediction and new prediction
            diff = calculate_difference(prevpred, newpred)

            # Level 2: Consider a new GP if the difference in predictions is greater than tolerance
            if diff > 0.1 : # You can adjust this threshold as needed
                # Update published position if the difference is greater than the tolerance
                pubpos = newpred
                prevpred = newpred
            else:
                # Keep the previous position if within tolerance
                pubpos = prevpred
                prevpred = newpred
            return pubpos


    def Holrd_predict(self, time_data, velocity):
        """
        future_time : A list containing  the prediction horizon time
        time_data : A list conatining training time data
        velocity : 2d array of xyz vel 

        """

        time_data = np.array(time_data).reshape(-1,1)
        velocity = np.array(velocity).reshape(3, -1)
        #self.get_logger().info(f" time_data shape{time_data.shape} : velocity shape {velocity.shape}")
        last_time = time_data[-1]
        future_time = np.linspace(last_time + 0.0, last_time + self.horizon, self.num_future_time_points)

        t_train = time_data    
        vel_x = velocity[0,:]
        vel_y = velocity[1,:]
        vel_z = velocity[2,:]
        #self.get_logger().warn(f"velx size:{len(vel_x)},vel x data {vel_x}" )
                       
        #setup the gp
        kernel = 0.5 * kernels.Matern52Kernel(1.0, ndim=1, axes=0)
        # gp1 = george.GP(kernel, solver=george.HODLRSolver,white_noise=np.log(0.1**2))
        # gp2 = george.GP(kernel, solver=george.HODLRSolver,white_noise=np.log(0.1**2))
        # gp3 = george.GP(kernel, solver=george.HODLRSolver,white_noise=np.log(0.1**2))    

        gp1 = george.GP(kernel, solver=george.HODLRSolver,mean=np.mean(vel_x),white_noise=np.log(0.2**2))
        gp2 = george.GP(kernel, solver=george.HODLRSolver,mean=np.mean(vel_y),white_noise=np.log(0.2**2))
        gp3 = george.GP(kernel, solver=george.HODLRSolver,mean=np.mean(vel_z),white_noise=np.log(0.2**2))   
        
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
    
    def pos_at_zero_vel(self, current_pos, predictions, dt):
        """
        inputs: predictions, delta_t, current_pos, veltol
        output: new position

        If there exists a value in predictions less than veltol,
        create a sub array containing prediction values to that point.
        Let x_t = current pos
        new_pos = current_pos + prediction value * dt
        current_pos = new_pos
        return current_pos
        """
        vel_sub=[]
        if (any(value <= self.veltol and value > 0.0 for value in predictions)):
            for prediction in predictions:
                if prediction >= self.veltol :
                    vel_sub.append(prediction)
            old_pos= current_pos

            for pred in vel_sub:
                #print("old_pos",old_pos,"pred:",pred,"current pos",current_pos)
                current_pos = old_pos + pred * dt
                old_pos = current_pos
        return current_pos

def main():
    rclpy.init()
    gpr_node = GPRNode()
    rclpy.spin(gpr_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
