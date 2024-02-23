import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import numpy as np
from filterpy.kalman import KalmanFilter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import GPy
import george
from george import kernels

class GPRNode(Node):
    def __init__(self):
        super().__init__('gpr_node')
        self.pose_sub = self.create_subscription(
            TransformStamped,
            'hand_poses',
            self.pose_callback,
            10
        )
        self.positions_pub = self.create_publisher(TransformStamped, 'predicted_positions', 10)

        # Initialize variables for position, time, and velocity data
        self.position_data = []  # Stores XYZ position data
        self.time_data = []  # Stores time data
        self.gp_models = [None, None, None]  # Stores separate GP models for XYZ dimensions
        #self.kalman_filter = self.initialize_kalman_filter()
        self.kalman_filter = self.initialize_kalman_filter(Q=np.diag([0.01, 0.01, 0.01, 0.001, 0.001, 0.001]))

        self.num_future_time_points = 10  # Number of future time points
        self.future_time_points = np.linspace(0.2, 2.0, num=self.num_future_time_points)
        self.tol = 0.2  # Set your tolerance value
        self.veltol = 0.02  # Set your velocity tolerance value

    def initialize_kalman_filter(self,Q):
        # Initialize a simple Kalman filter for 3D position data
        kalman_filter = KalmanFilter(dim_x=6, dim_z=3)
        kalman_filter.F = np.array([[1, 0, 0, 1, 0, 0],
                                    [0, 1, 0, 0, 1, 0],
                                    [0, 0, 1, 0, 0, 1],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])

        kalman_filter.H = np.array([[1, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0]])

        kalman_filter.P *= 1e3
        kalman_filter.R = np.eye(3) * 0.1
        kalman_filter.Q = Q  # Add white noise covariance matrix Q
        return kalman_filter

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
            # Apply Kalman filter to position data
            filtered_position = self.apply_kalman_filter()

            # Calculate XYZ velocity as the change in position over time
            dx = np.diff(filtered_position, axis=0)
            dt = np.diff(self.time_data)

            # Ensure that dt has one less element than the number of rows in dx
            if len(dt) < len(dx):
                dt = np.append(dt, dt[-1])  # Append the last element to match the length

            vel = dx / dt[:, None]

            # Train the GP model on time and dx/dt
            self.train_gpr_model(self.time_data[:-1], vel)

            # Update future time points based on the most recent data and velocity
            self.update_future_time_points()

            # Predict and publish positions at the updated future time points
            for future_time in self.future_time_points:
                self.predict_and_publish_position(future_time)

    def apply_kalman_filter(self):
        # Apply Kalman filter to position data
        filtered_position = []
        for pos in self.position_data:
            self.kalman_filter.predict()
            self.kalman_filter.update(np.array(pos))
            filtered_position.append(self.kalman_filter.x[:3].flatten())

        return np.array(filtered_position)

    def update_future_time_points(self):
        # Update the future time points based on the most recent data
        if len(self.time_data) > 1:
            # Calculate an estimate of the time it takes to reach zero velocity from the current velocity
            last_time = self.time_data[-1]
            self.future_time_points = np.linspace(last_time + 1.0, last_time + 2.0, num=self.num_future_time_points)

    def predict_and_publish_position(self, future_time):
        for i in range(3):
            if self.gp_models[i] is not None:
                # Predict the future velocity based on the trained GPR model
                predicted_velocity_i = self.gp_models[i].predict(np.array([[future_time]]))[0]
                #self.get_logger().warn(f'predicted vel { predicted_velocity_i }')

                # Check if the predicted velocity is less than the velocity tolerance
                if abs(predicted_velocity_i) < self.veltol:
                    # Calculate the time when velocity is close to zero
                    time_at_zero_velocity = self.predict_time_at_zero_velocity(future_time, i)

                    # Calculate the position when velocity is close to zero using numerical integration
                    position_at_zero_velocity = self.integrate_position_at_zero_velocity(time_at_zero_velocity)
                    self.get_logger().warn(f'position at zero vel: {position_at_zero_velocity}')

                    # Check if the difference between the previously published position and the calculated position is less than the tolerance
                    if self.is_within_tolerance(position_at_zero_velocity):
                        # Publish calculated position as a TransformStamped message
                        position_msg = TransformStamped()
                        position_msg.header.stamp = self.get_clock().now().to_msg()
                        position_msg.header.frame_id = 'camera_link'
                        position_msg.child_frame_id = 'pred_hand'

                        # Assuming position_at_zero_velocity is a 3D vector
                        position_msg.transform.translation.x = position_at_zero_velocity[0]
                        position_msg.transform.translation.y = position_at_zero_velocity[1]
                        position_msg.transform.translation.z = position_at_zero_velocity[2]

                        self.positions_pub.publish(position_msg)
                        self.get_logger().warn(f'published msg: {position_msg.transform.translation}')

    

    def is_within_tolerance(self, current_position):
        # Check if the difference between the previously published position and the current position is less than the tolerance
        if len(self.position_data) > 1:
            previous_position = self.position_data[-2]
            position_difference = np.linalg.norm(np.array(current_position) - np.array(previous_position))
            return position_difference < self.tol
        else:
            return True  # If there is no previous position, consider it within tolerance

    def predict_time_at_zero_velocity(self, future_time, dimension):
        # Use the trained GP model to predict the time when velocity is close to zero
        predicted_velocity_i = self.gp_models[dimension].predict(np.array([[future_time]]))[0]

        # Find the time at which velocity is close to zero using the veltol
        time_at_zero_velocity = future_time
        while abs(predicted_velocity_i) > self.veltol:
            time_at_zero_velocity += 0.03  # Adjust the step size as needed
            predicted_velocity_i = self.gp_models[dimension].predict(np.array([[time_at_zero_velocity]]))[0]

        return time_at_zero_velocity
    
    def integrate_position_at_zero_velocity(self, time_at_zero_velocity):
        # Numerical integration (trapezoidal rule) to calculate position when velocity is close to zero
        positions = []
        for dimension in range(3):
            indices = np.where(np.array(self.time_data) <= time_at_zero_velocity)[0]
            integration_result = np.trapz(np.array(self.position_data)[indices, dimension], x=np.array(self.time_data)[indices])
            positions.append(integration_result)
        return np.array(positions)



    def train_gpr_model(self, time_data, velocity):
        # Ensure that there is enough data for training
        if len(time_data) < 2:
            return

        # Use the actual time values for training
        X = np.array(time_data[:-1]).reshape(-1, 1)
        y = velocity

        # Set up the kernel
        #kernel = 0.5 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-8)
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-3))


        # Train separate GP models for each dimension
        self.gp_models = [None, None, None]  # Reset the list of models
        for i in range(3):
            # Create a new GP model
            gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
            gp_model.fit(X, y[:-1, i])  # Modify this line

            # Save the model in the list
            self.gp_models[i] = gp_model

        # Save the input data for future predictions
        self.X_train = X
        self.y_train = y



def main():
    rclpy.init()
    gpr_node = GPRNode()
    rclpy.spin(gpr_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
