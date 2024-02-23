#added KF
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from filterpy.kalman import KalmanFilter

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
        self.velocity_data = []  # Stores XYZ velocity data
        self.gp_model = None
        self.kalman_filter = self.initialize_kalman_filter()
        self.num_future_time_points = 10  # Number of future time points
        self.future_time_points = np.linspace(0.2, 2.0, num=self.num_future_time_points)
        self.tol = 0.05  # Set your tolerance value
        

    def initialize_kalman_filter(self):
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

            # Train the GP model on time and dx/dt
            self.train_gpr_model(dt, dx/dt)

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
            filtered_position.append(self.kalman_filter.x[:3])

        return np.array(filtered_position)

    def update_future_time_points(self):
        # Update the future time points based on the most recent data
        if len(self.time_data) > 1:
            # Calculate an estimate of the time it takes to reach zero velocity from the current velocity
            last_time = self.time_data[-1]
            self.future_time_points = np.linspace(last_time + 1.0, last_time + 10.0, num=self.num_future_time_points)

    def predict_and_publish_position(self, future_time):
        if self.gp_model is not None:
            # Predict the future position based on the trained GPR model
            #predicted_position = self.gp_model.predict(np.array([[future_time]]))[0]

            # Calculate the positions at zero velocity
            zero_velocity_positions = self.calculate_positions_at_zero_velocity(future_time)

            # Check if the difference between the previously published position and the predicted position is less than the tolerance
            if self.is_within_tolerance(zero_velocity_positions):
                # Publish predicted position as a TransformStamped message
                position_msg = TransformStamped()
                position_msg.header.stamp = self.get_clock().now().to_msg()
                position_msg.header.frame_id = 'camera_link'
                position_msg.child_frame_id = 'pred_hand'
                position_msg.transform.translation.x, position_msg.transform.translation.y, position_msg.transform.translation.z = zero_velocity_positions
                self.positions_pub.publish(position_msg)

    def is_within_tolerance(self, current_position):
        # Check if the difference between the previously published position and the current position is less than the tolerance
        if len(self.position_data) > 1:
            previous_position = self.position_data[-2]
            position_difference = np.linalg.norm(np.array(current_position) - np.array(previous_position))
            return position_difference < self.tol
        else:
            return True  # If there is no previous position, consider it within tolerance

    def calculate_positions_at_zero_velocity(self, future_time):
        # Interpolate positions at the predicted time when velocity will be zero
        if len(self.time_data) >= 2:
            xp = np.array(self.time_data)
            fp = np.array(self.position_data).T
            zero_velocity_positions = []

            for i in range(3):  # Loop over x, y, z dimensions
                # Interpolate position based on the available position data for each dimension
                interpolated_position = np.interp(future_time, xp, fp[i])
                zero_velocity_positions.append(interpolated_position)

            return zero_velocity_positions
        else:
            return [0.0, 0.0, 0.0]

    def train_gpr_model(self, dt, dx_dt):
        X = np.cumsum(dt)  # Cumulative sum of time intervals to get timestamps
        y = np.linalg.norm(dx_dt, axis=1)
        kernel = 0.5 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.15)
        self.gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        self.gp_model.fit(X.reshape(-1, 1), y)

def main():
    rclpy.init()
    gpr_node = GPRNode()
    rclpy.spin(gpr_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
