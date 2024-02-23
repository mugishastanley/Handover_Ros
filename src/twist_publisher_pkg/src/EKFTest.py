import numpy as np
import matplotlib.pyplot as plt
class EKF:
    def __init__(self, x0, P0, F, Q, H, R):
        self.x = x0  # State vector
        self.P = P0  # State covariance matrix
        self.F = F  # Transition matrix
        self.Q = Q  # Process noise covariance matrix
        self.H = H  # Measurement matrix
        self.R = R  # Measurement noise covariance matrix
 

    def predict(self, dt):
        """Predicts the state and covariance for the next time step.

        Args:
            dt (float): Time step between measurements.

        Returns:
            tuple: (x, P) - Updated state vector and covariance matrix.
        """
        # Predicted state
        x_pred = self.F @ self.x

        # Predicted covariance
        P_pred = self.F @ self.P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(self, z):
        """Updates the state and covariance based on a new measurement.

        Args:
            z (np.ndarray): New measurement vector.

        Returns:
            tuple: (x, P) - Updated state vector and covariance matrix.
        """
        # Innovation
        x_diff = z - self.H @ self.x

        # Linearize the state transition and measurement functions
        J_x = self.linearize_state_transition(dt)
        J_z = self.linearize_measurement(z)  # Assuming Gaussian distribution for measurement noise

        # Compute the Kalman gain
        S = J_z @ self.P @ J_z.T + self.R
        K = self.P @ J_z.T @ np.linalg.inv(S)

        # Update the state
        self.x = self.x + K @ x_diff

        # Update the covariance
        I = np.eye(self.x.shape[0])
        self.P = (I - K @ J_z) @ self.P
        return self.x, self.P
    
    def linearize_state_transition(self, dt):
        """Linearizes the state transition function at the current state.

        Args:
            dt (float): Time step between measurements.

        Returns:
            np.ndarray: Jacobian of the state transition function.
        """
        J_x = np.zeros((2, 2))
        J_x[0, 0] = 1
        J_x[0, 1] = dt
        J_x[1, 0] = 0
        J_x[1, 1] = 1
        return J_x

    def linearize_measurement(self, z):
        """Linearizes the measurement function at the current state.

        Args:
            z (np.ndarray): Current measurement vector.

        Returns:
            np.ndarray: Jacobian of the measurement function.
        """
        J_z = np.array([[1, 0]])
        return J_z

# Example usage
# Define parameters
dt = 0.1
x0 = np.array([0.0, 0.0])  # Initial state
P0 = 10 * np.eye(2)  # Initial covariance matrix
F = np.array([[1, dt], [0, 1]])  # Transition matrix
Q = 0.1 * np.eye(2)  # Process noise covariance matrix
H = np.array([[1, 0]])  # Measurement matrix
R = 0.01  # Measurement noise covariance matrix

# Initialize EKF
ekf = EKF(x0, P0, F, Q, H, R)

# Simulate data
dt = 0.1
t = np.arange(0, 10, dt)
z_true = np.zeros_like(t)
z_true += np.cos(2 * np.pi * t)

# Perform EKF filtering
z_filtered = []
for z in z_true:
    x_pred, P_pred = ekf.predict(dt)
    x_update, P_update = ekf.update(z)
    ekf.x = x_update
    ekf.P = P_update
    z_filtered.append(ekf.x[0])

# Plot results
plt.plot(t, z_true, label='True state')
plt.plot(t, z_filtered, label='EKF filtered state')
plt.legend()
plt.show()
