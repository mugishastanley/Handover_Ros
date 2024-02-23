import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class GPEKF:
    def __init__(self, x0, P0, F, Q, H, R, kernel):
        self.x = x0  # State vector
        self.P = P0  # State covariance matrix
        self.F = F  # Transition matrix
        self.Q = Q  # Process noise covariance matrix
        self.H = H  # Measurement matrix
        self.R = R  # Measurement noise covariance matrix
        self.kernel = kernel  # Gaussian process kernel

    def predict(self, dt):
        """Predicts the state and covariance for the next time step."""
        # Predicted state
        x_pred = self.F @ self.x

        # Predicted covariance
        p_pred, _ = self.gp_predict(x_pred)
        P_pred = p_pred + self.Q
        return x_pred, P_pred

    def update(self, z):
        """Updates the state and covariance based on a new measurement."""
        # Innovation
        z_diff = z - self.H @ self.x

        # GP Regression
        p_mean, p_var = self.gp_predict(z_diff)
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)

        # Updated state
        self.x += K @ z_diff

        # Updated covariance
        self.P = (np.eye(self.x.shape[0]) - K @ self.H) @ self.P
        return self.x, self.P

    def gp_predict(self, x):
        """Predicts the mean and variance of the state at a new point."""
        # GP prediction
        gp = GaussianProcessRegressor(kernel=self.kernel)
        # Dummy training data for illustration
        X_train = np.array([[0]])
        y_train = np.array([0])
        gp.fit(X_train, y_train)
        return gp.predict(x.reshape(-1, 1), return_std=True)


dt = 0.1
# Simulate data
t = np.arange(0, 10, dt)
z_true = np.zeros_like(t)
z_true += np.cos(2 * np.pi * t)

# Define parameters
x0 = np.array([0.0, 0.0])  # Initial state
P0 = 10 * np.eye(2)  # Initial covariance matrix
F = np.array([[1, dt], [0, 1]])  # Transition matrix
Q = 0.1 * np.eye(2)  # Process noise covariance matrix
H = np.array([[1, 0]])  # Measurement matrix
R = 0.01  # Measurement noise covariance matrix
kernel = RBF(length_scale=1)  # Gaussian process kernel

# Initialize GP-EKF
gp_ekf = GPEKF(x0, P0, F, Q, H, R, kernel)

# Perform GP-EKF filtering
z_filtered = []
for z in z_true:
    x_pred, P_pred = gp_ekf.predict(dt)
    x_update, P_update = gp_ekf.update(z)
    gp_ekf.x = x_update
    gp_ekf.P = P_update
    z_filtered.append(gp_ekf.x[0])

# Plot results
plt.plot(t, z_true, label='True state')
plt.plot(t, z_filtered, label='GP-EKF filtered state')
plt.legend()
plt.show()
