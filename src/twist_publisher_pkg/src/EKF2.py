import numpy as np
import matplotlib.pyplot as plt

# Nonlinear state transition function
def state_transition(x, dt):
    # Vehicle model with constant velocity
    x[0] += x[2] * dt
    x[1] += x[3] * dt
    return x

# Nonlinear measurement model
def measurement_model(x):
    # Measurement is the position (x, y)
    return np.array([x[0], x[1]])

# Extended Kalman Filter implementation
def extended_kalman_filter(x, P, z, dt):
    # Prediction Step
    x_pred = state_transition(x, dt)
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])  # Jacobian of state transition
    P_pred = np.dot(np.dot(F, P), F.T)

    # Update Step
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])  # Jacobian of measurement model
    K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(np.dot(np.dot(H, P_pred), H.T) + R))
    x_updated = x_pred + np.dot(K, z - measurement_model(x_pred))
    P_updated = P_pred - np.dot(np.dot(K, H), P_pred)

    return x_updated, P_updated

# Simulation parameters
num_steps = 100
dt = 0.1

# Initial state and covariance
x_true = np.array([0.0, 0.0, 1.0, 1.0])  # [x, y, vx, vy]
x_est = np.array([0.1, -0.2, 1.2, 1.1])  # Initial estimate
P_est = np.eye(4) * 0.1

# Measurement noise covariance
R = np.eye(2) * 0.1

# Arrays to store results for plotting
true_positions = np.zeros((num_steps, 2))
estimated_positions = np.zeros((num_steps, 2))

# Run the Extended Kalman Filter
for k in range(num_steps):
    # Simulate true system
    x_true = state_transition(x_true, dt)

    # Simulate measurement with noise
    z = measurement_model(x_true) + np.random.multivariate_normal([0, 0], R)

    # Run EKF
    x_est, P_est = extended_kalman_filter(x_est, P_est, z, dt)

    # Store results for plotting
    true_positions[k, :] = measurement_model(x_true)
    estimated_positions[k, :] = measurement_model(x_est)

# Plot the results
plt.plot(true_positions[:, 0], true_positions[:, 1], label='True Positions')
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], label='Estimated Positions', linestyle='dashed')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
