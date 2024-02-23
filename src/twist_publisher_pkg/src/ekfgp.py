import matplotlib.pyplot as plt
import numpy as np
import george
from george import kernels
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter

# Define a simple process model
def f_process(x, dt):
    # Use GP to predict the next state
    mean, _ = gp.predict(x, return_var=True)
    return mean

# Define a measurement function for the EKF
def h_measurement(x):
    return np.array([x[0]])

# Define the Jacobian matrix for the measurement function
def HJacobian(x):
    return np.array([[1.0]])

# Define the measurement function for the EKF
def Hx(x):
    return np.array([x[0]])

# Parameters
buffer_size = 10
timesteps = 40
dt = 1

# Generate synthetic time series data (sine wave with noise)
x_series = np.linspace(0, timesteps - 1, timesteps)
y_series = np.sin(x_series) + np.random.normal(0, 0.1, x_series.shape)

# Initialize George GP for learning the process model
kernel = 1.0 * kernels.ExpSquaredKernel(1.0)
gp = george.GP(kernel)

# EKF initialization
ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
ekf.x = np.array([y_series[0]])
ekf.P *= 0.2
ekf.R = np.diag([0.1])

# Buffer for training the GP
y_buffer = np.zeros(buffer_size)

# Now loop through the time series data and predict with the EKF
ekf_predictions = np.zeros(timesteps)
ekf_confidence = np.zeros(timesteps)
for t in range(timesteps):
    # Update buffer with the latest measurement
    y_buffer[:-1] = y_buffer[1:]
    y_buffer[-1] = y_series[t]

    # Update GP with the buffer data
    gp.compute(x_series[-buffer_size:], yerr=1.0)
    mean, _ = gp.predict(y_buffer, x_series[-1], return_var=True)

    if t > 0:
        ekf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)[0, 0]

    ekf.predict()
    if t < timesteps - 1:
        next_measurement = np.array([y_series[t + 1]])
        ekf.update(next_measurement, HJacobian, Hx)

    # Use GP to predict the next state for the process model
    ekf.x = np.array([mean])

    ekf_predictions[t] = ekf.x[0]
    ekf_confidence[t] = ekf.P[0, 0]

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(x_series, y_series, 'k-', lw=2, label='True Data')
plt.plot(x_series, ekf_predictions, 'b--', lw=2, label='EKF Prediction')
plt.fill_between(x_series,
                 ekf_predictions - 2 * np.sqrt(ekf_confidence),
                 ekf_predictions + 2 * np.sqrt(ekf_confidence),
                 color='blue', alpha=0.3, label='Confidence Interval')
plt.legend()
plt.title('1-Minute Ahead Time Series Prediction using GP-EKF (George) for Process Model (Online Streaming)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
