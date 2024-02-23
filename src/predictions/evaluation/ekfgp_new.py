import matplotlib.pyplot as plt
import numpy as np
import george
from george import kernels
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter

# Define a simple process model
def f_process(x, dt):
    process_noise_var = ekf.process_noise_var
    return x + np.random.normal(0, np.sqrt(process_noise_var), x.shape)

# Define a measurement function for the EKF that uses the GP mean
def h_measurement(x, gp):
    mean, variance = gp.predict(y_series, x, return_var=True)
    process_noise_var = variance
    return mean, process_noise_var

# Define the Jacobian matrix for the measurement function
def HJacobian(x):
    return np.array([[1.0]])

# Define the measurement function for the EKF
def Hx(x):
    return np.array([x[0]])

# Generate synthetic time series data (sine wave with noise)
timesteps = 25
dt = 1
x_series = np.linspace(0, timesteps - 1, timesteps)
y_series = np.sin(x_series) + np.random.normal(0, 0.1, x_series.shape)

# George GP initialization
kernel = 1.0 * kernels.ExpSquaredKernel(1.0)
gp = george.GP(kernel)
gp.compute(x_series, yerr=1.0)

# EKF initialization
ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
ekf.x = np.array([y_series[0]])
ekf.P *= 0.2
ekf.R = np.diag([0.1])

ekf.process_noise_var = 0.15

# Now loop through the time series data and predict with the EKF
ekf_predictions = np.zeros(timesteps)
ekf_confidence = np.zeros(timesteps)
for t in range(timesteps):
    if t > 0:
        _, process_noise_var = h_measurement(np.array([ekf_predictions[t - 1]]), gp)
        ekf.process_noise_var = process_noise_var
        ekf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=ekf.process_noise_var)[0, 0]

    ekf.predict()
    if t < timesteps - 1:
        next_measurement = np.array([y_series[t + 1]])
        ekf.update(next_measurement, HJacobian, Hx)

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
plt.title('1-Minute Ahead Time Series Prediction using GP-EKF (George)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()