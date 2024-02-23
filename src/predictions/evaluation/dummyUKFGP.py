
import matplotlib.pyplot as plt
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
# Define a simple process model that includes process noise from the GP variance

# Define a simple process model
def f_process(x, dt):
    # Access the `process_noise_var` from `ukf` and use it for the process noise
    process_noise_var = ukf.process_noise_var
    return x + np.random.normal(0, process_noise_var, x.shape)

# Define a measurement function for the UKF that uses the GP mean
def h_measurement(x, gp):
    # Now also return the variance which represents process noise
    mean, variance = gp.predict(x.reshape(-1, 1), return_std=True)
    process_noise_var = variance ** 2  # Squaring std to get the variance
    return mean, process_noise_var

# Generate synthetic time series data (sine wave with noise)
timesteps = 25
dt = 1  # In this scenario, 1 unit of time is equivalent to 1 minute
x_series = np.linspace(0, timesteps-1, timesteps)
y_series = np.sin(x_series) + np.random.normal(0, 0.1, x_series.shape)
# Gaussian Process initialization
kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel)
train_points = x_series.reshape(-1, 1)
train_targets = y_series
gp.fit(train_points, train_targets)

# UKF initialization
points = MerweScaledSigmaPoints(n=1, alpha=0.1, beta=2., kappa=0)
ukf = UnscentedKalmanFilter(dim_x=1, dim_z=1, dt=dt, hx=lambda x: h_measurement(x, gp)[0], fx=f_process, points=points)
ukf.x = np.array([train_targets[0]])  # Initialize with the first data point
ukf.P *= 0.2
ukf.R = np.diag([0.1])  # Measurement noise

ukf.process_noise_var = 0.1  # This default value will be updated each step in the loop
# Now loop through the time series data and predict with the UKF
ukf_predictions = np.zeros(timesteps)
ukf_confidence = np.zeros(timesteps)
for t in range(timesteps):
    # We need the process noise variance from the GP to be passed to the predict step
    if t > 0:
        _, process_noise_var = h_measurement(np.array([ukf_predictions[t - 1]]), gp)
        ukf.process_noise_var = process_noise_var
        ukf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=ukf.process_noise_var)[0, 0]
    ukf.predict()
    if t < timesteps - 1:
        next_measurement = np.array([y_series[t + 1]])
        ukf.update(next_measurement)
    # Store predictions and confidence
    ukf_predictions[t] = ukf.x[0]
    ukf_confidence[t] = ukf.P[0, 0]

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(x_series, y_series, 'k-', lw=2, label='True Data')
plt.plot(x_series, ukf_predictions, 'r--', lw=2, label='UKF Prediction')
plt.fill_between(x_series, 
                 ukf_predictions - 2 * np.sqrt(ukf_confidence),
                 ukf_predictions + 2 * np.sqrt(ukf_confidence),
                 color='orange', alpha=0.3, label='Confidence Interval')
plt.legend()
plt.title('1-Minute Ahead Time Series Prediction using GP-UKF')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()