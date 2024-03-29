import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import matplotlib.pyplot as plt
from filterpy.common import Q_discrete_white_noise

def fx(x, dt):
    F = np.array([[1, dt]], dtype=float)
    return np.dot(F, x)

def hx(x):
    # where measurements are [x_pos]
    return np.array([x[0]])

dt = 0.1
# create sigma points to use in the filter. This is standard for Gaussian processes
points = MerweScaledSigmaPoints(2, alpha=.1, beta=2., kappa=-1)

kf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=dt, fx=fx, hx=hx, points=points)
kf.P *= 0.2 # initial uncertainty
z_std = 0.1
kf.x = np.array([0, 0.5])  # initial state
kf.R = np.diag([z_std**2]) # 1 standard

# Use a simple diagonal covariance matrix for process noise
kf.Q = np.diag([z_std**2, z_std**2]) # 1 standard
#kf.Q = Q_discrete_white_noise(dim=1, dt=dt, var=0.01**2)

timesteps = 50
true_data = np.sin(np.linspace(0, 2 * np.pi, timesteps))  # True data is generated using a sine function
noisy_data = true_data + np.random.normal(0, z_std, timesteps)  # Noisy data is generated by adding random noise to the true data

filtered_data = []
covariances = []
for z in noisy_data:
    kf.predict()
    kf.update([z])
    filtered_data.append(kf.x[0])
    covariances.append(kf.P[0, 0])

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(true_data, label='True Data')
plt.scatter(range(len(noisy_data)), noisy_data, color='r', label='Noisy Data')
plt.plot(filtered_data, label='Filtered Data')
plt.legend()
plt.title('True Data, Noisy Data, and Filtered Data')
plt.xlabel('Time')
plt.ylabel('Value')

plt.subplot(2, 1, 2)
plt.plot(covariances, label='Covariance')
plt.legend()
plt.title('Covariance over Time')
plt.xlabel('Time')
plt.ylabel('Covariance')

plt.tight_layout()
plt.show()
