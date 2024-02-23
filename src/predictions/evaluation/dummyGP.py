import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Generate synthetic data
timesteps = 50
true_data = np.sin(np.linspace(0, 2 * np.pi, timesteps))
z_std = 0.1
noisy_data = true_data + np.random.normal(0, z_std, timesteps)

# Reshape data for scikit-learn
X = np.arange(timesteps).reshape(-1, 1)
y = noisy_data.reshape(-1, 1)

# Define Gaussian process kernel
#kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
kernel = 0.5 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)

# Create Gaussian process regressor
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
# Fit the Gaussian process to the data
gp.fit(X, y)

# Make predictions
x_pred = np.arange(0, timesteps, 0.1).reshape(-1, 1)
y_pred, sigma = gp.predict(x_pred, return_std=True)

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(true_data, label='True Data')
plt.scatter(range(len(noisy_data)), noisy_data, color='r', label='Noisy Data')
plt.plot(x_pred, y_pred, label='Gaussian Process Prediction')
plt.fill_between(x_pred.flatten(), (y_pred - sigma).flatten(), (y_pred + sigma).flatten(), alpha=0.2, color='blue')
plt.legend()
plt.title('True Data, Noisy Data, and Gaussian Process Prediction')
plt.xlabel('Time')
plt.ylabel('Value')

plt.subplot(2, 1, 2)
# No need to plot covariance for Gaussian Processes
plt.title('Covariance over Time (not applicable for Gaussian Processes)')
plt.xlabel('Time')
plt.ylabel('Covariance')

plt.tight_layout()
plt.show()