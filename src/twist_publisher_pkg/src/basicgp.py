import GPy
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic time series data
np.random.seed(42)
time_steps = np.arange(0, 10, 0.1)
y_train = np.sin(time_steps) + 0.1 * np.random.randn(len(time_steps))

# Use a subset of the time series for training
train_size = 80
X_train = time_steps[:train_size].reshape(-1, 1)
y_train = y_train[:train_size]

# Remaining time steps for testing
X_test = time_steps[train_size:].reshape(-1, 1)

# Define the squared exponential kernel
kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)

# Create a GP regression model
gp_model = GPy.models.GPRegression(X_train, y_train.reshape(-1, 1), kernel)

# Optimize the model parameters
gp_model.optimize(messages=True)

# Predict on future time steps
mu, cov = gp_model.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time_steps[:train_size], y_train, c='red', marker='o', label='Training Data')
plt.plot(time_steps[train_size:], mu, color='blue', label='GP Prediction')
plt.fill_between(time_steps[train_size:], (mu - np.sqrt(cov)).ravel(), (mu + np.sqrt(cov)).ravel(), alpha=0.2, color='blue')
plt.title('Gaussian Process Time Series Prediction with GPy')
plt.xlabel('Time Steps')
plt.ylabel('y')
plt.legend()
plt.show()
