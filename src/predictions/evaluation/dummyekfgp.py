import numpy as np
import GPy
import matplotlib.pyplot as plt

def initialize_state_covariance(initial_state, initial_state_estimate):
    state_covariance = np.outer(initial_state - initial_state_estimate, initial_state - initial_state_estimate)
    return state_covariance

def gp_regression(training_data_X, training_data_Y, test_point, kernel):
    gp = GPy.models.GPRegression(training_data_X, training_data_Y, kernel)
    gp.optimize(messages=True)

    mean, variance = gp.predict(test_point.reshape(1, -1))
    return mean[0, 0], variance[0, 0]

def compute_jacobian(previous_state_estimate, process_noise_covariance):
    jacobian = np.eye(len(previous_state_estimate))
    return jacobian

def update_covariance_matrix(predicted_covariance, process_noise_covariance, jacobian):
    covariance_matrix = predicted_covariance - np.dot(np.dot(np.dot(predicted_covariance, jacobian.T),
                                                           np.linalg.inv(np.dot(np.dot(jacobian, predicted_covariance),
                                                                              jacobian.T) + process_noise_covariance)),
                                                      np.dot(np.dot(predicted_covariance, jacobian.T),
                                                             np.linalg.inv(np.dot(np.dot(jacobian, predicted_covariance),
                                                                                jacobian.T) + process_noise_covariance)))
    return covariance_matrix



def gp_regression_measurement_update(previous_state_estimate):
    return np.array([previous_state_estimate]), np.array([[1.0]])  # mean and variance of the measurement


def compute_jacobian_measurement_update(previous_state_estimate):
    jacobian_measurement = np.array([[1.0, 0.0]])
    return jacobian_measurement

def measurement_update(previous_state_estimate, covariance_matrix, measurement, measurement_noise_covariance):
    predicted_measurement, predicted_measurement_covariance = gp_regression_measurement_update(previous_state_estimate)
    jacobian_measurement = compute_jacobian_measurement_update(previous_state_estimate)

    kalman_gain = np.dot(np.dot(covariance_matrix, jacobian_measurement.T),
                         np.linalg.inv(np.dot(np.dot(jacobian_measurement, covariance_matrix),
                                            jacobian_measurement.T) + measurement_noise_covariance))

    updated_state_estimate = previous_state_estimate + np.dot(kalman_gain, (measurement - predicted_measurement))
    updated_covariance = np.dot(np.eye(len(updated_state_estimate)) - np.dot(kalman_gain, jacobian_measurement),
                                covariance_matrix)

    return updated_state_estimate, updated_covariance

# Sample time series data
np.random.seed(42)
time_steps = 100
true_states = np.cumsum(np.random.normal(size=(time_steps, 2)), axis=0)
measurements = true_states[:, 0] + np.random.normal(scale=0.5, size=time_steps)

# Initialization
initial_state = np.array([0.0, 0.0])
initial_state_estimate = np.array([0.0, 0.0])
initial_state_covariance = initialize_state_covariance(initial_state, initial_state_estimate)
process_noise_covariance = np.eye(len(initial_state))
measurement_noise_covariance = np.eye(1)

# Initialize the GP model with some initial training data
gp = GPy.models.GPRegression(np.zeros((1, 1)), np.zeros((1, 1)), GPy.kern.RBF(input_dim=1))
gp.set_XY(np.array([[initial_state[0]]]), np.array([[measurements[0]]]))

# Main loop
estimated_states = np.zeros((time_steps, len(initial_state)))
predicted_states = np.zeros((time_steps, len(initial_state)))

current_state_estimate = initial_state_estimate
current_covariance = initial_state_covariance

for t in range(1, time_steps):
    # Prediction Step
    predicted_state, predicted_covariance = gp_regression(np.array([[current_state_estimate[0]]]),
                                                          np.array([[measurements[t-1]]]), np.array([[true_states[t, 0]]]),
                                                          gp.kern)
    jacobian_process_model = compute_jacobian(current_state_estimate, process_noise_covariance)
    current_covariance = update_covariance_matrix(predicted_covariance, process_noise_covariance,
                                                  jacobian_process_model)

    # Measurement Update Step
    current_state_estimate, current_covariance = measurement_update(predicted_state, current_covariance,
                                                                    measurements[t], measurement_noise_covariance)

    # # Update the GP model with the new data point
    # gp.update_model(False)  # Set to True if you want to optimize hyperparameters
    # X = np.vstack([gp.X, np.array([[true_states[t, 0]]])])
    # y = np.vstack([gp.Y, np.array([[measurements[t]]])])
    # gp.set_XY(X, y)

       # Update the GP model with the new data point
    gp.set_XY(np.vstack([gp.X, np.array([[true_states[t, 0]]])]),
              np.vstack([gp.Y, np.array([[measurements[t]]])]))
    gp.optimize(messages=True)  # Optional: Optimize hyperparameters

    # Store the current state estimate
    estimated_states[t] = current_state_estimate
    predicted_states[t] = predicted_state

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(true_states[:, 0], label='True States', marker='o', linestyle='-', color='blue')
plt.plot(measurements, label='Measurements', marker='x', linestyle='None', color='green')
plt.plot(predicted_states[:, 0], label='Predicted States (Before Update)', marker='o', linestyle='--', color='orange')
plt.plot(estimated_states[:, 0], label='Estimated States (After Update)', marker='o', linestyle='-', color='red')
plt.title('True States, Measurements, and Estimated States')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(true_states[:, 1], label='True Velocities', marker='o', linestyle='-', color='blue')
plt.plot(estimated_states[:, 1], label='Estimated Velocities', marker='o', linestyle='-', color='red')
plt.title('True Velocities and Estimated Velocities')
plt.xlabel('Time Step')
plt.ylabel('Velocity')
plt.legend()

plt.tight_layout()
plt.show()
