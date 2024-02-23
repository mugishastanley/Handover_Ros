import numpy as np
import george
from george import kernels
import matplotlib.pyplot as plt
import scipy.optimize as op

def initialize_state_covariance(initial_state, initial_state_estimate):
    state_covariance = np.outer(initial_state - initial_state_estimate, initial_state - initial_state_estimate)
    return state_covariance

def gp_regression(training_data_X, training_data_Y, test_point, kernel):
    gp = george.GP(kernel)
    gp.compute(training_data_X.flatten())
    mean, variance = gp.predict(training_data_Y.flatten(), test_point.flatten())
    return mean[0], variance[0]

# def compute_jacobian(previous_state_estimate, process_noise_covariance):
#     jacobian = np.eye(len(previous_state_estimate))
#     return jacobian

def compute_jacobian(previous_position_estimate, process_noise_covariance):
    jacobian = np.array([[1.0]])  # Since the state is 1-dimensional (position)
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
    jacobian_measurement = np.array([[1.0]])
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
true_positions = np.cumsum(np.random.normal(size=time_steps))
measurements = true_positions + np.random.normal(scale=0.5, size=time_steps)

# Initialization
initial_position = 0.0
initial_position_estimate = 0.0
initial_state_covariance = initialize_state_covariance(np.array([initial_position]), np.array([initial_position_estimate]))
process_noise_covariance = np.eye(1)
measurement_noise_covariance = np.eye(1)

# Initialize the GP model with one data point
kernel = 1.0 * kernels.ExpSquaredKernel(1.0)
gp = george.GP(kernel)

# Main loop
estimated_positions = np.zeros(time_steps)
predicted_positions = np.zeros(time_steps)

current_position_estimate = initial_position_estimate
current_covariance = initial_state_covariance

for t in range(1, time_steps):
    # Prediction Step
    predicted_position, predicted_covariance = gp_regression(np.array([[current_position_estimate]]),
                                                             np.array([[measurements[t-1]]]), np.array([[true_positions[t]]]),
                                                             kernel)
    jacobian_process_model = compute_jacobian(np.array([current_position_estimate]), process_noise_covariance)
    current_covariance = update_covariance_matrix(predicted_covariance, process_noise_covariance,
                                                  jacobian_process_model)

    # Measurement Update Step
    current_position_estimate, current_covariance = measurement_update(np.array([predicted_position]), current_covariance,
                                                                       measurements[t], measurement_noise_covariance)

    # # Update the GP model with the new data point
    # gp.compute(np.array([[true_positions[t]]]))
    # #gp.optimize(messages=True)  # Optional: Optimize hyperparameters

        # Update the GP model with the new data point
    gp.compute(np.array([[true_positions[t]]]))
    gp.kernel.pars = np.log([1.0, 1.0])  # Manually set the initial hyperparameters
    result = op.minimize(gp.nll, gp.kernel.pars, args=(np.array([measurements[t]]),))
    gp.kernel.pars = result.x


    # Store the current state estimate
    estimated_positions[t] = current_position_estimate
    predicted_positions[t] = predicted_position

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(true_positions, label='True Positions', marker='o', linestyle='-', color='blue')
plt.plot(measurements, label='Measurements', marker='x', linestyle='None', color='green')
plt.plot(predicted_positions, label='Predicted Positions (Before Update)', marker='o', linestyle='--', color='orange')
plt.plot(estimated_positions, label='Estimated Positions (After Update)', marker='o', linestyle='-', color='red')
plt.title('True Positions, Measurements, and Estimated Positions')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()

plt.tight_layout()
plt.show()
