import numpy as np

def find_position_at_zero_velocity(velocities, current_position, dt, zero_velocity_threshold):
    # Find the index where velocity first crosses zero
    zero_velocity_index = np.argmax(np.abs(velocities) < zero_velocity_threshold)

    # If zero velocity is not found, return current position
    if zero_velocity_index == len(velocities) - 1:
        return current_position

    # Integrate velocities from the current position up to the zero velocity point
    positions = np.cumsum(velocities[:zero_velocity_index]) * dt

    # Calculate the position at zero velocity
    position_at_zero_velocity = current_position + positions[-1]

    return position_at_zero_velocity

# Example usage:
velocities = [1.0, 2.0, 0.5, -1.0, -2.0, -0.5, 0.0, 1.0]
current_position = 2.0
dt = 1.0
zero_velocity_threshold = 0.0

result = find_position_at_zero_velocity(velocities, current_position, dt, zero_velocity_threshold)
print("Position at zero velocity:", result)
