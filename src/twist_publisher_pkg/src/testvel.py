
def zero_vel_point(timesteps, prediction_list, vtol):
    predicted_vel = []
    for i in range(1, len(timesteps)):
        delta_prediction = prediction_list[i] - prediction_list[i - 1]
        delta_time = timesteps[i] - timesteps[i - 1]
        if delta_time == 0:  # To avoid division by zero
            predicted_vel.append(float('inf'))
        else:
            predicted_vel.append(abs(delta_prediction / delta_time))

    print(f"predicted vel{predicted_vel}")
    
    for i, vel in enumerate(predicted_vel):
        if vel < vtol:
            return timesteps[i+1], prediction_list[i+1]
    
    return None, None  # If no velocity is below the threshold


# Example usage:
timesteps = [0.1, 0.2, 0.3, 0.4, 0.5]
prediction_list = [1, 2, 1.5, 0.5, 0.2]
vtol = 8

time, prediction = zero_vel_point(timesteps, prediction_list, vtol)
if time is not None:
    print("Time:", time)
    print("Prediction:", prediction)
else:
    print("No point with velocity less than threshold found.")
