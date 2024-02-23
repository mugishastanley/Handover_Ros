def zero_vel_point(timesteps, prediction_list, vtol):
    predicted_vel = [abs(prediction / timestep) for prediction, timestep in zip(prediction_list, timesteps)]
    print (f"predicted vel {predicted_vel}")
    
    for i, vel in enumerate(predicted_vel):
        if vel < vtol:
            return timesteps[i], prediction_list[i]
    
    return None, None  # If no velocity is below the threshold


# Example usage:
timesteps = [0.1, 0.2, 0.3, 0.4, 0.5]
prediction_list = [-1, -2, 1.5, 0.55, -0.2]
vtol = 5

time, prediction = zero_vel_point(timesteps, prediction_list, vtol)
if time is not None:
    print("Time:", time)
    print("Prediction:", prediction)
else:
    print("No point with velocity less than threshold found.")