class YourClass:
    def __init__(self, veltol=0.1):
        self.veltol = veltol

    def pos_at_zero_vel(self, current_pos, predictions, dt):
        vel_sub = []
        if any(value <= self.veltol for value in predictions):
            for prediction in predictions:
                if prediction >= self.veltol:
                    vel_sub.append(prediction)
            old_pos = current_pos

            for prediction in vel_sub:
                current_pos = old_pos + prediction * dt
                old_pos = current_pos
        return current_pos

# Example usage:
if __name__ == "__main__":
    obj = YourClass()

    # Example 1
    current_pos_1 = 10.0
    predictions_1 = [0.3, 0.2, 0.1]
    dt_1 = 0.1
    result_1 = obj.pos_at_zero_vel(current_pos_1, predictions_1, dt_1)
    print("Example 1 Result:", result_1)

    # Example 2
    current_pos_2 = 5.0
    predictions_2 = [0.05, 0.2, 0.3, 0.05]
    dt_2 = 0.2
    result_2 = obj.pos_at_zero_vel(current_pos_2, predictions_2, dt_2)
    print("Example 2 Result:", result_2)
