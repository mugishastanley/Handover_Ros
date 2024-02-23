import unittest
from decimal import Decimal

class GPRNode:
    def __init__(self):
        # Initialize necessary attributes
        pass

    def update_position_with_velocities(self, current_pos, predictions, dt):
        """
        Update the current position based on predicted velocities.

        Args:
            current_pos (float): The current position.
            predictions (list): List of predicted velocities.
            dt (float): Time step.

        Returns:
            float: Updated position.
        """
        # Filter velocities greater than the tolerance
        valid_velocities = [vel for vel in predictions if vel > self.veltol]

        # Update position based on valid velocities
        for velocity in valid_velocities:
            current_pos += velocity * dt

        return current_pos
    
        # vel_sub=[]
        # if (any(value <= self.veltol and value > 0.0 for value in predictions)):
        #     for prediction in predictions:
        #         if prediction >= self.veltol :
        #             vel_sub.append(prediction)
        #     old_pos= current_pos

        #     for pred in vel_sub:
        #         #print("old_pos",old_pos,"pred:",pred,"current pos",current_pos)
        #         current_pos = old_pos + pred * dt
        #         old_pos = current_pos
        # return current_pos


class TestGPRNode(unittest.TestCase):
    def setUp(self):
        # Initialize an instance of GPRNode or mock necessary objects
        self.gpr_node = GPRNode()
        self.gpr_node.veltol = 0.05  # Set an example velocity tolerance

    def test_update_position_with_valid_velocities(self):
        # Test when there are valid velocities greater than the tolerance
        current_pos = 10.0
        predictions = [0.2, 0.3, 0.1, 0.25, ]
        dt = 0.1

        updated_pos = self.gpr_node.update_position_with_velocities(current_pos, predictions, dt)

        # The expected updated position is obtained by summing the product of each valid velocity and dt
        expected_pos = current_pos + sum([vel * dt for vel in predictions if vel > self.gpr_node.veltol])

        self.assertAlmostEqual(updated_pos, expected_pos, places=3)



    def test_update_position_with_no_valid_velocities(self):
        # Test when there are no valid velocities greater than the tolerance
        current_pos = Decimal('10.0')
        predictions = [Decimal('0.05'), Decimal('0.08'), Decimal('0.09')]
        dt = Decimal('0.1')

        updated_pos = self.gpr_node.update_position_with_velocities(current_pos, predictions, dt)

        # Set a small delta value based on your precision requirements
        delta = Decimal('1e-2')  # Adjust this value as needed

        # The expected updated position is the same as the current position since there are no valid velocities
        self.assertAlmostEqual(updated_pos, current_pos, delta=delta)


    def test_update_position_with_empty_predictions(self):
        # Test when predictions list is empty
        current_pos = 10.0
        predictions = []
        dt = 0.1
        updated_pos = self.gpr_node.update_position_with_velocities(current_pos, predictions, dt)

        # Round the values before comparison to avoid floating-point precision issues
        rounded_updated_pos = round(updated_pos, 5)  # Adjust the rounding precision as needed
        rounded_current_pos = round(current_pos, 5)

        self.assertEqual(rounded_updated_pos, rounded_current_pos)

if __name__ == '__main__':
    unittest.main()
