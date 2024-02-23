import unittest
import numpy as np

class YourClass:
    def __init__(self, veltol):
        self.veltol = veltol

    def pos_at_zero_vel(self, current_pos, predictions, dt):
        vel_sub = [prediction for prediction in predictions if 0 < prediction <= self.veltol]

        if vel_sub:
            old_pos = current_pos

            for pred in vel_sub:
                print("old_pos:", old_pos, "pred:", pred, "current_pos:", current_pos)
                current_pos = old_pos + pred * dt
                old_pos = current_pos

        return current_pos

class TestYourClass(unittest.TestCase):
    def setUp(self):
        self.your_instance = YourClass(0.1)


    def assertAlmostEqual(self, first, second, delta=1e-9, msg=None):
        np.testing.assert_allclose(first, second, rtol=delta, atol=delta, err_msg=msg)


    def test_pos_at_zero_vel_no_change(self):
        current_pos = 10.0
        predictions = [0.1, 0.2, 0.3]
        dt = 0.5
        result = self.your_instance.pos_at_zero_vel(current_pos, predictions, dt)
        self.assertAlmostEqual(result, current_pos, delta=1e-10, msg="Position should remain unchanged.")

    def test_pos_at_zero_vel_with_changes(self):
        current_pos = 5.0
        predictions = [0.05, 0.1, 0.2, 0.15]
        dt = 0.5
        result = self.your_instance.pos_at_zero_vel(current_pos, predictions, dt)
        expected_result = current_pos + sum([pred * dt for pred in predictions if 0 < pred <= self.your_instance.veltol])
        self.assertAlmostEqual(result, expected_result, delta=1e-8, msg="Position should change according to the given conditions.")

    def test_pos_at_zero_vel_empty_predictions(self):
        current_pos = 8.0
        predictions = []
        dt = 0.5
        result = self.your_instance.pos_at_zero_vel(current_pos, predictions, dt)
        self.assertAlmostEqual(result, current_pos, delta=1e-8, msg="Position should remain unchanged when predictions list is empty.")

if __name__ == '__main__':
    unittest.main()
