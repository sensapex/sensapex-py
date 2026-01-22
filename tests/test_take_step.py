from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from sensapex.sensapex import UMP


class TestTakeStep(TestCase):
    """Test cases for um_take_step wrapper."""

    def setUp(self):
        """Set up a UMP instance with mocked SDK."""
        self.dev_id = 1
        self.ump = UMP(address=b"169.254.255.255", group=0, start_poller=False, handle_atexit=False)
        self.ump.call = Mock()

    def assertCallArgsEqual(self, actual_args, expected_values):
        """Assert that SDK call arguments match expected values.

        First element is the function name (string), rest are ctypes objects.
        """
        self.assertEqual(len(actual_args), len(expected_values),
            f"Argument count mismatch: expected {len(expected_values)}, got {len(actual_args)}")

        for i, (actual, expected) in enumerate(zip(actual_args, expected_values)):
            if isinstance(expected, str):
                self.assertEqual(actual, expected, f"Arg {i}: expected {expected}, got {actual}")
            elif isinstance(expected, float) and np.isnan(expected):
                self.assertTrue(np.isnan(actual.value),
                    f"Arg {i}: expected nan, got {actual.value}")
            else:
                self.assertEqual(actual.value, expected,
                    f"Arg {i}: expected {expected}, got {actual.value}")

    def test_array_speed_4axis(self):
        """Array of speeds passes explicit per-axis values."""
        distance = [10.5, -20.5, 30.5, -40.5]
        speed = [100, 200, 300, 400]

        self.ump.take_step(self.dev_id, distance, speed, mode=1, max_acceleration=500)
        actual_sdk_call_args = self.ump.call.call_args[0]
        self.assertCallArgsEqual(actual_sdk_call_args, [
            "um_take_step", self.dev_id,
            10.5, -20.5, 30.5, -40.5,
            100, 200, 300, 400,
            1, 500,
        ])

    def test_array_speed_pads_missing_axes(self):
        """Missing axes default to 0."""
        distance = [10.0, 20.0]
        speed = [100, 200]

        self.ump.take_step(self.dev_id, distance, speed)
        actual_sdk_call_args = self.ump.call.call_args[0]
        self.assertCallArgsEqual(actual_sdk_call_args, [
            "um_take_step", self.dev_id,
            10.0, 20.0, 0.0, 0.0,
            100, 200, 0, 0,
            0, 0,
        ])

    def test_single_speed_calculates_linear_speeds(self):
        """Single speed calculates proportional per-axis speeds (3-4-5 triangle)."""
        distance = [30.0, 40.0, 0.0, 0.0]
        speed = 100

        self.ump.take_step(self.dev_id, distance, speed)
        actual_sdk_call_args = self.ump.call.call_args[0]
         # speed_x = 100 * 30/50 = 60, speed_y = 100 * 40/50 = 80
        self.assertCallArgsEqual(actual_sdk_call_args, [
            "um_take_step", self.dev_id,
            30.0, 40.0, 0.0, 0.0,
            60, 80, 0, 0,
            0, 0,
        ])

    def test_single_speed_zero_distance_gets_zero_speed(self):
        """Axes with zero distance get zero speed (not min_speed)."""
        distance = [10.0, 0.0, 20.0, 0.0]
        speed = 100

        self.ump.take_step(self.dev_id, distance, speed)
        actual_sdk_call_args = self.ump.call.call_args[0]

        self.assertEqual(actual_sdk_call_args[7].value, 0)  # y speed = 0
        self.assertEqual(actual_sdk_call_args[9].value, 0)  # d speed = 0

    def test_single_speed_respects_min_speed(self):
        """Small proportional speeds are clamped to min_speed (1)."""
        distance = [100.0, 0.1, 0.0, 0.0]
        speed = 10

        self.ump.take_step(self.dev_id, distance, speed)
        actual_sdk_call_args = self.ump.call.call_args[0]

        self.assertGreaterEqual(actual_sdk_call_args[7].value, 1)
