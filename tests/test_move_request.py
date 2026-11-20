from ctypes import c_float, c_int
from unittest import TestCase, skip
from unittest.mock import Mock

import numpy as np

from sensapex.sensapex import MoveRequest


class TestMoveRequest(TestCase):
    def setUp(self):
        self.dev_id = 1
        self.max_accel = 1
        self.mock_ump = Mock()
        self.mock_ump.default_max_accelerations = {self.dev_id: self.max_accel}
        self.start_pos = (0., 0., 0.)
        self.mock_ump.get_pos = Mock(return_value=self.start_pos)
        self.mock_ump.call = Mock()

    def assertCTypesArgsEqual(self, actual_args, expected_values, msg=None):
        """Assert that ctypes arguments match expected Python values.

        Args:
            actual_args: tuple of ctypes objects from mock.call_args
            expected_values: list of expected Python values (int, float, or nan)
            msg: optional message prefix for failure output
        """
        self.assertEqual(len(actual_args), len(expected_values),
            f"Argument count mismatch: expected {len(expected_values)}, got {len(actual_args)}")

        for i, (actual, expected) in enumerate(zip(actual_args, expected_values)):
            actual_val = actual.value
            if np.isnan(expected):
                self.assertTrue(np.isnan(actual_val),
                    f"Arg {i}: expected nan, got {actual_val}")
            else:
                self.assertEqual(actual_val, expected,
                    f"Arg {i}: expected {expected}, got {actual_val}")

    def test_start_sends_proper_args(self):
        dest = (4., 1., 1.)
        speed = 2
        mode = 1  # simultaneous
        move = MoveRequest(self.mock_ump, self.dev_id, dest, speed, simultaneous=True)
        self.mock_ump.call.assert_not_called()
        move.start()
        actual_call_args = self.mock_ump.call.call_args[0][1:]  # skip "um_goto_position_ext"
        self.assertCTypesArgsEqual(actual_call_args, [
            self.dev_id, dest[0], dest[1], dest[2], np.nan, speed, speed, speed, speed, mode, self.max_accel,
        ])

    def test_simultaneous_moves_do_not_have_more_calls_to_make(self):
        dest = (4., 1., 1.)
        speed = 2
        move = MoveRequest(self.mock_ump, self.dev_id, dest, speed, simultaneous=True)
        move.start()
        self.assertFalse(move.has_more_calls_to_make())

    def test_nonsimultaneous_moves_have_3_calls_to_make(self):
        dest = (4., 1., 1.)
        speed = 2
        move = MoveRequest(self.mock_ump, self.dev_id, dest, speed, simultaneous=False)
        move.start()
        self.assertTrue(move.has_more_calls_to_make())
        move.make_next_call()
        self.assertTrue(move.has_more_calls_to_make())
        move.make_next_call()
        self.assertFalse(move.has_more_calls_to_make())

    @skip("Not implemented yet")
    def test_nonsimultaneous_only_move_for_changed_values(self):
        dest = (self.start_pos[0], 1., 1.)
        speed = 2
        move = MoveRequest(self.mock_ump, self.dev_id, dest, speed, simultaneous=False)
        move.start()
        self.assertTrue(move.has_more_calls_to_make())
        move.make_next_call()
        self.assertFalse(move.has_more_calls_to_make())

    def test_xzy_first_for_extraction(self):
        self.mock_ump.get_device.return_value.is_stage = False
        dest = (-4., 1., 1.)
        speed = 2.0
        mode = 0
        move = MoveRequest(self.mock_ump, self.dev_id, dest, speed, simultaneous=False)

        # Move 1: X only (extraction pulls X first)
        move.start()
        actual_call_args = self.mock_ump.call.call_args[0][1:]  # skip "um_goto_position_ext"
        self.assertCTypesArgsEqual(actual_call_args, [
            self.dev_id, dest[0], np.nan, np.nan, np.nan, speed, speed, speed, speed, mode, self.max_accel,
        ])

        # Move 2: X + Z
        move.make_next_call()
        actual_call_args = self.mock_ump.call.call_args[0][1:]
        self.assertCTypesArgsEqual(actual_call_args, [
            self.dev_id, dest[0], np.nan, dest[2], np.nan, speed, speed, speed, speed, mode, self.max_accel,
        ])

        # Move 3: X + Y + Z
        move.make_next_call()
        actual_call_args = self.mock_ump.call.call_args[0][1:]
        self.assertCTypesArgsEqual(actual_call_args, [
            self.dev_id, dest[0], dest[1], dest[2], np.nan, speed, speed, speed, speed, mode, self.max_accel,
        ])

    def test_yzx_for_insertion(self):
        self.mock_ump.get_device.return_value.is_stage = False
        dest = (4., 1., 1.)
        speed = 2
        mode = 0
        move = MoveRequest(self.mock_ump, self.dev_id, dest, speed, simultaneous=False)

        # Move 1: Y only (insertion moves Y first)
        move.start()
        actual_call_args = self.mock_ump.call.call_args[0][1:]  # skip "um_goto_position_ext"
        self.assertCTypesArgsEqual(actual_call_args, [
            self.dev_id, np.nan, dest[1], np.nan, np.nan, speed, speed, speed, speed, mode, self.max_accel,
        ])

        # Move 2: Y + Z
        move.make_next_call()
        actual_call_args = self.mock_ump.call.call_args[0][1:]
        self.assertCTypesArgsEqual(actual_call_args, [
            self.dev_id, np.nan, dest[1], dest[2], np.nan, speed, speed, speed, speed, mode, self.max_accel,
        ])

        # Move 3: X + Y + Z
        move.make_next_call()
        actual_call_args = self.mock_ump.call.call_args[0][1:]
        self.assertCTypesArgsEqual(actual_call_args, [
            self.dev_id, dest[0], dest[1], dest[2], np.nan, speed, speed, speed, speed, mode, self.max_accel,
        ])


class TestStallStep(TestCase):
    def setUp(self):
        self.dev_id = 1
        self.ump = Mock()
        self.ump.default_max_accelerations = {self.dev_id: 0}
        self.ump.get_pos = Mock(return_value=(0., 0., 0.))

    def _retry_after(self, dest, moved_to):
        """Start a move to *dest*, end its first attempt at *moved_to*, then start the retry."""
        move = MoveRequest(self.ump, self.dev_id, dest, 470, simultaneous=True, linear=True, max_attempts=3)
        move.start()
        self.ump.get_pos.return_value = moved_to
        move.start()
        return move

    def _step_calls(self):
        return [c for c in self.ump.mock_calls if c[0] == "take_step"]

    def test_first_attempt_never_steps(self):
        move = MoveRequest(self.ump, self.dev_id, (0., 0., -700.), 470, simultaneous=True, linear=True)
        move.start()
        self.assertEqual(self._step_calls(), [])

    def test_stalled_axis_steps_toward_target_before_the_retry(self):
        move = self._retry_after((0., 0., -700.), (0., 0., 0.))
        names = [c[0] for c in self.ump.mock_calls if c[0] in ("take_step", "call")]
        self.assertEqual(names, ["call", "take_step", "call"])
        dev, distance, speed = self._step_calls()[0][1]
        self.assertEqual(dev, self.dev_id)
        np.testing.assert_array_equal(distance, [0, 0, -1, 0])
        np.testing.assert_array_equal(speed, [0, 0, 470, 0])
        self.assertEqual(move.stall_steps, 1)

    def test_axis_that_closed_most_of_the_gap_does_not_step(self):
        self._retry_after((0., 0., -700.), (0., 0., -600.))
        self.assertEqual(self._step_calls(), [])

    def test_short_miss_does_not_step(self):
        self._retry_after((0., 0., -3.), (0., 0., 0.))
        self.assertEqual(self._step_calls(), [])

    def test_only_the_stalled_axis_steps(self):
        self._retry_after((500., 0., 700.), (500., 0., 0.))
        np.testing.assert_array_equal(self._step_calls()[0][1][1], [0, 0, 1, 0])
