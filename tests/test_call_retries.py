import threading
import warnings
from ctypes import c_int
from unittest import TestCase
from unittest.mock import Mock, call

from sensapex.sensapex import UMP, UMError


class TestUMPCallRetries(TestCase):
    def setUp(self):
        self.ump = object.__new__(UMP)
        self.ump.lock = threading.RLock()
        self.ump.h = object()
        self.ump._debug = False
        self.ump.lib = Mock()
        self.ump.lib.um_last_error.return_value = -2
        self.ump.lib.um_errorstr.return_value = b"timeout"

    def test_retry_passes_retries_as_keyword_not_positional(self):
        self.ump.lib.um_fake.side_effect = [-1, -1, -1, 5]
        arg = c_int(1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(self.ump.call("um_fake", arg, retries=3), 5)

        self.assertEqual(self.ump.lib.um_fake.call_args_list, [call(self.ump.h, arg)] * 4)
        self.assertEqual(len(caught), 3)

    def test_retry_exhaustion_raises_um_error(self):
        self.ump.lib.um_fake.return_value = -1
        arg = c_int(7)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(UMError):
                self.ump.call("um_fake", arg, retries=2)

        self.assertEqual(self.ump.lib.um_fake.call_args_list, [call(self.ump.h, arg)] * 3)
        self.assertEqual(len(caught), 2)
