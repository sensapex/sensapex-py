import atexit
import ctypes
import os
import platform
import sys
import threading
import time
from ctypes import (
    c_int,
    c_uint,
    c_ulong,
    c_short,
    c_ushort,
    c_byte,
    c_void_p,
    c_char,
    c_char_p,
    c_longlong,
    byref,
    POINTER,
    pointer,
    Structure,
    c_float,
)
from timeit import default_timer
from typing import Dict

import numpy as np

SOCKET = c_int
if sys.platform == "win32" and platform.architecture()[0] == "64bit":
    SOCKET = c_longlong

LIBUM_MAX_MANIPULATORS = 254
LIBUM_MAX_LOG_LINE_LENGTH = 256
LIBUM_DEF_TIMEOUT = 20
LIBUM_DEF_BCAST_ADDRESS = b"169.254.255.255"
LIBUM_DEF_GROUP = 0
LIBUM_MAX_MESSAGE_SIZE = 1502
LIBUM_ARG_UNDEF = float("nan")
X_AXIS = 1
Y_AXIS = 2
Z_AXIS = 4
D_AXIS = 8

# error codes
LIBUM_NO_ERROR     = (0,)   # No error
LIBUM_OS_ERROR     = (-1,)  # Operating System level error
LIBUM_NOT_OPEN     = (-2,)  # Communication socket not open
LIBUM_TIMEOUT      = (-3,)  # Timeout occurred
LIBUM_INVALID_ARG  = (-4,)  # Illegal command argument
LIBUM_INVALID_DEV  = (-5,)  # Illegal Device Id
LIBUM_INVALID_RESP = (-6,)  # Illegal response received


class sockaddr_in(Structure):
    _fields_ = [
        ("family", c_short),
        ("port", c_ushort),
        ("in_addr", c_byte * 4),
        ("zero", c_byte * 8),
    ]


log_func_ptr = ctypes.CFUNCTYPE(c_void_p, c_int, c_void_p, POINTER(c_char), POINTER(c_char))


class um_positions(Structure):
    _fields_ = [
        ("x", c_int),
        ("y", c_int),
        ("z", c_int),
        ("w", c_int),
        ("updated", c_ulong),
    ]


# used in v0.600 and later
class um_state_v0_600(Structure):
    _fields_ = [
        ("last_received_time", c_ulong),
        ("socket", SOCKET),
        ("own_id", c_int),
        ("message_id", c_int),
        ("last_device_sent", c_int),
        ("last_device_received", c_int),
        ("retransmit_count", c_int),
        ("refresh_time_limit", c_int),
        ("last_error", c_int),
        ("last_os_errno", c_int),
        ("timeout", c_int),
        ("udp_port", c_int),
        ("last_status", c_int * LIBUM_MAX_MANIPULATORS),
        ("drive_status", c_int * LIBUM_MAX_MANIPULATORS),
        ("drive_status_id", c_ushort * LIBUM_MAX_MANIPULATORS),
        ("addresses", sockaddr_in * LIBUM_MAX_MANIPULATORS),
        ("cu_address", sockaddr_in),
        ("last_positions", um_positions * LIBUM_MAX_MANIPULATORS),
        ("laddr", sockaddr_in),
        ("raddr", sockaddr_in),
        ("errorstr_buffer", c_char * LIBUM_MAX_LOG_LINE_LENGTH),
        ("verbose", c_int),
        ("log_func_ptr", log_func_ptr),
        ("log_print_arg", c_void_p),
    ]


# used before v0.600
class um_state_pre_v0_600(Structure):
    _fields_ = [
        ("last_received_time", c_ulong),
        ("socket", SOCKET),
        ("own_id", c_int),
        ("message_id", c_int),
        ("last_device_sent", c_int),
        ("last_device_received", c_int),
        ("retransmit_count", c_int),
        ("refresh_time_limit", c_int),
        ("last_error", c_int),
        ("last_os_errno", c_int),
        ("timeout", c_int),
        ("udp_port", c_int),
        ("last_status", c_int * LIBUM_MAX_MANIPULATORS),
        ("addresses", sockaddr_in * LIBUM_MAX_MANIPULATORS),
        ("cu_address", sockaddr_in),
        ("last_positions", um_positions * LIBUM_MAX_MANIPULATORS),
        ("laddr", sockaddr_in),
        ("raddr", sockaddr_in),
        ("errorstr_buffer", c_char * LIBUM_MAX_LOG_LINE_LENGTH),
        ("verbose", c_int),
        ("log_func_ptr", log_func_ptr),
        ("log_print_arg", c_void_p),
    ]


class MoveRequest(object):
    """Class for coordinating and tracking moves.
    """
    max_retries = 3

    def __init__(self, ump, dev, dest, speed, simultaneous=True, linear=False, max_acceleration=0, retry_threshold=0.4):
        dest = [float(x) for x in dest]

        self._next_move_index = 0
        self.dev = dev
        self.finished = False
        self.finished_event = threading.Event()
        self.interrupt_reason = None
        self.interrupted = False
        self.last_pos = None
        self._retries = 0
        self._retry_threshold = np.array([retry_threshold] * 4)
        self.speed = speed
        self.start_time = timer()
        self.target_pos = dest
        self.ump = ump

        linear = linear and simultaneous
        dest4 = dest + [float("nan")] * (4 - len(dest))  # extend to 4 values
        dest4 = [d if d is not None else float("nan") for d in dest4]

        self.start_pos = self._read_position()
        diff = [float(d - c) for d, c in zip(dest4, self.start_pos) if d != float("nan")]
        if linear:
            dist = max(1., np.linalg.norm(diff))
            speed = [max(1., speed * abs(d / dist)) for d in diff]
            speed = speed + [0] * (4 - len(speed))
        else:
            speed = [max(1., speed)] * 4  # speed < 1 crashes the uMp

        if max_acceleration == 0 or max_acceleration is None:
            if self.ump.default_max_accelerations[dev] is not None:
                max_acceleration = self.ump.default_max_accelerations[dev]
            else:
                max_acceleration = 0

        if simultaneous:
            self.estimated_duration = max(np.array(diff) / speed[: len(diff)])
            self._moves = (self._movement_args(max_acceleration, dest4, speed, simultaneous),)
        else:
            self.estimated_duration = sum(np.array(diff) / speed[: len(diff)])
            if self.start_pos[0] < dest[0]:  # starting behind the dest means insertion
                just_y = dest4[:]
                just_y[0] = float("nan")
                just_y[2] = float("nan")
                just_yz = dest4[:]
                just_yz[0] = float("nan")
                self._moves = (
                    self._movement_args(max_acceleration, just_y, speed, simultaneous),
                    self._movement_args(max_acceleration, just_yz, speed, simultaneous),
                    self._movement_args(max_acceleration, dest4, speed, simultaneous),
                )
            else:  # extraction
                # TODO handle nan for x, as well as start == dest?
                just_x = dest4[:]
                just_x[1] = float("nan")
                just_x[2] = float("nan")
                just_xz = dest4[:]
                just_xz[1] = float("nan")
                self._moves = (
                    self._movement_args(max_acceleration, just_x, speed, simultaneous),
                    self._movement_args(max_acceleration, just_xz, speed, simultaneous),
                    self._movement_args(max_acceleration, dest4, speed, simultaneous),
                )

    def _movement_args(self, max_acceleration, pos4, speed, simultaneous):
        mode = int(bool(simultaneous))  # whether all axes move simultaneously
        return [c_int(self.dev)] + [c_float(x) for x in pos4] + [c_int(int(x)) for x in speed + [mode] + [max_acceleration]]

    def interrupt(self, reason):
        self.ump.call("um_stop", c_int(self.dev))
        self.interrupt_reason = reason
        self.interrupted = True
        self.finished = True
        self.finished_event.set()

    def finish(self):
        self.last_pos = self._read_position()
        self.finished = True
        self.finished_event.set()

    def start(self):
        self._next_move_index = 0
        self.make_next_call()

    def is_in_progress(self):
        return self.ump.is_busy(self.dev)

    def can_retry(self):
        return self._retries < self.max_retries

    def _read_position(self):
        return np.array(self.ump.get_pos(self.dev, timeout=-1))

    def is_close_enough(self):
        pos = self._read_position()
        target = np.array(self.target_pos).astype(float)
        err = np.abs(pos - target)
        mask = np.isfinite(err)
        return np.all(err[mask] < self._retry_threshold[: len(mask)][mask])

    def retry(self):
        self._retries += 1
        self.start()

    def has_more_calls_to_make(self):
        return self._next_move_index < len(self._moves)

    def make_next_call(self):
        self.ump.call("um_goto_position_ext", *self._moves[self._next_move_index])
        self._next_move_index += 1


class UMError(Exception):
    def __init__(self, msg, errno, oserrno):
        Exception.__init__(self, msg)
        self.errno = errno
        self.oserrno = oserrno


_timer_offset = time.time() - default_timer()


def timer():
    global _timer_offset
    return _timer_offset + default_timer()


class UMP(object):
    """Wrapper for the Sensapex uMp API.
    
    All calls except get_ump are thread-safe.
    """
    _last_move: Dict[int, MoveRequest]

    _lib = None
    _lib_path = None
    _single = None
    _um_state = None

    @classmethod
    def set_library_path(cls, path):
        cls._lib_path = path

    @classmethod
    def get_lib(cls):
        if cls._lib is None:
            cls._lib = cls.load_lib()
            cls._lib.um_get_version.restype = c_char_p
        return cls._lib

    @classmethod
    def load_lib(cls):
        path = os.path.abspath(os.path.dirname(__file__))
        if sys.platform == "win32":
            if cls._lib_path is not None:
                return ctypes.windll.LoadLibrary(os.path.join(cls._lib_path, "umsdk"))

            try:
                return ctypes.windll.umsdk
            except OSError:
                pass

            return ctypes.windll.LoadLibrary(os.path.join(path, "umsdk"))
        else:
            if cls._lib_path is not None:
                return ctypes.cdll.LoadLibrary(os.path.join(cls._lib_path, "libum.so"))

            return ctypes.cdll.LoadLibrary(os.path.join(path, "libum.so"))

    @classmethod
    def get_um_state_class(cls):
        if cls._um_state is None:
            version = cls.get_lib().um_get_version().decode("ascii")
            if version >= "v0.600":
                cls._um_state = um_state_v0_600
            else:
                cls._um_state = um_state_pre_v0_600
        return cls._um_state

    @classmethod
    def get_ump(cls, address=None, group=None, start_poller=True):
        """Return a singleton UM instance.
        """
        # question: can we have multiple UM instances with different address/group ?
        if cls._single is None:
            cls._single = UMP(address=address, group=group, start_poller=start_poller)
        return cls._single

    def __init__(self, address=None, group=None, start_poller=True):
        self.lock = threading.RLock()
        if self._single is not None:
            raise Exception("Won't create another UM object. Use get_ump() instead.")
        self._timeout = 200

        # duration that manipulator must be not busy before a move is considered complete.
        self.move_expire_time = 50e-3

        self._retry_threshold = 0.4
        self.default_max_accelerations = {}

        self.lib = self.get_lib()
        self.lib.um_errorstr.restype = c_char_p

        min_version = (0, 918)
        min_version_str = "v{:d}.{:d}".format(*min_version)
        max_version = (0, 920)
        max_version_str = "v{:d}.{:d}".format(*max_version)
        version_str = self.sdk_version()
        version = tuple(map(int, version_str.lstrip(b"v").split(b".")))

        assert version >= min_version, f"SDK version {min_version_str} or later required (your version is {version_str})"
        assert version <= max_version, f"SDK version {max_version_str} or lower required (your version is {version_str})"

        self.h = None
        self.open(address=address, group=group)

        # keep track of requested moves and whether they completed, failed, or were interrupted.
        self._last_move = {}  # {device: MoveRequest}
        # last time each device was seen moving
        self._last_busy_time = {}

        self._um_has_axis_count = hasattr(self.lib, "um_get_axis_count")
        self._axis_counts = {}

        self.devices = {}

        self.poller = PollThread(self)
        if start_poller:
            self.poller.start()

    def get_device(self, dev_id):
        """

        Returns
        -------
        SensapexDevice
        """
        if dev_id not in self.devices:
            all_devs = self.list_devices()
            if dev_id not in all_devs:
                raise Exception(f"Invalid sensapex device ID {dev_id}. Options are: {all_devs!r}")
            self.devices[dev_id] = SensapexDevice(dev_id)
        return self.devices[dev_id]

    def sdk_version(self):
        """Return version of UM SDK.
        """
        self.lib.um_get_version.restype = ctypes.c_char_p
        return self.lib.um_get_version()

    def list_devices(self, max_id=50):
        """Return a list of all connected device IDs.
        """
        devarray = (c_int * max_id)()
        r = self.call("um_get_device_list", byref(devarray), c_int(max_id))
        devs = [devarray[i] for i in range(r)]

        return devs

    def axis_count(self, dev):
        if not self._um_has_axis_count:
            return 4
        c = self._axis_counts.get(dev, None)
        if c is None:
            c = self.call("um_get_axis_count", dev)
            self.set_axis_count(dev, c)
        return c

    def set_axis_count(self, dev, count):
        self._axis_counts[dev] = count

    def call(self, fn, *args):
        with self.lock:
            if self.h is None:
                raise TypeError("UM is not open.")
            rval = getattr(self.lib, fn)(self.h, *args)
            if rval < 0:
                err = self.lib.um_last_error(self.h)
                errstr = self.lib.um_errorstr(err)
                if err == -1:
                    oserr = self.lib.um_last_os_errno(self.h)
                    raise UMError(f"UM OS Error {oserr:d}: {os.strerror(oserr)}", None, oserr)
                else:
                    raise UMError(f"UM Error {err:d}: {errstr}  From {fn}{args!r}", err, None)
            return rval

    def set_max_acceleration(self, dev, max_acc):
        self.default_max_accelerations[dev] = max_acc

    def open(self, address=None, group=None):
        """Open the UM device at the given address.
        
        The default address "169.254.255.255" should suffice in most situations.
        """
        address = LIBUM_DEF_BCAST_ADDRESS if address is None else address
        group = LIBUM_DEF_GROUP if group is None else group
        if self.h is not None:
            raise TypeError("UM is already open.")
        addr = ctypes.create_string_buffer(address)
        self.lib.um_open.restype = c_longlong
        ptr = self.lib.um_open(addr, c_uint(self._timeout), c_int(group))
        if ptr <= 0:
            raise RuntimeError("Error connecting to UM:", self.lib.um_errorstr(ptr))
        self.h = pointer(self.get_um_state_class().from_address(ptr))
        atexit.register(self.close)

    def close(self):
        """Close the UM device.
        """
        if self.poller.is_alive():
            self.poller.stop()
            self.poller.join()
        with self.lock:
            self.lib.um_close(self.h)
            self.h = None

    def is_positionable(self, dev_id):
        return dev_id != 30

    def get_pos(self, dev, timeout=0):
        """Return the absolute position of the specified device (in um).
        
        If *timeout* == 0, then the position is returned directly from cache
        and not queried from the device.
        """
        if timeout is None:
            timeout = self._timeout
        xyzwe = c_float(), c_float(), c_float(), c_float(), c_int()
        timeout = c_int(timeout)

        self.call("um_get_positions", c_int(dev), timeout, *[byref(x) for x in xyzwe])

        n_axes = self.axis_count(dev)
        return [x.value for x in xyzwe[:n_axes]]

    def goto_pos(self, dev, dest, speed, simultaneous=True, linear=False, max_acceleration=0):
        """Request the specified device to move to an absolute position (in um).

        Parameters
        ----------
        dev : int
            ID of device to move
        dest : array-like
            X,Y,Z,W coordinates to move to. Values may be NaN or omitted to leave
            the axis unaffected.
        speed : float
            Manipulator speed in um/sec
        simultaneous: bool
            If True, then all axes begin moving at the same time
        linear : bool
            If True, then axis speeds are scaled to produce more linear movement, requires simultaneous
        max_acceleration : int
            Maximum acceleration in um/s^2

        Returns
        -------
        move_id : int
            Unique ID that can be used to retrieve the status of this move at a later time.
        """
        next_move = MoveRequest(self, dev, dest, speed, simultaneous, linear, max_acceleration, self._retry_threshold)
        with self.lock:
            last_move = self._last_move.pop(dev, None)
            if last_move is not None:
                last_move.interrupt("started another move before the previous finished")

            self._last_move[dev] = next_move

            next_move.start()

        return next_move

    def is_busy(self, dev):
        """Return True if the specified device is currently moving.

        Note: this should not be used to determine whether a move has completed;
        use MoveRequest.finished or .finished_event as returned from goto_pos().
        """
        # idle/complete=0; moving>0; failed<0
        try:
            return self.call("um_get_drive_status", c_int(dev)) > 0
        except UMError as err:
            if err.errno in (LIBUM_NOT_OPEN, LIBUM_INVALID_DEV):
                raise
            else:
                return False

    def stop(self, dev):
        """Stop the specified manipulator.
        """
        with self.lock:
            self.call("um_stop", c_int(dev))
            move = self._last_move.pop(dev, None)
            if move is not None:
                move.interrupt("stop requested before move finished")

    def set_pressure(self, dev, channel, value):
        return self.call("umc_set_pressure_setting", dev, int(channel), c_float(value))

    def get_pressure(self, dev, channel):
        p = c_float()
        self.call("umc_get_pressure_setting", dev, int(channel), byref(p))
        return p.value

    def measure_pressure(self, dev, channel):
        p = c_float()
        self.call("umc_measure_pressure", dev, int(channel), byref(p))
        return p.value

    def set_valve(self, dev, channel, value):
        return self.call("umc_set_valve", dev, int(channel), int(value))

    def get_valve(self, dev, channel):
        return self.call("umc_get_valve", dev, int(channel))

    def set_custom_slow_speed(self, dev, enabled):
        feature_custom_slow_speed = 32
        return self.call("um_set_ext_feature", c_int(dev), c_int(feature_custom_slow_speed), c_int(enabled))

    def get_custom_slow_speed(self, dev):
        feature_custom_slow_speed = 32
        return self.call("um_get_ext_feature", c_int(dev), c_int(feature_custom_slow_speed))

    def get_um_param(self, dev, param):
        value = c_int()
        self.call("um_get_param", c_int(dev), c_int(param), *[byref(value)])
        return value

    def set_um_param(self, dev, param, value):
        return self.call("um_set_param", c_int(dev), c_int(param), value)

    def calibrate_zero_position(self, dev):
        self.call("um_init_zero", dev, X_AXIS | Y_AXIS | Z_AXIS | D_AXIS)

    def calibrate_load(self, dev):
        self.call("ump_calibrate_load", dev)

    def calibrate_pressure(self, dev, channel, delay):
        self.call("umc_pressure_calib", dev, channel, delay)

    def get_soft_start_state(self, dev):
        feature_soft_start = 33
        return self.call("um_get_ext_feature", c_int(dev), c_int(feature_soft_start))

    def set_soft_start_state(self, dev, enabled):
        feature_soft_start = 33
        return self.call("um_set_ext_feature", c_int(dev), c_int(feature_soft_start), c_int(enabled))

    def get_soft_start_value(self, dev):
        return self.get_um_param(dev, 15)

    def set_soft_start_value(self, dev, value):
        return self.set_um_param(dev, 15, value)

    def set_retry_threshold(self, threshold):
        """
        If we miss any axis by too much, try again.

        Parameters
        ----------
        threshold : float
            Maximum allowable error in µm.
        """
        self._retry_threshold = threshold

    def recv_all(self):
        """Receive all queued position/status update packets and update any pending moves.
        """
        self.call("um_receive", 0)
        self._update_moves()

    def _update_moves(self):
        with self.lock:
            for dev, move in list(self._last_move.items()):
                if move.is_in_progress():
                    continue
                if move.has_more_calls_to_make():
                    move.make_next_call()
                elif move.can_retry() and not move.is_close_enough():
                    move.retry()
                else:
                    self._last_move.pop(dev)
                    move.finish()


class SensapexDevice(object):
    """UM wrapper for accessing a single sensapex device.

    Example:
    
        dev = SensapexDevice(1)  # get handle to manipulator 1
        pos = dev.get_pos()
        pos[0] += 10000  # add 10 um to x axis 
        dev.goto_pos(pos, speed=10)
    """

    def __init__(self, dev_id: int, callback=None, n_axes=None, max_acceleration=0):
        self.dev_id = dev_id
        self.ump = UMP.get_ump()

        # Save max acceleration from config
        if max_acceleration is None:
            max_acceleration = 0
        self.set_max_acceleration(max_acceleration)

        # some devices will fail when asked how many axes they have; this
        # allows a manual override.
        if n_axes is not None:
            self.set_n_axes(n_axes)

        self.ump.poller.add_callback(dev_id, self._change_callback)
        self.callbacks = []

        if callback is not None:
            self.add_callback(callback)

    def set_n_axes(self, n_axes):
        self.ump.set_axis_count(self.dev_id, n_axes)

    def set_max_acceleration(self, max_acceleration):
        self.ump.set_max_acceleration(self.dev_id, max_acceleration)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def get_pos(self, timeout=None):
        return self.ump.get_pos(self.dev_id, timeout=timeout)

    def goto_pos(self, pos, speed, simultaneous=True, linear=False, max_acceleration=0):
        return self.ump.goto_pos(
            self.dev_id, pos, speed, simultaneous=simultaneous, linear=linear, max_acceleration=max_acceleration
        )

    def is_busy(self):
        return self.ump.is_busy(self.dev_id)

    def stop(self):
        return self.ump.stop(self.dev_id)

    def _change_callback(self, dev_id, new_pos, old_pos):
        for cb in self.callbacks:
            cb(self, new_pos, old_pos)

    def set_pressure(self, channel, value):
        """
        Parameters
        ----------
        channel : int
            channel number
        value : float
            pressure in kPa
        """
        return self.ump.set_pressure(self.dev_id, int(channel), float(value))

    def get_pressure(self, channel):
        """
        Returns
        -------
        float
            expected pressure in kPa
        """
        return self.ump.get_pressure(self.dev_id, int(channel))

    def measure_pressure(self, channel):
        """
        Returns
        -------
        float
            actual pressure in kPa
        """
        return self.ump.measure_pressure(self.dev_id, int(channel))

    def set_valve(self, channel, value):
        return self.ump.set_valve(self.dev_id, int(channel), int(value))

    def get_valve(self, channel):
        return self.ump.get_valve(self.dev_id, int(channel))

    def set_lens_position(self, pos):
        return self.ump.call("ums_set_lens_position", c_int(self.dev_id), c_int(pos))

    def get_lens_position(self):
        return self.ump.call("ums_get_lens_position", c_int(self.dev_id))

    def set_custom_slow_speed(self, enabled):
        return self.ump.set_custom_slow_speed(self.dev_id, enabled)

    def calibrate_zero_position(self):
        self.ump.calibrate_zero_position(self.dev_id)

    def calibrate_load(self):
        self.ump.calibrate_load(self.dev_id)

    def calibrate_pressure(self, channel, delay=0):
        self.ump.calibrate_pressure(self.dev_id, channel, delay)

    def get_soft_start_state(self):
        return self.ump.get_soft_start_state(self.dev_id)

    def set_soft_start_state(self, enabled):
        return self.ump.set_soft_start_state(self.dev_id, enabled)

    def get_soft_start_value(self):
        return self.ump.get_soft_start_value(self.dev_id).value

    def set_soft_start_value(self, value):
        return self.ump.set_soft_start_value(self.dev_id, value)


class PollThread(threading.Thread):
    """Thread to poll for all manipulator position changes.

    Running this thread ensures that calling get_pos will always return the most recent
    values available.

    An optional callback function is called periodically with a list of
    device IDs from which position updates have been received.
    """

    def __init__(self, ump, interval=0.03):
        self.ump = ump
        self.callbacks = {}
        self.interval = interval
        self.lock = threading.RLock()
        self.__stop = False
        threading.Thread.__init__(self)
        self.daemon = True

    def start(self):
        self.__stop = False
        threading.Thread.start(self)

    def stop(self):
        self.__stop = True

    def add_callback(self, dev_id, callback):
        with self.lock:
            self.callbacks.setdefault(dev_id, []).append(callback)

    def remove_callback(self, dev_id, callback):
        with self.lock:
            self.callbacks[dev_id].remove(callback)

    def run(self):
        ump = self.ump
        last_pos = {}

        while True:
            try:
                if self.__stop:
                    break

                # read all updates waiting in queue
                ump.recv_all()

                # check for position changes and invoke callbacks
                with self.lock:
                    callbacks = self.callbacks.copy()

                for dev_id, dev_callbacks in callbacks.items():
                    if len(callbacks) == 0:
                        continue
                    if ump.is_positionable(dev_id):
                        new_pos = ump.get_pos(dev_id, timeout=0)
                        old_pos = last_pos.get(dev_id)
                        changed = new_pos != old_pos
                    else:
                        # TODO what do pressure devices need here?
                        changed = False
                        new_pos = None
                        old_pos = None

                    if changed:
                        for cb in dev_callbacks:
                            cb(dev_id, new_pos, old_pos)

                time.sleep(self.interval)

            except Exception:
                print("Error in sensapex poll thread:")
                sys.excepthook(*sys.exc_info())
                time.sleep(1)
