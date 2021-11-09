import atexit
import faulthandler
import time
from contextlib import contextmanager
from ctypes import (
    byref,
    c_uint32,
    c_uint8,
    POINTER,
    Structure,
    c_int,
    c_uint16,
    c_bool,
    sizeof,
    c_float,
)
from threading import Thread
from typing import Union, Iterable, Dict, List, Any, Tuple

import numpy as np
from typing_extensions import Literal

import pyqtgraph as pg
from sensapex import SensapexDevice, UMError
from sensapex.sensapex import um_state, UMP, LIBUM_MAX_MANIPULATORS, LIBUM_TIMEOUT

faulthandler.enable()

LIBUM_MAX_DEVS = 0xFFFF
UMA_REG_COUNT = 10


class _uma_state_struct(Structure):
    _fields_ = [
        ("um_ctx", POINTER(um_state)),
        ("um_dev", c_int),
        ("wait_trig", c_bool),
        ("reg_values", c_uint32 * UMA_REG_COUNT),
    ]


class _uma_capture_struct(Structure):
    _fields_ = [
        ("status", c_uint8),
        ("flags", c_uint8),
        ("index", c_uint16),
        ("ts", c_uint32),
        ("current", c_uint16),
        ("voltage", c_uint16),
    ]


class UMA(object):
    """Class representing a uMa device ( itself attached to a uMp )"""

    _recv_handlers: List[Tuple[Union[str, None], Any, Any]]

    UMA_CAPTURES_PER_PACKET = 1440 // sizeof(_uma_capture_struct)

    PARAMETERS = {
        "ic_bridge_enabled": {
            "initial_value": False,
        },
        "ic_bridge_gain": {
            "initial_value": 0,
        },
        "ic_cfast_enabled": {
            "initial_value": False,
        },
        "ic_cfast_gain": {
            "initial_value": 0,
        },
        "clamp_mode": {
            "initial_value": "VC",
        },
        "current_input_range": {
            "initial_value": 200e-12,
        },
        "current_output_range": {
            "initial_value": 150e-9,  # TODO test that this is correct
        },
        "holding_current": {
            "initial_value": 0,
        },
        "holding_voltage": {
            "initial_value": 0,
        },
        "sample_rate": {
            "initial_value": 9776,
        },
        "trig_bit": {
            "initial_value": None,  # todo what's this  # todo what is this really?
        },
        "vc_cfast_enabled": {
            "initial_value": False,
        },
        "vc_cfast_gain": {
            "initial_value": 0,
        },
        "vc_cslow_enabled": {
            "initial_value": False,
        },
        "vc_cslow_gain": {
            "initial_value": 0,
        },
        "vc_cslow_tau": {
            "initial_value": 0,
        },
        "vc_serial_resistance_enabled": {
            "initial_value": False,
        },
        "vc_serial_resistance_gain": {
            "initial_value": 0,
        },
        "vc_serial_resistance_lag_filter": {
            "initial_value": False,
        },
        "vc_serial_resistance_prediction_rise_factor": {
            "initial_value": 2,
        },
        "vc_serial_resistance_tau": {
            "initial_value": 0,
        },
        "vc_voltage_offset": {
            "initial_value": None,  # todo this isn't initialized, so it's probably 0x0, which translates to -50mV?
        },
        "receive_waits_for_trigger": {  # todo how do we best support this
            "initial_value": None,  # todo what is this really?
        },
        "zap": {
            "initial_value": False,  # todo is zap really part of this
        },
    }

    def __init__(self, uMp: SensapexDevice):
        # TODO should we enforce only making one of these per device
        self._recv_handlers = []
        self._state = _uma_state_struct()
        self._lock = uMp.ump.lock
        self._libuma = uMp.ump.libuma
        self.sensapex = uMp.ump
        self.call("init", uMp.ump.h, c_int(uMp.dev_id))
        self._param_cache = {name: conf["initial_value"] for name, conf in self.PARAMETERS.items()}
        # TODO how do we track the following initial states: no reset ( and what even is it? )

        self._run_recv_thread = True
        self._pause_recv_thread = True
        self._recv_thread = Thread(target=self._do_recv_forever, daemon=True)
        self._recv_thread.start()
        self._recv_buffer = (_uma_capture_struct * self.UMA_CAPTURES_PER_PACKET)()
        self._np_recv_buffer = np.frombuffer(
            self._recv_buffer,
            dtype=[
                ("status", c_uint8),
                ("flags", c_uint8),
                ("index", c_uint16),
                ("ts", c_uint32),
                ("current", c_uint16),
                ("voltage", c_uint16),
            ],
        )

    def call(self, fn_name: str, *args) -> int:
        """
        Make a raw call to a uMa function in the C API.

        Parameters
        ----------
        fn_name : str
            A valid function name in the libuma library, without the "uma_" prefix.
        args
            The arguments to the function, not including the `uma_state`. These must already be cast as ctypes.

        Returns
        -------
        int
            The C API's return value, which should always be greater than or equal to zero.

        Raises
        ------
        UMError
            If the return value is less than zero, this will raise an error with as much information as we know about
            the failure.
        """
        fn_name = f"uma_{fn_name}"
        return self.sensapex.call_lib_fn(self._libuma, fn_name, byref(self._state), *args)

    def send_stimulus_raw(self, stimulus: np.ndarray, trigger_sync: bool = False) -> None:
        """
        TODO

        Parameters
        ----------
        stimulus : ndarray
            Stimulus data, which will be cast to 32-bit integers. The scaling depends on the clamp mode.
            For VC, only the bottom 9 bits are usable, not including sign. Values are scaled to ±700mV.
            For IC, only the bottom 17 bits are usable, not including sign. Values are scaled to ±`current_input_range`.
        trigger_sync : bool
            TODO what does this end up doing?
        """
        if len(stimulus.shape) != 1:
            raise ValueError(f"Stimulus may only be 1D. Received {stimulus.shape}.")
        if stimulus.dtype != int:
            raise ValueError("Raw stimulus must be sent as integers.")
        too_big = 2 ** 9 if self.get_clamp_mode() == "VC" else 2 ** 17
        biggest_val = np.max(np.abs(stimulus))
        print(f"largest stim val {biggest_val}")
        if biggest_val > too_big:
            # TODO is this too expensive to check?
            raise ValueError(f"Stimulus values may not exceed ±{too_big}")
        stim_len = stimulus.shape[0]
        stimulus = stimulus.astype(c_int)
        # Sensapex segfaults on more than 749 points, so chunk it
        for stim_chunk in np.array_split(stimulus, range(min(stim_len, 749), stim_len, 749)):
            c_stim = np.ctypeslib.as_ctypes(stim_chunk)
            self.call("stimulus", c_int(stim_chunk.shape[0]), c_stim, c_bool(trigger_sync))

    def send_stimulus_scaled(self, stimulus: np.ndarray, trigger_sync: bool = False, scale=1):
        scale = self._adjust_scale_for_input(scale)
        print(f"scaled stim at scale {scale} with max value {np.max(np.abs(stimulus))}")
        self.send_stimulus_raw((stimulus / scale).astype(int), trigger_sync)

    def _adjust_scale_for_input(self, scale=1, as_mode=None):
        if (as_mode or self.get_clamp_mode()) == "VC":
            return scale * 0.7 / 2 ** 9
        else:  # IC
            return scale * self.get_current_input_range() / 2 ** 17

    def get_clamp_mode(self) -> str:
        return self.get_param("clamp_mode")

    def set_clamp_mode(self, mode: Union[Literal["VC"], Literal["IC"]]) -> None:
        """
        Set the clamping mode.

        This will disable all incompatible compensation circuits, and re-enable any compensations that were previously
        enabled.

        Parameters
        ----------
        mode
            One of "IC" for current clamp, or "VC" for voltage clamp.
        """
        if mode == self.get_clamp_mode():
            return
        self._param_cache["clamp_mode"] = mode
        with self.pause_receiving():
            if mode == "IC":
                self._enable_ic_mode()
            elif mode == "VC":
                self._enable_vc_mode()
            else:
                raise ValueError(f"'{mode}' is not a valid clamp mode. Only 'VC' and 'IC' are accepted.")

    def _enable_ic_mode(self):
        self.set_vc_cfast(enabled=False, _remember_enabled=False)
        self.set_vc_cslow(enabled=False, _remember_enabled=False)
        self.set_vc_serial_resistance(enabled=False, _remember_enabled=False)
        self.call("set_current_clamp_mode", c_bool(True))
        self.set_ic_cfast(enabled=self.get_param("ic_cfast_enabled"))
        self.set_ic_bridge(enabled=self.get_param("ic_bridge_enabled"))

    def _enable_vc_mode(self):
        self.set_ic_cfast(enabled=False, _remember_enabled=False)
        self.set_ic_bridge(enabled=False, _remember_enabled=False)
        self.call("set_current_clamp_mode", c_bool(False))
        self.set_vc_cslow(enabled=self.get_param("vc_cslow_enabled"))
        self.set_vc_serial_resistance(enabled=self.get_param("vc_serial_resistance_enabled"))
        self.set_vc_cfast(enabled=self.get_param("vc_cfast_enabled"))

    VALID_SAMPLE_RATES = (1221, 4883, 9776, 19531, 50000, 100000, 200000)

    def set_sample_rate(self, rate: int) -> None:
        if rate not in self.VALID_SAMPLE_RATES:
            raise ValueError(f"'{rate}' is an invalid sample rate. Choose from {self.VALID_SAMPLE_RATES}.")
        with self.pause_receiving():
            self._param_cache["sample_rate"] = rate
            self.call("set_sample_rate", c_int(rate))

    VALID_CURRENT_INPUT_RANGES = (200e-12, 2000e-12, 20000e-12, 200000e-12)

    def set_current_input_range(self, current_range: float) -> None:
        """
        Sets the input range of current values. ( Note that no similar method exists for the voltage range, which is
        always ±700mV. )

        Parameters
        ----------
        current_range : float
            One of the VALID_CURRENT_RANGES (200pA, 2nA, 20nA or 200nA). All current input will then be scaled to ± this
            value.
        """
        if current_range not in self.VALID_CURRENT_INPUT_RANGES:
            raise ValueError(
                f"'{current_range}' is not a valid current range. Choose from {self.VALID_CURRENT_INPUT_RANGES}."
            )
        with self.pause_receiving():
            self._param_cache["current_input_range"] = current_range
            self.call("set_range", c_int(int(current_range / 1e-12)))  # Convert to int pA first

    def get_current_input_range(self) -> float:
        return self._param_cache["current_input_range"]

    VALID_CURRENT_OUTPUT_RANGES = (3.75e-9, 150e-9)

    def set_current_output_range(self, current_range: float):
        if current_range not in self.VALID_CURRENT_OUTPUT_RANGES:
            raise ValueError(
                f"'{current_range}' is not a valid current range. Choose from {self.VALID_CURRENT_OUTPUT_RANGES}."
            )
        self._param_cache["current_output_range"] = current_range
        # TODO test that this number is working correctly (the header file's boolean and values are seemingly switched)
        self.call("enable_cc_higher_range", c_bool(current_range == 3.75e-9))

    def get_current_output_range(self) -> float:
        return self._param_cache["current_output_range"]

    def zap(self, enabled: bool = True, duration: float = None):
        # TODO test this. understand zap. thread the sleep.
        with self.pause_receiving():
            self.call("set_zap", c_bool(enabled))
        if duration is not None:
            time.sleep(duration)
            with self.pause_receiving():
                self.call("set_zap", c_bool(False))  # TODO or `not enabled`?

    def set_vc_cslow(self, enabled: bool = None, gain: float = None, tau: float = None, _remember_enabled=True) -> None:
        """Set the C-slow compensation circuit, only available in voltage-clamp mode.

        Parameters
        ----------
        enabled : bool
        gain : float
            Gain value in farads, between 0-255 pF, with a 1 pF precision.
        tau: float
            Tau value in s, between 0-2542.5 µs, with a 9.97 µs precision.
        _remember_enabled
            Internal use.
        """
        # TODO test this
        if enabled and self.get_clamp_mode() != "VC":
            raise ValueError("C-slow compensation cannot be enabled in IC mode")
        if gain is not None:
            self._param_cache["vc_cslow_gain"] = gain
        if tau is not None:
            self._param_cache["vc_cslow_tau"] = tau
        if _remember_enabled and enabled is not None:
            self._param_cache["vc_cslow_enabled"] = enabled
        if enabled:
            self.call("set_vc_cslow_gain", c_float(self.get_param("vc_cslow_gain")))
            self.call("set_vc_cslow_tau", c_float(self._param_cache["vc_cslow_tau"]))
        elif enabled is not None:
            self.call("set_vc_cslow_gain", c_float(0.0))  # it's enough to just set either to 0

    def set_vc_cfast(self, enabled: bool = None, gain: float = None, _remember_enabled=True) -> None:
        """Set the C-fast compensation circuit for voltage-clamp mode.

        Parameters
        ----------
        enabled : bool
        gain : float
            Gain in farads, between 0-10.875 pF, with a 0.75 pF precision that starts at 0.375 pF. I.e. [0, 0.375e-12),
            [0.375e-12, 1.125e-12), [1.125e-12, 1.875e-12), ...
        _remember_enabled
            Internal use.
        """
        # TODO test this.
        if enabled and self.get_clamp_mode() != "IC":
            raise ValueError("cfast compensation cannot be enabled in VC mode")
        if gain is not None:
            if gain < 0 or gain > 10.875e-12:
                raise ValueError("C-fast gain must be between 0-10.875 pF.")
            self._param_cache["vc_cfast_gain"] = gain
        if _remember_enabled and enabled is not None:
            self._param_cache["vc_cfast_enabled"] = enabled
        if enabled:
            self.call("set_vc_cfast_gain", c_float(self.get_param("vc_cfast_gain") * 1e12))
        elif enabled is not None:
            self.call("set_vc_cfast_gain", c_float(0.0))

    def set_ic_cfast(self, enabled: bool = None, gain: float = None, _remember_enabled=True) -> None:
        """Set C-fast compensation circuit for current-clamp mode.

        Parameters
        ----------
        enabled : bool
        gain : float
            Gain in farads, between 0-31.875 pF, with a 0.125 pF precision.
        _remember_enabled
            Internal use
        """
        # TODO test
        if enabled and self.get_clamp_mode() != "IC":
            raise ValueError("cfast compensation cannot be enabled in VC mode")
        if _remember_enabled and enabled is not None:
            self._param_cache["ic_cfast_enabled"] = enabled
        if gain is not None:
            if gain < 0 or gain > 31.875e-12:
                raise ValueError("IC C-fast gain must be between 0-32 pF.")
            self._param_cache["ic_cfast_gain"] = gain
        if enabled:
            self.call("set_cc_cfast_gain", c_int(int(self.get_param("ic_cfast_gain") * 1e12)))
        elif enabled is not None:
            self.call("set_cc_cfast_gain", c_int(0))

    def set_vc_serial_resistance(
        self,
        enabled: bool = None,
        gain: float = None,
        tau: float = None,
        prediction_rise_factor: Union[Literal[2], Literal[3]] = None,
        lag_filter: bool = None,
        _remember_enabled=True,
    ) -> None:
        """Set the serial resistance compensation circuit in voltage-clamp mode.

        Parameters
        ----------
        enabled : bool
        gain : float
            Gain value in Ω, between 0-25.3125 MΩ, with a 99.3 kΩ precision.
        tau : float
            Tau value in seconds, between 0-768 µs, with a 3 µs precision.
        prediction_rise_factor : int
            2 or 3.
        lag_filter : bool
        _remember_enabled
            Internal use.
        """
        if enabled and self.get_clamp_mode() != "VC":
            raise ValueError("Serial resistance compensation cannot be enabled in IC mode")
        if enabled and self.get_current_input_range() == 200e-12:
            raise ValueError("Serial resistance does not function at ±200e-12 input range")
        if gain is not None:
            self._param_cache["vc_serial_resistance_gain"] = gain
        if prediction_rise_factor is not None:
            self._param_cache["vc_serial_resistance_prediction_rise_factor"] = prediction_rise_factor
        if tau is not None:
            self._param_cache["vc_serial_resistance_tau"] = tau
        if lag_filter is not None:
            self._param_cache["vc_serial_resistance_lag_filter"] = lag_filter
        if _remember_enabled and enabled is not None:
            self._param_cache["vc_serial_resistance_enabled"] = enabled

        if enabled:
            enable_3x_gain = c_bool(self._param_cache["vc_serial_resistance_prediction_rise_factor"] == 3)
            self.call("set_vc_rs_pred_3x_gain", enable_3x_gain)
            self.call("set_vc_rs_fast_lag_filter", c_bool(self._param_cache["vc_serial_resistance_lag_filter"]))
            self.call("set_vc_rs_corr_tau", c_float(self.get_param("vc_serial_resistance_tau") * 1e6))
            self.call("set_vc_rs_corr_gain", c_float(self.get_param("vc_serial_resistance_gain") / 1e6))
        elif enabled is not None:
            self.call("set_vc_rs_corr_gain", c_float(0))
        # TODO testing

    def set_ic_bridge(self, enabled: bool = None, gain: int = None, _remember_enabled=True):
        """In current-clamp mode, set the bridge compensation circuit.

        Parameters
        ----------
        enabled : bool
        gain : int
            Gain in Ω, between 0-40 MΩ, with a 0.16 MΩ precision.
        _remember_enabled
            Internal use.
        """
        if enabled and self.get_clamp_mode() != "IC":
            raise ValueError("bridge compensation cannot be enabled in VC mode")
        if gain is not None:
            if gain < 0 or gain > 40e6:
                raise ValueError("Bridge gain must be between 0-40 MΩ")
            self._param_cache["ic_bridge_gain"] = gain
        if _remember_enabled and enabled is not None:
            self._param_cache["ic_bridge_enabled"] = enabled
        if enabled:
            self.call("set_cc_bridge_gain", c_int(self.get_param("ic_bridge_gain") // 1e6))
        elif enabled is not None:
            self.call("set_cc_bridge_gain", c_int(0))
        # TODO test

    def _do_recv_forever(self):
        while self._run_recv_thread:
            if self._pause_recv_thread:
                time.sleep(1)
                continue
            try:
                # TODO race? what if this and stop are called at the same time?
                self.call("recv", self.UMA_CAPTURES_PER_PACKET, self._recv_buffer)
            except UMError as e:
                if e.errno == LIBUM_TIMEOUT:
                    print("timed out")  # TODO
                else:
                    raise e
            else:
                float_cast_data = {}
                for column, handler, scale in self._recv_handlers:
                    if column is None:
                        handler(self._np_recv_buffer)
                    elif scale is None:
                        handler(self._np_recv_buffer[column])
                    else:
                        if column not in float_cast_data:
                            float_cast_data[column] = self._np_recv_buffer[column].astype(float)
                        handler(float_cast_data[column] * scale)  # 0.7 / 2**9

    def start_receiving(self):
        """
        Turn on the data receiving. To access the data received, pass a handler into either of the
        `add_receive_data_handler_*` methods.
        """
        self.call("start")
        self._pause_recv_thread = False

    def stop_receiving(self):
        self._pause_recv_thread = True
        self.call("stop")

    def add_receive_data_handler_raw(self, handler, column: str = None):
        """
            TODO mention scaled methods, removal

        Parameters
        ----------
        handler
            This callable should accept a single ndarray argument with a data type that mirrors `_uma_capture_struct` or
            one of its columns (if the `column` parameter is set). Data domains and ranges of this object will vary.
            "voltage" will be `±2**9 -> ±700mV`. "current" will be `±2**17 -> ±self.current_range`. "ts" (which stands
            for "timestamp") will be `µs`.
        column : str
            If set, this specifies that only one column of the data will be passed to the handler (e.g. "voltage").
        """
        valid_columns = self._np_recv_buffer.dtype.names
        if column is not None and column not in valid_columns:
            raise ValueError(f"'{column}' is not a valid column name for captured data. Choose from {valid_columns}")
        with self._lock:
            self._recv_handlers.append((column, handler, None))

    def add_receive_data_handler_scaled(self, handler, column: str, scale=1):
        """TODO"""
        if column == "ts":
            scale *= 1e-6
        elif column == "voltage":
            scale *= 0.7 / (2 ** 15)
        elif column == "current":
            scale *= self.get_current_output_range() / (2 ** 15)
        with self._lock:
            self._recv_handlers.append((column, handler, scale))

    def remove_receive_data_handler(self, handler, column=None):
        """ TODO """
        with self._lock:
            self._recv_handlers = [(c, h, s) for c, h, s in self._recv_handlers if h != handler and c != column]

    def quit(self):
        """TODO"""
        self.stop_receiving()
        self._run_recv_thread = False
        self._recv_thread.join()

    @contextmanager
    def pause_receiving(self):
        """TODO"""
        was_receiving = self.is_receiving()
        self.stop_receiving()
        yield
        if was_receiving:
            self.start_receiving()

    def is_receiving(self):
        return self._run_recv_thread and not self._pause_recv_thread

    def get_holding_current(self):
        return self.get_param("holding_current")

    def get_holding_voltage(self):
        return self.get_param("holding_voltage")

    def set_holding_current(self, hold_at: float):
        """Set the holding current in current-clamp mode.

        Parameters
        ----------
        hold_at : float
            The holding current in amps, between ±``get_current_input_range()``, with 17 bits of precision.
        """
        # TODO test
        if abs(hold_at) > self.get_current_input_range():
            raise ValueError(f"Requested holding current of {hold_at} is outside the current input range of"
                             f" ±{self.get_current_input_range()}")
        self._param_cache["holding_current"] = hold_at
        scaled_current = self._param_cache["holding_current"] / self._adjust_scale_for_input(as_mode="IC")
        with self.pause_receiving():
            self.call("set_cc_dac", c_int(int(scaled_current)))
        # todo test that can I set this safely when in the wrong mode

    def set_holding_voltage(self, hold_at: float):
        """Set the holding voltage in voltage-clamp mode.

        Parameters
        ----------
        hold_at : float
            The holding voltage in volts, between ±700mV, with 1.37 mV precision.
        """
        # TODO test
        if abs(hold_at) > 0.7:
            raise ValueError(f"Requested holding voltage of {hold_at} is outside the voltage input range of ±0.7")
        self._param_cache["holding_voltage"] = hold_at
        scaled_voltage = self._param_cache["holding_voltage"] / self._adjust_scale_for_input(as_mode="VC")
        with self.pause_receiving():
            self.call("set_vc_dac", c_int(int(scaled_voltage)))
        # todo test that can I set this safely when in the wrong mode

    def get_param(self, name):
        # The sdk doesn't provide this feature
        if name not in self._param_cache:
            raise ValueError(f"{name} is not a valid parameter name. Choose from {set(self._param_cache.keys())}")
        return self._param_cache[name]

    def get_params(self, param_names: Iterable[str] = None) -> Dict:
        if param_names is None:
            param_names = self._param_cache.keys()
        return {name: self._param_cache[name] for name in param_names}

    def set_param(self, param, value):
        pass  # TODO

    def set_vc_voltage_offset(self, offset: float):
        """

        Parameters
        ----------
        offset : float
            Voltage offset, between ±50mV, with a 97µV precision.
        """
        # TODO test
        with self.pause_receiving():
            self.call("set_vc_voltage_offset", c_float(offset * 1e3))


if __name__ == "__main__":
    UMP.set_debug_mode(True)
    um = UMP.get_ump()

    # # LIBUM_SHARED_EXPORT int um_set_feature(um_state *hndl, const int dev, const int id, const int value);
    # um.call("um_set_feature", 1, 13, 1)
    # print(f"feature 13: {um.call('um_get_feature', 1, 13)}")
    # sleep(5)
    # print(f"feature 13: {um.call('um_get_feature', 1, 13)}")

    print(um.list_devices())
    dev1 = um.get_device(1)
    uma = UMA(dev1)

    # dev_array = (c_int * LIBUM_MAX_DEVS)()
    # # _uma_lib.uma_get_device_list.restype = c_int
    # # init_ret = _uma_lib.uma_init", handle, c_int(1))
    # n_devs = uma.call("get_device_list", byref(dev_array), c_int(LIBUM_MAX_MANIPULATORS))
    # if n_devs >= 0:
    #     print([dev_array[i] for i in range(n_devs)])

    range_of_current = 20000e-12
    read_sample_rate = 9776

    uma.set_current_input_range(range_of_current)
    uma.set_current_output_range(150e-9)
    uma.set_sample_rate(read_sample_rate)

    trig = True
    uma.call("set_trig_bit", c_bool(trig))  # TODO what does this do?
    uma.call("set_wait_trig", c_bool(False))

    uma.set_clamp_mode("VC")
    uma.set_vc_voltage_offset(0)
    uma.set_vc_cfast(False)
    uma.set_vc_cslow(False)
    uma.set_vc_serial_resistance(False)

    # Stimulus setup
    N_SAMPLES = 1000
    np_stim_raw = np.zeros((N_SAMPLES,), dtype=int)
    np_stim_scaled = np.zeros((N_SAMPLES,), dtype=float)

    w = pg.GraphicsLayoutWidget()
    p1 = w.addPlot(row=0, col=0)
    p2 = w.addPlot(row=1, col=0)
    p3 = w.addPlot(row=2, col=0)
    p1.setLabels(left=("current", "A"))
    p2.setLabels(left=("voltage", "V"))
    p2.setXLink(p1)
    p3.setLabels(bottom=("time", "s"), left="status")
    p3.setXLink(p1)

    w.show()

    graph_data = np.zeros(
        int(5 * read_sample_rate), dtype=[("ts", float), ("current", float), ("voltage", float), ("status", int)]
    )
    t_offset = None

    def handle_raw_recv(received_data):
        global graph_data, t_offset
        t_offset = received_data["ts"][0] if t_offset is None else t_offset
        graph_data = np.roll(graph_data, -len(received_data))

        graph_data[-len(received_data) :]["ts"] = (received_data["ts"] - t_offset) * 1e-6
        # ±range_of_current pA w.r.t. ground/common
        # read_data[-len(np_buff):]["current"] = (1e-12 * range_of_current / (2 ** 15)) * np_buff["current"]
        graph_data[-len(received_data) :]["current"] = (range_of_current / (2 ** 15)) * (
            received_data["current"].astype(int) - 2 ** 15
        )
        # ±700 mV w.r.t. ground/common
        # read_data[-len(np_buff):]["voltage"] = np_buff["voltage"] * (0.7 / 2 ** 15)
        graph_data[-len(received_data) :]["voltage"] = (0.7 / 2 ** 15) * (
            received_data["voltage"].astype(int) - 2 ** 15
        )
        graph_data[-len(received_data) :]["status"] = received_data["status"]

    def handle_scaled_ts_recv(received_data):
        global graph_data, t_offset
        t_offset = received_data[0] if t_offset is None else t_offset
        graph_data = np.roll(graph_data, -len(received_data))

        graph_data[-len(received_data) :]["ts"] = received_data - t_offset

    def handle_scaled_current_recv(received_data):
        graph_data[-len(received_data) :]["current"] = received_data

    def handle_scaled_voltage_recv(received_data):
        graph_data[-len(received_data) :]["voltage"] = received_data

    def handle_scaled_status_recv(received_data):
        graph_data[-len(received_data) :]["status"] = received_data

    # uma.add_receive_data_handler_raw(handle_raw_recv)
    uma.add_receive_data_handler_scaled(handle_scaled_ts_recv, "ts")
    uma.add_receive_data_handler_scaled(handle_scaled_current_recv, "current")
    uma.add_receive_data_handler_scaled(handle_scaled_voltage_recv, "voltage")
    uma.add_receive_data_handler_scaled(handle_scaled_status_recv, "status")
    uma.start_receiving()

    def update_plots():
        t = graph_data["ts"]
        p1.plot(t, graph_data["current"], clear=True)
        p2.plot(t, graph_data["voltage"], clear=True)
        p3.plot(t, graph_data["status"], clear=True)

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update_plots)
    timer.start(10)

    def insert_stim(raw=None, scaled=None):
        global np_stim_raw
        if raw is None and scaled is None:
            raw = 30
        if raw is not None:
            np_stim_raw[0:-20] = raw
            uma.send_stimulus_raw(np_stim_raw, trig)
        else:
            np_stim_scaled[0:-20] = scaled
            uma.send_stimulus_scaled(np_stim_scaled)

    def stim_train():
        def _do_stims():
            if uma.get_clamp_mode() == "IC":
                max_stim = uma.get_current_input_range()
                steps = 17
            else:
                max_stim = 0.7
                steps = 9
            for i in range(steps):
                insert_stim(scaled=max_stim / (2 ** i))
            time.sleep(0.1)
            for i in range(steps):
                insert_stim(scaled=-max_stim / (2 ** i))
                time.sleep(0.1)

        t = Thread(target=_do_stims, daemon=True)
        t.start()

    stim_timer = pg.QtCore.QTimer()
    # stim_timer.timeout.connect()
    # stim_timer.start(1000)

    def cleanup():
        timer.stop()
        stim_timer.stop()
        uma.quit()

    atexit.register(cleanup)

    def pause():
        time.sleep(0.1)
        uma.stop_receiving()

    def unpause():
        uma.start_receiving()

    def test_stim():
        def _do_test():
            global timer, stim_timer, graph_data
            start_ts = graph_data["ts"][-1]
            print(f"latest timestamp on a sample at start stim {start_ts}")
            insert_stim(2 ** 9)
            print("sleeping")
            time.sleep(2)
            local_read_data = graph_data.copy()
            try:
                trig_index = np.argwhere(local_read_data["current"] > 20e-12)[0, 0]
                trig_time = local_read_data[trig_index]["ts"]
                print(f"Stim detected at t={trig_time}")
                print(f"time difference: {trig_time - start_ts}")
            except IndexError:
                print("Could not detect a current > 20e-12")

        Thread(target=_do_test, daemon=True).start()
