import atexit

import faulthandler

import time
from contextlib import contextmanager
from ctypes import (
    byref,
    c_uint32,
    c_uint8,
    POINTER,
    c_int16,
    Structure,
    c_int,
    c_uint16,
    c_bool,
    sizeof,
    c_float,
)
from threading import Thread
from typing import Union

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

    UMA_CAPTURES_PER_PACKET = 1440 // sizeof(_uma_capture_struct)

    def __init__(self, sensapex_connection: UMP, uMp: SensapexDevice):
        self._recv_handlers_raw = []
        self._state = _uma_state_struct()
        self._lock = sensapex_connection.lock
        self._libuma = sensapex_connection.libuma
        self.sensapex = sensapex_connection
        self.call("init", sensapex_connection.h, c_int(uMp.dev_id))
        # `init` sets the following:
        self._sample_rate = 9776
        self._current_input_range = 200e-12
        self._clamp_mode = "VC"
        self._compensations = {
            "cslow": {
                "enabled": False,
                "gain": None,  # TODO what is this actually?
                "tau": None,  # TODO what is this actually?
            },
            "serial_resistance": {
                "enabled": False,
                "gain": None,  # TODO what is this actually?
                "tau": None,  # TODO what is this actually?
                "range": None,  # TODO what is this actually?
                "lag_filter": None,  # TODO what is this actually?
            },
            "cfast": {"enabled": False, "gain": None,},  # TODO what is this actually?
            "bridge": {"enabled": False, "gain": None,},  # TODO what is this actually?
        }
        # TODO how do we track the following initial states: no reset, no RUN, no ZAP
        # TODO wouldn't it be better to ask for these values from the sdk?

        self._voltage_range = 700e-3
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
            Up to 749 points of stimulus, which will be cast to 32-bit integers. The scaling depends on the clamp mode.
            For VC, only the bottom 9 bits are usable, not including sign. Values are scaled to ±700mV.
            For IC, only the bottom 17 bits are usable, not including sign. Values are scaled to ±`current_input_range`.
        trigger_sync : bool
            TODO what does this end up doing?
        """
        if len(stimulus.shape) != 1:
            raise ValueError(f"Stimulus may only be 1D. Received {stimulus.shape}.")
        stim_len = stimulus.shape[0]
        if stim_len > 749:
            raise ValueError(f"Stimulus must have 749 or fewer points. Received {stim_len}.")
        # TODO check for clipping?
        stimulus = stimulus.astype(c_int)
        c_stimulus = np.ctypeslib.as_ctypes(stimulus)
        self.call("stimulus", c_int(stim_len), c_stimulus, c_bool(trigger_sync))

    def set_clamp_mode(self, mode: Union[Literal["VC"], Literal["IC"]]) -> None:
        """
        Enable either current (IC) or voltage (VC) clamp mode.

        This will disable all incompatible compensation circuits, and re-enable any compensations that were previously
        enabled.

        Parameters
        ----------
        mode
            Either "IC" for current clamp or "VC" for voltage clamp.
        """
        if mode == self._clamp_mode:
            return
        self._clamp_mode = mode
        if mode == "IC":
            self._enable_vc_mode()
        elif mode == "VC":
            self._enable_ic_mode()
        else:
            raise ValueError(f"'{mode}' is not a valid clamp mode. Only 'VC' and 'IC' are accepted.")

    def _enable_ic_mode(self):
        self.set_cfast(enabled=False)
        self.set_bridge(enabled=False)
        self.set_cslow(**self._compensations["cslow"])
        self.set_serial_resistance(**self._compensations["serial_resistance"])
        self.call("set_current_clamp_mode", c_bool(False))

    def _enable_vc_mode(self):
        self.set_cslow(enabled=False)
        self.set_serial_resistance(enabled=False)
        self.set_cfast(**self._compensations["cfast"])
        self.set_bridge(**self._compensations["bridge"])
        self.call("set_current_clamp_mode", c_bool(True))

    VALID_SAMPLE_RATES = (1221, 4883, 9776, 19531, 50000, 100000, 200000)

    def set_sample_rate(self, rate: int) -> None:
        if rate not in self.VALID_SAMPLE_RATES:
            raise ValueError(f"'{rate}' is an invalid sample rate. Choose from {self.VALID_SAMPLE_RATES}.")
        with self.pause_receiving():
            self._sample_rate = rate
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
            self._current_input_range = current_range
            self.call("set_range", c_int(int(current_range / 1e-12)))  # Convert to int pA first

    VALID_CURRENT_OUTPUT_RANGES = (3.75e-9, 150e-9)

    def set_current_output_range(self, current_range: float):
        if current_range not in self.VALID_CURRENT_OUTPUT_RANGES:
            raise ValueError(
                f"'{current_range}' is not a valid current range. Choose from {self.VALID_CURRENT_OUTPUT_RANGES}."
            )
        self.call("enable_cc_higher_range", c_bool(current_range == 3.75e-9))

    def zap(self, duration=None):
        with self.pause_receiving():
            self.call("set_zap", c_bool(True))
        if duration is not None:
            time.sleep(duration)
            with self.pause_receiving():
                self.call("set_zap", c_bool(False))

    def set_cslow(self, enabled: bool = None, gain: float = None, tau: float = None) -> None:
        if enabled and self._clamp_mode != "VC":
            raise ValueError("cslow compensation cannot be enabled in IC mode")
        # TODO

    def set_cfast(self, enabled: bool = None, gain: float = None) -> None:
        if enabled and self._clamp_mode != "IC":
            raise ValueError("cfast compensation cannot be enabled in VC mode")
        # TODO

    def set_serial_resistance(
        self,
        enabled: bool = None,
        gain: float = None,
        prediction_rise_factor: int = None,  # 2 or 3
        tau: float = None,
        lag_filter: bool = None,
    ) -> None:
        if enabled and self._clamp_mode != "VC":
            raise ValueError("serial resistance compensation cannot be enabled in IC mode")
        # TODO

    def set_bridge(self, enabled: bool = None, gain: float = None):
        if enabled and self._clamp_mode != "IC":
            raise ValueError("bridge compensation cannot be enabled in VC mode")
        # TODO

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
                for column, handler in self._recv_handlers_raw:
                    if column is None:
                        handler(self._np_recv_buffer)
                    else:
                        handler(self._np_recv_buffer[column])

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

    def add_receive_data_handler_raw(self, handler, column=None):
        """

        Parameters
        ----------
        handler
            This callable should accept a single ndarray argument with a data type that mirrors `_uma_capture_struct` or
            one of its columns (if the `column` parameter is set). Data domains and ranges of this object will vary.
            "voltage" will be `±2**10 -> ±700mV`. "current" will be `±2**13 -> ±self.current_range`. "ts" (which stands
            for "timestamp") will be `TODO`.
            TODO mention conversion methods
        column : str
            If set, this specifies that only one column of the data will be passed to the handler (e.g. "voltage").
        """
        valid_columns = self._np_recv_buffer.dtype.names
        if column is not None and column not in valid_columns:
            raise ValueError(f"'{column}' is not a valid column name for captured data. Choose from {valid_columns}")
        self._recv_handlers_raw.append((column, handler))

    def quit(self):
        self.stop_receiving()
        self._run_recv_thread = False
        self._recv_thread.join()

    @contextmanager
    def pause_receiving(self):
        was_receiving = self.is_receiving()
        self.stop_receiving()
        yield
        if was_receiving:
            self.start_receiving()

    def is_receiving(self):
        return self._run_recv_thread and not self._pause_recv_thread


if __name__ == "__main__":
    UMP.set_debug_mode(True)
    um = UMP.get_ump()

    # LIBUM_SHARED_EXPORT int um_set_feature(um_state *hndl, const int dev, const int id, const int value);
    # um.call("um_set_feature", 1, 13, 1)
    # print(f"feature 13: {um.call('um_get_feature', 1, 13)}")
    # sleep(5)
    # print(f"feature 13: {um.call('um_get_feature', 1, 13)}")

    print(um.list_devices())
    dev1 = um.get_device(1)
    uma = UMA(um, dev1)

    # _uma_lib = cdll.LoadLibrary("/home/martin/src/acq4/uma-sdk/src/lib/libuma.so")
    dev_array = (c_int * LIBUM_MAX_DEVS)()
    # _uma_lib.uma_get_device_list.restype = c_int
    # init_ret = _uma_lib.uma_init", handle, c_int(1))
    n_devs = uma.call("get_device_list", byref(dev_array), c_int(LIBUM_MAX_MANIPULATORS))
    if n_devs >= 0:
        print([dev_array[i] for i in range(n_devs)])

    range_of_current = 20000e-12
    read_sample_rate = 9776

    uma.set_current_input_range(range_of_current)
    uma.set_sample_rate(read_sample_rate)
    uma.set_clamp_mode("VC")

    # Set VC output
    # ±500 mV w.r.t. Vcm
    #  1 mV (10-bit DAC)
    #  * @brief Set output voltage, 9 bits DAC at 9 lower bits, sign at bit 10, trig at bit 11
    vc_output = 0  # V
    trig = True
    vc_dac = int(2 ** 8 * abs(vc_output) / 0.5) | (0 if vc_output >= 0 else 2 ** 9) | (0 if not trig else 2 ** 11)
    uma.call("set_vc_dac", c_int16(vc_dac))

    uma.call("set_trig_bit", c_bool(trig))  # TODO what does this do?
    uma.call("set_wait_trig", c_bool(False))

    # Amplifier correction parameters
    voltage_offset = 0

    # units: mV
    uma.call("set_vc_voltage_offset", c_float(voltage_offset))

    # uma.call("set_vc_cfast_gain", c_float(value))
    # uma.call("set_vc_cslow_gain", c_float(value))
    # uma.call("set_vc_cslow_tau", c_float(value))

    uma.call("set_vc_rs_corr_gain", c_float(0))
    # uma.call("set_vc_rs_corr_tau", c_float(value))

    # uma.call("enable_vc_rs_fast_lag_filter", c_bool(value))

    uma.call("enable_vc_rs_pred_3x_gain", c_bool(False))

    # ±100 nA or ±2.5 nA CC command range
    uma.call("enable_cc_higher_range", c_bool(False))

    # 17 bit DAC; range set by enable_cc_higher_range
    uma.call("set_cc_dac", c_int(0))

    # uma.call("set_cc_cfast_gain", c_int(value))
    # uma.call("set_cc_bridge_gain", c_int(0))

    # Stimulus setup
    N_SAMPLES = 600  # max 749
    np_stimulus = np.zeros((N_SAMPLES,))

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
        int(5.0 * read_sample_rate), dtype=[("ts", float), ("current", float), ("voltage", float), ("status", int)]
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

    def handle_scaled_recv(received_data):
        global graph_data, t_offset
        t_offset = received_data["ts"][0] if t_offset is None else t_offset
        graph_data = np.roll(graph_data, -len(received_data))

        graph_data[-len(received_data) :]["ts"] = received_data["ts"] - t_offset
        graph_data[-len(received_data) :]["current"] = received_data["current"]
        graph_data[-len(received_data) :]["voltage"] = received_data["voltage"]
        graph_data[-len(received_data) :]["status"] = received_data["status"]

    uma.add_receive_data_handler_raw(handle_raw_recv)
    uma.start_receiving()

    def update_plots():
        t = graph_data["ts"]
        p1.plot(t, graph_data["current"], clear=True)
        p2.plot(t, graph_data["voltage"], clear=True)
        p3.plot(t, graph_data["status"], clear=True)

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update_plots)
    timer.start(10)

    def insert_stim(magnitude=30):
        global np_stimulus
        np_stimulus[0:-20] = magnitude
        uma.send_stimulus_raw(np_stimulus, trig)

    def stim_train():
        def _do_stims():
            for i in range(18):
                print(f"stimming at 2**{i}")
                insert_stim((2 ** i))
                time.sleep(0.1)
                print(f"stimming at 2**{i}-1")
                insert_stim((2 ** i) - 1)
            time.sleep(0.1)
            for i in range(18):
                print(f"stimming at -2**{i}")
                insert_stim(-(2 ** i))
                time.sleep(0.1)
                print(f"stimming at -2**{i} + 1")
                insert_stim(-(2 ** i) + 1)
                time.sleep(0.1)

        t = Thread(target=_do_stims, daemon=True)
        t.start()

    stim_timer = pg.QtCore.QTimer()
    stim_timer.timeout.connect(insert_stim)

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
            insert_stim(2**9)
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
