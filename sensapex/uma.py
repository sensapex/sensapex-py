import atexit

import faulthandler

import time
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

import numpy as np

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

    def __init__(self, sensapex_connection: UMP, uMp: SensapexDevice):
        self._state = _uma_state_struct()
        self._lock = sensapex_connection.lock
        self._libuma = sensapex_connection.libuma
        self.sensapex = sensapex_connection
        self.call("init", sensapex_connection.h, c_int(uMp.dev_id))

    def call(self, fn_name: str, *args):
        fn_name = f"uma_{fn_name}"
        self.sensapex.write_debug(f"uMa calling fn {fn_name}")
        return self.sensapex.call_lib_fn(self._libuma, fn_name, byref(self._state), *args)
        # return self.sensapex.call_uma_fn(fn_name, self._state, *args)

    def send_stimulus(self, stimulus: np.ndarray, trigger_sync: bool = False) -> None:
        """

        Parameters
        ----------
        stimulus : ndarray
            Up to 749 points of stimulus, which will be cast to 32-bit integers. The units depend on the clamp mode.
            For VC, TODO. For IC, TODO.
        trigger_sync : bool
            TODO what does this end up doing?
        """
        if len(stimulus.shape) != 1:
            raise ValueError(f"Stimulus may only be 1D. Received {stimulus.shape}.")
        stim_len = stimulus.shape[0]
        if stim_len > 749:
            raise ValueError(f"Stimulus must have 749 or fewer points. Received {stim_len}.")
        # TODO check for clipping if passed in longs/floats?
        stimulus = stimulus.astype(c_int)
        c_stimulus = np.ctypeslib.as_ctypes(stimulus)
        self.call("stimulus", c_int(stim_len), c_stimulus, c_bool(trigger_sync))


UMP.set_debug_mode(True)
um = UMP.get_ump()

# LIBUM_SHARED_EXPORT int um_set_feature(um_state *hndl, const int dev, const int id, const int value);
# um.call("um_set_feature", 1, 13, 1)
# print(f"feature 13: {um.call('um_get_feature', 1, 13)}")
# sleep(5)
# print(f"feature 13: {um.call('um_get_feature', 1, 13)}")

uma_state = _uma_state_struct()

handle = um.h
print(um.list_devices())
dev1 = um.get_device(1)
uma = UMA(um, dev1)

# _uma_lib = cdll.LoadLibrary("/home/martin/src/acq4/uma-sdk/src/lib/libuma.so")
devarray = (c_int * LIBUM_MAX_DEVS)()
# _uma_lib.uma_get_device_list.restype = c_int
# init_ret = _uma_lib.uma_init", handle, c_int(1))
n_devs = uma.call("get_device_list", byref(devarray), c_int(LIBUM_MAX_MANIPULATORS))
if n_devs >= 0:
    print([devarray[i] for i in range(n_devs)])

range_of_current = 20000  # {200, 2000, 20000, 200000} pA
read_sample_rate = 9776

# set _input_ current range in VC mode
# 200 = ±200 pA
# 2000 = ±2 nA
# 20000 = ±20 nA
# 200000 = ±200 nA
uma.call("set_range", c_int(range_of_current))

uma.call("set_sample_rate", c_int(read_sample_rate))  # 1221, 4883, 9776, 19531, 50000
uma.call("set_current_clamp_mode", c_bool(False))

# Set VC output
# ±500 mV w.r.t. Vcm
#  1 mV (10-bit DAC)
#  * @brief Set output voltage, 9 bits DAC at 9 lower bits, sign at bit 10, trig at bit 11
vc_output = 0  # V
trig = True
vc_dac = int(2 ** 8 * abs(vc_output) / 0.5) | (0 if vc_output >= 0 else 2 ** 9) | (0 if not trig else 2 ** 11)
uma.call("set_vc_dac", c_int16(vc_dac))

uma.call("set_trig_bit", c_bool(trig))
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
# stimulus = (c_int * N_SAMPLES)()
# np_stimulus = np.frombuffer(stimulus, dtype=c_int)
np_stimulus = np.zeros((N_SAMPLES,))
np_stimulus[0:-20] = 30

# Start recording samples
uma.call("start")

UMA_CAPTURES_PER_PACKET = 1440 // sizeof(_uma_capture_struct)

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

buffer = (_uma_capture_struct * UMA_CAPTURES_PER_PACKET)()
np_buff = np.frombuffer(
    buffer,
    dtype=[
        ("status", c_uint8),
        ("flags", c_uint8),
        ("index", c_uint16),
        ("ts", c_uint32),
        ("current", c_uint16),
        ("voltage", c_uint16),
    ],
)

read_data = np.zeros(
    int(5.0 * read_sample_rate), dtype=[("ts", float), ("current", float), ("voltage", float), ("status", int)]
)
t_offset = None

run_recv_thread = True
pause_recv_thread = False


def recv_thread():
    global read_data, t_offset
    while run_recv_thread:
        while pause_recv_thread:
            time.sleep(1)
        try:
            uma.call("recv", UMA_CAPTURES_PER_PACKET, buffer)
        except UMError as e:
            if e.errno == LIBUM_TIMEOUT:
                print("timed out")
            else:
                raise e
        t_offset = t_offset or np_buff["ts"][0]
        read_data = np.roll(read_data, -len(np_buff))

        read_data[-len(np_buff):]["ts"] = (np_buff["ts"] - t_offset) * 1e-6
        # ±range_of_current pA w.r.t. ground/common
        # read_data[-len(np_buff):]["current"] = (1e-12 * range_of_current / (2 ** 15)) * np_buff["current"]
        read_data[-len(np_buff):]["current"] = (1e-12 * range_of_current / (2 ** 15)) * (np_buff["current"].astype(int) - 2 ** 15)
        # ±700 mV w.r.t. ground/common
        # read_data[-len(np_buff):]["voltage"] = np_buff["voltage"] * (0.7 / 2 ** 15)
        read_data[-len(np_buff):]["voltage"] = (0.7 / 2 ** 15) * (np_buff["voltage"].astype(int) - 2 ** 15)
        read_data[-len(np_buff):]["status"] = np_buff["status"]


thread = Thread(target=recv_thread, daemon=True)
thread.start()


def update_plots():
    t = read_data["ts"]
    p1.plot(t, read_data["current"], clear=True)
    p2.plot(t, read_data["voltage"], clear=True)
    p3.plot(t, read_data["status"], clear=True)


timer = pg.QtCore.QTimer()
timer.timeout.connect(update_plots)
timer.start(10)


def insert_stim():
    uma.send_stimulus(np_stimulus, trig)


stim_timer = pg.QtCore.QTimer()
stim_timer.timeout.connect(insert_stim)


# stim_timer.start(1000)


def cleanup():
    global run_recv_thread
    timer.stop()
    stim_timer.stop()
    run_recv_thread = False
    uma.call("stop")


atexit.register(cleanup)


def pause():
    global pause_recv_thread
    pause_recv_thread = True
    time.sleep(0.1)
    uma.call("stop")


def unpause():
    global pause_recv_thread
    uma.call("start")
    pause_recv_thread = False


def test_stim():
    global timer, stim_timer, read_data
    pause()
    # MC the next uma call will hang
    start_ts = read_data["ts"][-1]
    print(f"latest timestamp on a sample at start stim {start_ts}")
    insert_stim()
    print("starting")
    unpause()
    print("sleeping")
    time.sleep(2)
    local_read_data = read_data.copy()
    try:
        trig_index = np.argwhere(local_read_data["current"] > 20e-12)[0, 0]
        trig_time = local_read_data[trig_index]["ts"]
        print(f"Stim detected at t={trig_time}")
        print(f"time difference: {trig_time - start_ts}")
    except IndexError:
        print("Could not detect a current > 20e-12")
