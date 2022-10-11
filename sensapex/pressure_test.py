import argparse
import sys
import time

import numpy as np
import pyqtgraph as pg
from sensapex import UMP
from sensapex.sensapex import LIBUM_DEF_BCAST_ADDRESS
from sensapex.utils import bytes_str


parser = argparse.ArgumentParser(
    description="Test for sensapex pressure controllers."
)
parser.add_argument("device", type=int, help="Device ID to test")
parser.add_argument(
    "--library-path", "-l", type=str, dest="library_path", default=None, help="Folder containing the umsdk library"
)
parser.add_argument("--address", "-a", type=bytes_str, default=LIBUM_DEF_BCAST_ADDRESS, help="Device network address")
parser.add_argument("--debug", "-d", default=False, action="store_true", help="Turn on debug logging")
parser.add_argument("--group", "-g", type=int, default=0, help="Device group number")
args = parser.parse_args()

UMP.set_library_path(args.library_path)
ump = UMP.get_ump(address=args.address, group=args.group)
if args.debug:
    try:
        ump.set_debug_mode(True)
    except Exception as e:
        print(f"Could not enable Sensapex debug mode: {e}")
time.sleep(2)
devids = ump.list_devices()
devs = {i: ump.get_device(i) for i in devids}

print("SDK version:", ump.sdk_version())
print("Found device IDs:", devids)

dev = devs[args.device]


app = pg.mkQApp()
win = pg.GraphicsLayoutWidget()
plots = [win.addPlot(row=i, col=0, labels={"left": (f"channel {i+1} (kPa)"), "bottom": ("time", "s")}) for i in range(8)]
for i in range(0, 8):
    plots[i].setXLink(plots[0])
    plots[i].addLine(x=0)
err_plots = [win.addPlot(row=i, col=1) for i in range(8)]
err_curves = [plt.plot(symbol='o') for plt in err_plots]
for i in range(8):
    err_plots[i].addLine([0, 0], angle=45)

win.resize(600, 1000)
win.show()


for chan in range(1, 9):
    start_valve = dev.get_valve(chan)

    dev.set_valve(chan, 0)
    assert dev.get_valve(chan) == 0, f"channel {chan} failed valve test"

    dev.set_valve(chan, 1)
    assert dev.get_valve(chan) == 1, f"channel {chan} failed valve test"

    dev.set_valve(chan, start_valve)

    accuracy = np.inf

    pressure_values = range(-50, 51, 10)
    results = np.zeros((len(pressure_values), 2))
    results[:] = np.nan
    pre_samples = 30
    post_samples = 40

    for i, target_pressure in enumerate(pressure_values):
        pressure_data = []
        time_vals = []
        dev.set_pressure(chan, 0.0)
        for j in range(pre_samples):
            pressure_data.append(dev.measure_pressure(chan))
            time_vals.append(time.perf_counter())
            time.sleep(0.001)
        set_time = time.perf_counter()
        dev.set_pressure(chan, target_pressure)
        for j in range(post_samples):
            pressure_data.append(dev.measure_pressure(chan))
            time_vals.append(time.perf_counter())
            time.sleep(0.001)
        pressure_data = np.array(pressure_data)
        results[i] = [target_pressure, pressure_data[-10:].mean()]
        if target_pressure != 0:
            accuracy = min(accuracy, 1.0 - (((np.array(pressure_data[-10:]) - target_pressure)**2).mean()**0.5) / target_pressure)
        plots[chan-1].plot(np.array(time_vals) - set_time, pressure_data)
        err_curves[chan-1].setData(results[:,0], results[:,1], symbol='o')
        app.processEvents()
    dev.set_pressure(chan, 0.0)

    app.processEvents()

    print(f"Channel {chan} pressure accuracy: {accuracy}")




if sys.flags.interactive == 0:
    app.exec_()

