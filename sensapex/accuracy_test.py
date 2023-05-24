import argparse
import sys
import time

import numpy as np
import pyqtgraph as pg
from sensapex import UMP
from sensapex.sensapex import LIBUM_DEF_BCAST_ADDRESS
from sensapex.utils import bytes_str


parser = argparse.ArgumentParser(
    description="Test for sensapex devices; perform a series of random moves while rapidly polling the device position"
                " and state."
)
parser.add_argument("device", type=int, help="Device ID to test")
parser.add_argument(
    "--library-path", "-l", type=str, dest="library_path", default=None, help="Folder containing the umsdk library"
)
parser.add_argument("--address", "-a", type=bytes_str, default=LIBUM_DEF_BCAST_ADDRESS, help="Device network address")
parser.add_argument("--debug", "-d", action="store_true", help="Turn on debug logging")
parser.add_argument("--group", "-g", type=int, default=0, help="Device group number")
parser.add_argument(
    "--x", action="store_true", default=False, dest="x", help="True = Random X axis values. False = keep start position"
)
parser.add_argument(
    "--y", action="store_true", default=False, dest="y", help="True = Random Y axis values. False = keep start position"
)
parser.add_argument(
    "--z", action="store_true", default=False, dest="z", help="True = Random Z axis values. False = keep start position"
)

parser.add_argument("--speed", type=float, default=1000, help="Movement speed in um/sec")
parser.add_argument(
    "--distance", type=float, default=10, help="Max distance to travel in um (relative to current position)"
)
parser.add_argument("--iter", type=int, default=10, help="Number of positions to test")
parser.add_argument("--acceleration", type=float, default=0, help="Max speed acceleration")
parser.add_argument(
    "--retry-threshold",
    type=float,
    default=None,
    dest="retry_threshold",
    help="Distance error threshold at which to retry a move",
)
parser.add_argument(
    "--linear", action="store_true", default=False, dest="linear", help="Move all 3 axes simultaneously"
)
parser.add_argument(
    "--high-res",
    action="store_true",
    default=False,
    dest="high_res",
    help="Use high-resolution time sampling rather than poller's schedule",
)
parser.add_argument(
    "--start-pos",
    type=str,
    default=None,
    dest="start_pos",
    help="x,y,z starting position (by default, the current position is used)",
)
parser.add_argument(
    "--test-pos",
    type=str,
    default=None,
    dest="test_pos",
    help="x,y,z position to test (by default, random steps from the starting position are used)",
)
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
win.show()
plots = [
    win.addPlot(labels={"left": ("x position", "m"), "bottom": ("time", "s")}),
    win.addPlot(labels={"left": ("y position", "m"), "bottom": ("time", "s")}),
    win.addPlot(labels={"left": ("z position", "m"), "bottom": ("time", "s")}),
]
plots[1].setYLink(plots[0])
plots[2].setYLink(plots[0])
plots[1].setXLink(plots[0])
plots[2].setXLink(plots[0])

win.nextRow()
errplots = [
    win.addPlot(labels={"left": ("x error", "m"), "bottom": ("time", "s")}),
    win.addPlot(labels={"left": ("y error", "m"), "bottom": ("time", "s")}),
    win.addPlot(labels={"left": ("z error", "m"), "bottom": ("time", "s")}),
]
errplots[1].setYLink(errplots[0])
errplots[2].setYLink(errplots[0])
errplots[0].setXLink(plots[0])
errplots[1].setXLink(plots[0])
errplots[2].setXLink(plots[0])

if args.linear:
    win.nextRow()
    linerrplots = [
        win.addPlot(labels={"left": ("x linear error", "m"), "bottom": ("time", "s")}),
        win.addPlot(labels={"left": ("y linear error", "m"), "bottom": ("time", "s")}),
        win.addPlot(labels={"left": ("z linear error", "m"), "bottom": ("time", "s")}),
    ]
    linerrplots[1].setYLink(linerrplots[0])
    linerrplots[2].setYLink(linerrplots[0])
    linerrplots[0].setXLink(plots[0])
    linerrplots[1].setXLink(plots[0])
    linerrplots[2].setXLink(plots[0])

start = time.perf_counter()
pos = [[], [], []]
tgt = [[], [], []]
err = [[], [], []]
closest = [[], [], []]
linear_err = [[], [], []]
bus = []
mov = []
times = []

lastupdate = time.perf_counter()


def update(moving=True):
    global lastupdate, n_axes
    timeout = -1 if args.high_res else 0
    position = dev.get_pos(timeout=timeout)
    s = dev.is_busy()
    m = not move_req.finished
    bus.append(int(s))
    mov.append(int(m))
    now = time.perf_counter() - start
    times.append(now)

    # calculate closest point to the line from starting position to target
    target_to_pos = position - target
    target_to_last = last_position_before_move - target
    target_to_last /= np.linalg.norm(target_to_last)
    closest_pos = target + np.dot(target_to_pos, target_to_last) * target_to_last
    dist = position - closest_pos

    for i in range(min(3, n_axes)):
        pos[i].append((position[i] - start_pos[i]) * 1e-6)
        tgt[i].append((target[i] - start_pos[i]) * 1e-6)
        if moving:
            err[i].append(np.nan)
            # only update linear error when moving
            closest[i].append((closest_pos[i] - start_pos[i]) * 1e-6)
            linear_err[i].append(dist[i] * 1e-6)
        else:
            # only update position error when stopped
            err[i].append(pos[i][-1] - tgt[i][-1])
            closest[i].append(np.nan)
            linear_err[i].append(np.nan)


def update_plots():
    for i in range(3):
        plots[i].clear()
        plots[i].addItem(
            pg.PlotCurveItem(times, bus[:-1], stepMode=True, pen=None, brush=(0, 255, 0, 40), fillLevel=0),
            ignoreBounds=True,
        )
        plots[i].addItem(
            pg.PlotCurveItem(times, mov[:-1], stepMode=True, pen=None, brush=(255, 0, 0, 40), fillLevel=0),
            ignoreBounds=True,
        )
        plots[i].plot(times, tgt[i], pen="r")
        plots[i].plot(times, pos[i], symbol="o", symbolSize=5)
        plots[i].plot(times, closest[i], pen=(100, 100, 100))
        errplots[i].plot(times, err[i], clear=True, connect="finite")
        if args.linear:
            linerrplots[i].plot(times, linear_err[i], clear=True, connect="finite")


if args.start_pos is None:
    start_pos = dev.get_pos()
else:
    start_pos = np.array(list(map(float, args.start_pos.split(","))))

print("Starting position:", start_pos)
diffs = []
errs = []
positions = []
n_axes = dev.n_axes()
assert (
    len(start_pos) == n_axes
), f"Starting position {start_pos} has length {len(start_pos)}, but device has {n_axes} axes."
if args.test_pos is None:
    moves = np.random.random(size=(args.iter, n_axes)) * args.distance
    move_axes = np.array([args.x, args.y, args.z, False])[:n_axes]
    assert np.any(move_axes), "No axes selected to move (use --x, --y, --z, or --test-pos)"
    moves[:, ~move_axes] = 0
    targets = np.array(start_pos)[np.newaxis, :] + moves
    print(f"Distance to move each axis:\n{moves}")
    print(f"Target positions:\n{targets}")
else:
    # just move back and forth between start and test position
    test_pos = np.array(list(map(float, args.test_pos.split(","))))
    targets = np.zeros((args.iter, 3))
    targets[::2] = start_pos[None, :]
    targets[1::2] = test_pos[None, :]
speeds = [args.speed] * args.iter


dev.stop()

if args.retry_threshold is not None:
    ump.set_retry_threshold(args.retry_threshold)

for i in range(args.iter):
    target = targets[i]
    last_position_before_move = dev.get_pos()
    move_req = dev.goto_pos(
        target, speed=speeds[i], linear=args.linear, simultaneous=args.linear, max_acceleration=args.acceleration
    )
    while not move_req.finished:
        update(moving=True)
        time.sleep(0.002)
    waitstart = time.perf_counter()
    while time.perf_counter() - waitstart < 1.0:
        update(moving=False)
        time.sleep(0.002)
        # time.sleep(0.05)
    p2 = dev.get_pos(timeout=200)
    positions.append(p2)
    diff = (p2 - target) * 1e-6
    diffs.append(diff)
    errs.append(np.linalg.norm(diff))

    print(
        f"{i} attempts: {move_req.attempts}  error: {errs[-1]*1e6:0.2f} um "
        f"  [{' '.join(['%0.2f'%(x*1e6) for x in diff])}]"
    )

update_plots()

dev.goto_pos(start_pos, args.speed)
print(f"mean: {np.mean(errs) * 1e6:0.2f} um   max: {np.max(errs) * 1e6:0.2f} um")

if sys.flags.interactive == 0:
    app.exec_()
