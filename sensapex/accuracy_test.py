import argparse
import logging
import os
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
parser.add_argument("device", type=int, help="Device ID to test", nargs="?")
parser.add_argument(
    "--library-path", "-l", type=str, dest="library_path", default=None, help="Folder containing the umsdk library. By default, the library path is determined automatically."
)
parser.add_argument("--address", "-a", type=bytes_str, default=LIBUM_DEF_BCAST_ADDRESS, help=f"Device network broadcast address (default is {LIBUM_DEF_BCAST_ADDRESS.decode()})")
parser.add_argument("--debug", action="store_true", help="Turn on debug logging")
parser.add_argument("--group", "-g", type=str, default='A', help="Device group number (default is 'A'; see 'Device group configuration' on the touchscreen)")
parser.add_argument(
    "-x", action="store_true", default=False, dest="x", help="True = Random X axis values. False = keep start position"
)
parser.add_argument(
    "-y", action="store_true", default=False, dest="y", help="True = Random Y axis values. False = keep start position"
)
parser.add_argument(
    "-z", action="store_true", default=False, dest="z", help="True = Random Z axis values. False = keep start position"
)
parser.add_argument(
    "-d", action="store_true", default=False, dest="d", help="True = Random D axis values. False = keep start position"
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
    help="Distance error threshold (µm) at which to retry a move",
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

if args.library_path is not None:
    if not os.path.isdir(args.library_path):
        print(f"Library path {args.library_path} does not exist or is not a folder.")
        sys.exit(1)
    print(f"Adding {args.library_path} to library search path")
    UMP.set_library_path(args.library_path)

group = args.group
try:
    group = int(group)
except ValueError:
    assert len(group) == 1 and group.isalpha(), f"Group must be a single letter or integer, got {group!r}"
    group = ord(group.upper()) - ord('A')
group_letter = chr(group + ord('A'))
print(f"Using device group {group_letter} ({group})")

ump = UMP.get_ump(address=args.address, group=group)
print("SDK version:", ump.sdk_version().decode())

if args.debug:
    try:
        logging.basicConfig(level=logging.DEBUG)
        ump.set_debug_mode(True)
    except Exception as e:
        print(f"Could not enable Sensapex debug mode: {e}")

time.sleep(2)  # allow SDK time to find devices
devids = ump.list_devices()
print(f"Found {len(devids)} device ID{'s' if len(devids) != 1 else ''}:", devids)
if len(devids) == 0:
    print("No devices found. Things you can check:")
    print(" - Ensure the device is powered on and connected.")
    print(" - Verify the group setting is correct (see --group; this should match the")
    print("   group letter shown on the touchscreen under ☰ > Device group configuration).")
    print(" - Verify the network address (see --address; this should be the broadcast address")
    print("   compatible with the IP address shown on the touchscreen under ⚙ > Manipulator info).")
    print(" - Verify the device can be seen by the Sensapex firmware updater tool")
    sys.exit(1)

devs = {i: ump.get_device(i) for i in devids}

if args.device is None:
    print("Please specify a device ID (see above for available devices).")
    sys.exit(1)
dev = devs[args.device]
print(f"Testing device {args.device} ({dev.n_axes()} axes)")

n_axes = dev.n_axes()
axes = 'xyzd'[0:n_axes]

def check_pos_arg(name, arg):
    if arg is not None:
        vals = np.array(list(map(float, arg.split(","))))
        if len(vals) != n_axes:
            print(f"Error: {name} position {vals} has length {len(vals)}, but device has {n_axes} axes.")
            sys.exit(1)
        return np.array(vals)
    else:
        return None

# Determine starting position
start_pos = check_pos_arg("start", args.start_pos)
if start_pos is None:
    start_pos = np.array(dev.get_pos())
print("Starting position:", start_pos)

# Determine test position(s)
test_pos = check_pos_arg("test", args.test_pos)
if test_pos is None:
    # choose a set of random moves from the starting position, with the specified max distance along each axis
    moves = np.random.random(size=(args.iter, n_axes)) * args.distance
    move_axes = np.array([args.x, args.y, args.z, args.d])[:n_axes]
    if not np.any(move_axes):
        print("No axes selected to move (use -x, -y, -z, -d for random moves; or --test-pos for a specific test location)")
        sys.exit(1)
    moves[:, ~move_axes] = 0
    targets = np.array(start_pos)[np.newaxis, :] + moves
    print(f"Distance to move each axis:\n{moves}")
    print(f"Target positions:\n{targets}")
else:
    # just move back and forth between start and test position
    targets = np.zeros((args.iter, n_axes))
    targets[::2] = start_pos[None, :]
    targets[1::2] = test_pos[None, :]
    print(f"Testing moves between {start_pos} and {test_pos}:\n{targets}")

speeds = [args.speed] * args.iter
print(f"Using speed {args.speed} um/s for all moves")

# Just in case..
dev.stop()

if args.retry_threshold is not None:
    print(f"Setting move retry threshold to {args.retry_threshold} µm")
    ump.set_retry_threshold(args.retry_threshold)



# Start UI
app = pg.mkQApp()
win = pg.GraphicsLayoutWidget()
win.resize(1600, 800)
win.setWindowTitle(f"Sensapex accuracy test: device {args.device}, group {group_letter}")
win.show()

# create position plots for all axes and link together
plots = []
for i,ax in enumerate(axes):
    plots.append(
        win.addPlot(labels={"left": (f"{ax} position", "m"), "bottom": ("time", "s")})
    )
    plots[i].setYLink(plots[0])
    plots[i].setXLink(plots[0])

# add legend to last column
plots[-1].addLegend()

# create position error plots
win.nextRow()
errplots = []
for i,ax in enumerate(axes):
    errplots.append(
        win.addPlot(labels={"left": (f"{ax} error", "m"), "bottom": ("time", "s")})
    )
    # link y axes of error plots together
    errplots[i].setYLink(errplots[0])
    # link x axes of error plots to position plots
    errplots[i].setXLink(plots[0])



# create linear error plots (how far does move deviate from a straight line)
if args.linear:
    win.nextRow()
    linerrplots = []
    for i,ax in enumerate(axes):
        linerrplots.append(
            win.addPlot(labels={"left": (f"{ax} linear error", "m"), "bottom": ("time", "s")})
        )
        # link y axes of linear error plots together
        linerrplots[i].setYLink(linerrplots[0])
        # link x axes of linear error plots to position plots
        linerrplots[i].setXLink(plots[0])


# initialize test data arrays and timing
start = time.perf_counter()
pos = [[] for _ in axes]  # current position at each time step
tgt = [[] for _ in axes]  # target position at each time step
err = [[] for _ in axes]  # distance from position to target at each time step
final_errors = []  # error in position after each move completes
closest = [[] for _ in axes]    # closest point along the line from start to target at each time step
linear_err = [[] for _ in axes] # deviation from linear at each time step
bus = []  # device busy state at each time step
mov = []  # move request state at each time step
times = []  # time stamps for each update

lastupdate = time.perf_counter()


def update(moving=True):
    """Get current device position and state, then update position and error arrays
    """
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

    for i,ax in enumerate(axes):
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
    for i,ax in enumerate(axes):
        plots[i].clear()
        plots[i].addItem(
            pg.PlotCurveItem(times, bus[:-1], stepMode=True, pen=None, brush=(0, 255, 0, 40), fillLevel=0, name="busy"),
            ignoreBounds=True,
        )
        plots[i].addItem(
            pg.PlotCurveItem(times, mov[:-1], stepMode=True, pen=None, brush=(255, 0, 0, 40), fillLevel=0, name="moving"),
            ignoreBounds=True,
        )
        plots[i].plot(times, tgt[i], pen="r", name='target')
        plots[i].plot(times, pos[i], symbol="o", symbolSize=5, name='position')
        plots[i].plot(times, closest[i], pen=(100, 100, 100), name='closest')
        errplots[i].plot(times, err[i], clear=True, connect="finite", name='error')
        if args.linear:
            linerrplots[i].plot(times, linear_err[i], clear=True, connect="finite", name='linear error')

    # update all Y ranges
    plot_y_range = (
        min(np.nanmin(tgt[i]), np.nanmin(pos[i]), np.nanmin(closest[i])),
        max(np.nanmax(tgt[i]), np.nanmax(pos[i]), np.nanmax(closest[i]))
    )
    plots[0].setYRange(*plot_y_range)
    err_y_range = (np.nanmin(err), np.nanmax(err))
    if np.isfinite(err_y_range[0]) and np.isfinite(err_y_range[1]):
        errplots[0].setYRange(*err_y_range)
    if args.linear:
        linear_err_y_range = (np.nanmin(linear_err), np.nanmax(linear_err))
        if np.isfinite(linear_err_y_range[0]) and np.isfinite(linear_err_y_range[1]):
            linerrplots[0].setYRange(*linear_err_y_range)
    
    # update X ranges
    plots[0].enableAutoRange(x=True)



# Run the test!

for i in range(args.iter):
    # Move to next location
    target = targets[i]
    last_position_before_move = dev.get_pos()
    kwargs = dict(pos=target, speed=speeds[i], linear=args.linear, simultaneous=args.linear, max_acceleration=args.acceleration)
    print(f"\nStarting move {i+1}/{args.iter}")
    print(f"  to {kwargs['pos']}")
    print(f"  speed {kwargs['speed']} um/s")
    print(f"  linear: {kwargs['linear']}, simultaneous: {kwargs['simultaneous']}, max_acceleration: {kwargs['max_acceleration']}")
    move_req = dev.goto_pos(**kwargs)

    # Update data while in motion
    while not move_req.finished:
        update(moving=True)
        time.sleep(0.002)

    if move_req.interrupted:
        print(f"Move {i} failed after {move_req.attempts} attempts")
        print(f"  starting position: {move_req.start_pos}")
        print(f"  target position: {move_req.target_pos}")
        print(f"  final position: {move_req.last_pos}")
        print(f"  reason: {move_req.interrupt_reason}")
        if move_req.last_pos is None:
            print(f"   error requesting last position: {move_req.last_pos_exception}")
    else:
        print(f"Move {i} completed after {move_req.attempts} attempt{'s' if move_req.attempts != 1 else ''}")

    # Pause before starting next move; keep updating (watching for motion after the move has completed)
    waitstart = time.perf_counter()
    while time.perf_counter() - waitstart < 1.0:
        update(moving=False)
        time.sleep(0.002)
        # time.sleep(0.05)

    # Measure final error in position for each move
    p2 = dev.get_pos(timeout=200)
    diff = (p2 - target) * 1e-6
    final_errors.append(np.linalg.norm(diff))

    print(
        f"  final position error: {final_errors[-1]*1e6:0.2f} µm "
        f"     [{' '.join([f'{x*1e6:0.2f}' for x in diff])}]"
    )

    # Plot all collected data so far
    update_plots()
    app.processEvents()

print(f"\n-----------")
print(f"mean position error: {np.mean(final_errors) * 1e6:0.2f} µm")
print(f"max position error: {np.max(final_errors) * 1e6:0.2f} µm")


# Go back to the starting position
dev.goto_pos(start_pos, args.speed)

# Run the qt event loop (if not in interactive mode) so that the plot window stays open
if sys.flags.interactive == 0:
    app.exec_()
