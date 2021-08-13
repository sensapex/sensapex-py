import argparse
import sys
import time

from sensapex import SensapexDevice, UMP
from sensapex.sensapex import LIBUM_DEF_BCAST_ADDRESS
from sensapex.utils import bytes_str

parser = argparse.ArgumentParser(
    description="Test for sensapex devices; prints position and status updates continuously."
)
parser.add_argument(
    "--library-path", "-l", type=str, dest="library_path", default=None, help="Folder containing the umsdk library"
)
parser.add_argument("--address", "-a", type=bytes_str, default=LIBUM_DEF_BCAST_ADDRESS, help="Device network address")
parser.add_argument("--group", "-g", type=int, default=0, help="Device group number")
parser.add_argument("--debug", "-d", action="store_true", help="Turn on debug logging")
args = parser.parse_args()

UMP.set_library_path(args.library_path)
um = UMP.get_ump(address=args.address, group=args.group)
try:
    um.set_debug_mode(args.debug)
except RuntimeError as e:
    print(f"Could not enable Sensapex debug mode: {e}")
devids = um.list_devices()
devs = {i: SensapexDevice(i) for i in devids}

print("SDK version:", um.sdk_version())
print("Found device IDs:", devids)


def print_pos(timeout=None):
    line = ""
    for i in devids:
        dev = devs[i]
        try:
            pos = str(dev.get_pos(timeout=timeout))
        except Exception as err:
            pos = str(err.args[0])
        pos = pos + " " * (30 - len(pos))
        line += f"{i:d}:  {pos}"
    print(line)


t = time.time()
while True:
    t1 = time.time()
    dt = t1 - t
    t = t1
    line = f"{dt:3f}"
    for id in sorted(list(devs.keys())):
        line += f"   {id:d}: {devs[id].get_pos(timeout=0)} busy: {devs[id].is_busy()}"
    line += "                           \r"
    print(line, end=" ")
    sys.stdout.flush()
    time.sleep(0.01)
