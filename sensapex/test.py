import argparse
import sys
import time

from sensapex import SensapexDevice, UMP
from sensapex.sensapex import LIBUM_DEF_BCAST_ADDRESS


def _bytes_str(s):
    return bytes(s, "utf-8")


parser = argparse.ArgumentParser(
    description="Test for sensapex devices; prints position and status updates continuously."
)
parser.add_argument(
    "--library-path", type=str, dest="library_path", default=None, help="Folder containing the umsdk library"
)
parser.add_argument("--address", type=_bytes_str, default=LIBUM_DEF_BCAST_ADDRESS, help="Device network address")
parser.add_argument("--group", type=int, default=0, help="Device group number")
args = parser.parse_args()

UMP.set_library_path(args.library_path)
um = UMP.get_ump(address=args.address, group=args.group)
try:
    um.set_packet_capture_on_exception(True)
except (ImportError, RuntimeError, OSError) as ex:
    print(f"packet capture cannot be enabled: {ex}")
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
