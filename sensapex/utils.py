import platform
import subprocess
import threading
import time
from types import FunctionType
from typing import Iterable

packet_count_param = "-n" if platform.system().lower() == "windows" else "-c"


# def scan_subnet_trio(subnet: str, timeout: int = 1) -> Set[str]:
#     return trio.run(_async_scan_subnet_trio, subnet, timeout)
#
#
# async def _async_scan_subnet_trio(subnet: str, timeout: int) -> Set[str]:
#     responsive = set()
#     pings_in_progress = 0
#
#     async def do_ping(host: str):
#         nonlocal pings_in_progress, responsive
#         command = ["ping", packet_count_param, "1", host]
#         while pings_in_progress >= 10:
#             await trio.sleep(0.1)
#         with trio.move_on_after(timeout):
#             pings_in_progress += 1
#             process = await trio.run_process(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             if process.returncode == 0:
#                 responsive.add(host)
#         pings_in_progress -= 1
#
#     async with trio.open_nursery() as nursery:
#         [nursery.start_soon(do_ping, str(h)) for h in IPv4Network(subnet)]
#
#     return responsive


class PingThread(threading.Thread):
    def __init__(self, addresses: Iterable[str], on_complete):
        self._addresses = addresses
        self._on_complete = on_complete
        self._stop_called = False
        super(PingThread, self).__init__()

    def stop(self):
        self._stop_called = True

    def run(self):
        responsive = set()
        pings_in_progress = 0
        lock = threading.Lock()

        def do_ping(host: str):
            nonlocal pings_in_progress, responsive
            command = ["ping", packet_count_param, "1", host]
            while pings_in_progress >= 10:
                if self._stop_called:
                    return
                time.sleep(0.1)
            with lock:
                pings_in_progress += 1
            try:
                process = subprocess.run(command, timeout=1, capture_output=True)
                if process.returncode == 0:
                    responsive.add(host)
            except subprocess.TimeoutExpired:
                pass
            with lock:
                pings_in_progress -= 1

        threads = [threading.Thread(target=do_ping, args=(h,)) for h in self._addresses]
        [th.start() for th in threads]
        for th in threads:
            if not self._stop_called:
                th.join()
        if not self._stop_called:
            self._on_complete(responsive)
