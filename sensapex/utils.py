import platform
import subprocess
import threading
import time
from typing import Iterable

packet_count_param = "-n" if platform.system().lower() == "windows" else "-c"


# class TrioThread(threading.Thread):
#     def __init__(self, addresses: Iterable[str], on_complete):
#         self._addresses = addresses
#         self._on_complete = on_complete
#         self._stop_called = False
#         super(TrioThread, self).__init__(daemon=True)
#
#     def run(self):
#         self._on_complete(trio.run(self._async_scan_subnet_trio))
#
#     def stop(self):
#         self._stop_called = True
#
#     async def _async_scan_subnet_trio(self) -> Set[str]:
#         responsive = set()
#         pings_in_progress = 0
#
#         async def do_ping(host: str, cancel_scope: trio.CancelScope):
#             nonlocal pings_in_progress
#             command = ["ping", packet_count_param, "1", host]
#             while pings_in_progress >= 10:
#                 if self._stop_called:
#                     cancel_scope.cancel()
#                 await trio.sleep(0.1)
#             with trio.move_on_after(1):
#                 pings_in_progress += 1
#                 process = await trio.run_process(
#                     command,
#                     check=False,
#                     stdout=subprocess.DEVNULL,
#                     stderr=subprocess.DEVNULL,
#                 )
#                 if process.returncode == 0:
#                     responsive.add(host)
#             pings_in_progress -= 1
#
#         async with trio.open_nursery() as nursery:
#             self._nursery = nursery
#             [nursery.start_soon(do_ping, h, nursery.cancel_scope) for h in self._addresses]
#
#         return responsive


class ScanThread(threading.Thread):
    def __init__(self, addresses: Iterable[str], on_complete):
        self._addresses = addresses
        self._on_complete = on_complete
        self._stop_called = False
        super(ScanThread, self).__init__(daemon=True)

    def stop(self):
        self._stop_called = True

    def run(self):
        responsive = set()
        pings_in_progress = 0
        lock = threading.Lock()

        def do_ping(host: str):
            nonlocal pings_in_progress
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
