import platform


def bytes_str(s):
    return bytes(s, "utf-8")


packet_count_param = "-n" if platform.system().lower() == "windows" else "-c"


# def ping(host, timeout=1):
#     """Ping *host* once and return the latency, or None if no response is received.
#     """
#     timeout = int(timeout)
#     try:
#         now = time.time()
#         addr = get_host_ip(host)
#         out = subprocess.check_output(['ping', '-c1', '-n', addr], timeout=timeout).decode()
#         if time.time() - now > 1:
#             print("!!!!", host)
#             print(out)
#     except subprocess.TimeoutExpired:
#         return None
#     except Exception as exc:
#         raise Exception("Ping %s failed: %s" % (host, exc.output.decode()))
#
#     n_recv = None
#     for line in out.split('\n'):
#         m = re.match(r'(\d) packets transmitted, (\d) received.*', line)
#         if m is not None:
#             n_sent = int(m.groups()[0])
#             n_recv = int(m.groups()[1])
#             if n_recv < n_sent:
#                 print(host, 'timeout')
#                 return None
#         m = re.match(r'rtt min/avg/max/mdev = (?P<min>[0-9\.]+)/(?P<avg>[0-9\.]+)/(?P<max>[0-9\.]+)/.*', line)
#         if m is not None:
#             if n_recv is None:
#                 break
#             lat = float(m.group('max'))
#             return lat * 1e-3
#
#     raise Exception("Could not parse ping output: %s" % out)
#
#
# def ping_parallel(addrs, timeout=3, max_workers=16):
#     """Ping multiple addresses in parallel.
#     """
#     exec = concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(addrs)))
#     futures = {}
#     for addr in addrs:
#         futures[addr] = exec.submit(ping, addr, timeout)
#     return {addr: fut.result() for (addr, fut) in futures.items()}
#
#
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
#
#
# class ScanThread(threading.Thread):
#     def __init__(self, addresses: Iterable[str], on_complete):
#         self._addresses = addresses
#         self._on_complete = on_complete
#         self._stop_called = False
#         super(ScanThread, self).__init__(daemon=True)
#
#     def stop(self):
#         self._stop_called = True
#
#     def run(self):
#         responsive = set()
#         pings_in_progress = 0
#         lock = threading.Lock()
#
#         def do_ping(host: str):
#             nonlocal pings_in_progress
#             command = ["ping", packet_count_param, "1", host]
#             while pings_in_progress >= 10:
#                 if self._stop_called:
#                     return
#                 time.sleep(0.1)
#             with lock:
#                 pings_in_progress += 1
#             try:
#                 process = subprocess.run(command, timeout=1, capture_output=True)
#                 if process.returncode == 0:
#                     responsive.add(host)
#             except subprocess.TimeoutExpired:
#                 pass
#             with lock:
#                 pings_in_progress -= 1
#
#         threads = [threading.Thread(target=do_ping, args=(h,)) for h in self._addresses]
#         [th.start() for th in threads]
#         for th in threads:
#             if not self._stop_called:
#                 th.join()
#         if not self._stop_called:
#             self._on_complete(responsive)
