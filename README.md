# Sensapex SDK

This python library wraps the C SDK provided by Sensapex's umsdk. It provides general access to the
functions present therein, as well as a device-based abstraction.

### Installation

`pip install sensapex`

This library comes packaged
with [the latest umsdk library](http://dist.sensapex.com/misc/um-sdk/latest/) for windows. All other
versions can be downloaded and/or compiled separately and configured with `UMP.set_library_path`.

### Usage

```python
from sensapex import UMP

ump = UMP.get_ump()
dev_ids = ump.list_devices()

stage = ump.get_device(20)
stage.calibrate_zero_position()

manipulator = ump.get_device(4)
manipulator.goto_pos((2500.0412, 6810.0003, 15830.1419), speed=200)

pressure = ump.get_device(30)
pressure.set_pressure(1, 2.3e-4)
```

Also included are some simple test scripts. The following will report on all devices present:

```bash
python -m sensapex.test
```

Or for a more involved test of your hardware as it moves around randomly,
install [pyqtgraph](https://pyqtgraph.org) in your environment and use e.g.:

```bash
python -m sensapex.accuracy_test <device_id>
```

Where `<device_id>` should be replaced with the numerical ID of your device, as shown by `python -m sensapex.test`.

#### ACQ4 Integration

Sensapex stages (uMs), manipulators (uMp) and pressure controllers (uMc) are all well integrated into the [ACQ4](http://acq4.org) neurophysiology platform. To access these devices from ACQ4, simply install `sensapex` into your ACQ4 environment and add appropriate configuration. In `default.cfg` add:
```yaml
drivers:
    sensapex:
        group: 0  # [default] of Group 'A' or whatever numeric index corresponds to your hardware
        debug: False  # [default] or True to get debug logging as described below
        address: "169.254.255.255"  # [default] or whatever your hardware's broadcast address is
        driverPath: None  # [default] to using standard OS library searches or set to a custom location
```

And in `devices.cfg`, add any number of the following device configuration sections (named distinctly):
```yaml
Stage:
    driver: "Sensapex"
    nAxes: 3
    deviceId: 20
    # ...

ObjectiveChanger:
    driver: 'SensapexObjectiveChanger'
    deviceId: 20
    # ...

Manipulator1:
    driver: "Sensapex"
    isManipulator: True
    deviceId: 1
    # ...

Pressure1:
    Driver: "SensapexPressureControl"
    deviceId: 30
    pressureChannel: 1
```

#### Debug

You can turn on debugging to produce detailed logs and network packet captures. First,
install [Wireshark](https://www.wireshark.org/download.html) (or for linux, use your package manager
to get the `pcaputils` package). Make sure the user account you use has permission to run
the `dumpcap` program. Next, install the psutil python package:

```bash
pip install psutil
```

Once those are installed, you can turn on the debug logging for your SDK wrapper:

```python
from sensapex import UMP

UMP.set_debug_mode(True)
ump = UMP.get_ump()
```

This will create a directory, `sensapex-debug/` in the current working directory, populated with a
log file and a pcap file. Repeated initializations of the debug mode will create addition pcap files
and append to the log file. These can be sent to
[Sensapex](mailto:support@sensapex.com) along with any relevant details, such as:

* A description of errant behavior
* The color of each of the relevant device lights
* A picture of the touchscreen state
* Steps to reproduce and how consistently it occurs

### Authorship

Copyright (c) 2016-2021 Sensapex Oy

Thanks to the following for contributions:

* Luke Campagnola
* Ari Salmi
* Martin Chase
* Thomas Braun

### Changelog

#### 1.022.8
* Include ACQ4 integration instructions in README

#### 1.022.7
* Packaging fix to include driver binary.
* find_library to find libum in default locations.

#### 1.022.6
* More accuracy_test improvements.
* Handle floating point movements.

#### 1.022.5
* The accuracy_test script now takes `--linear` to test linear movements.

#### 1.022.4
* Binary installer works in develop mode
* Errors no longer prevent other devices from moving
* Allow init args for UMP.get_device (to avoid n_axes race)

#### 1.022.3
* Bad MoveRequests should blow up on init, rather than during movement.

#### 1.022.2
* A bunch of fixes from Luke.

#### 1.022.1

* Debug mode: logs, hardware details and PCAP
* Bugfix in default library path for test scripts

#### 1.022.0

* Setup bdist_wheel that pre-downloads the SDK
* Bump SDK version to the latest

#### 1.021.2

* SDK version bump

#### 1.021.1

* Move-finish error capture
* Use new windows library name

#### 1.021.0

* Update to new version

#### 0.920.4

* Pressure devices don't need positioning callbacks

#### 0.920.3

* Expose more pressure functions.

#### 0.920.2

* Workaround for sdk bug in motion planning
