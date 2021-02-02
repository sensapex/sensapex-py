# Sensapex SDK

This python library wraps the C SDK provided by Sensapex's umsdk. It provides
general access to the functions present therein, as well as a device-based
abstraction.

### Installation

`pip install sensapex`

Download [the latest umsdk library](http://dist.sensapex.com/misc/um-sdk/rc/) for your platform.

### Usage

```python
from sensapex import UMP

UMP.set_library_path("/path/which/contains/umsdk/binary/for/your/platform/")
ump = UMP.get_ump()
ump.list_devices()

stage = ump.get_device(1)
stage.calibrate_zero_position()

manipulator = ump.get_device(4)
manipulator.goto_pos((-2500.0412, 6810.0003, 15830.1419), speed=2)

pressure = ump.get_device(30)
pressure.set_pressure(1, 2.3e-4)
```

Also included are some simple sanity checks. The following will report on all 
devices present:

```bash
python -m sensapex.test
```

Or for a more involved test of your hardware as it moves around randomly,
install [pyqtgraph](https://pyqtgraph.org) in your environment and use e.g.:

```bash
STAGE_DEVID=1
python -m sensapex.accuracy_test $STAGE_DEVID
```

### Authorship

Copyright (c) 2016-2020 Luke Campagnola

Thanks to the following for contributions:

 * Ari Salmi
 * Martin Chase
 * Thomas Braun

### Changelog

#### 0.920.4

* Pressure devices don't need positioning callbacks

#### 0.920.3

* Expose more pressure functions.

#### 0.920.2

* Workaround for sdk bug in motion planning
