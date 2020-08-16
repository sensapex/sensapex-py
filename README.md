# Sensapex SDK

This python library wraps the C SDK provided by Sensapex's umsdk. It provide general access to the functions present therein, as well as a device-based abstraction.

### Installation

`pip install sensapex-sdk`

Download [the latest umsdk library](http://dist.sensapex.com/misc/um-sdk/rc/).

### Usage

```python
from sensapex import UMP

UMP.set_library_path("/path/which/contains/umsdk/for/your/platform/")
ump = UMP.get_ump()
ump.list_devices()

stage = ump.get_device(1)
stage.calibrate_zero_position()

manipulator = ump.get_device(4)
manipulator.goto_pos((-2500.0412, 6810.0003, 15830.1419), speed=2)

pressure = ump.get_device(30)
pressure.set_pressure(1, 2.3e-4)
```