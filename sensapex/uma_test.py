"""

Tests to write:

- linearity: can we stimulate and record over the full range with low error? 
- timing: 
    Do stimuli appear where we expect them to with no jitter? 
    Are recorded responses synchronized?
    Does sample rate remain accurate over long recordings?
- model cell: do we get expected peak / steady state resistances?
- noise: are electrical noise measurements in expected range?
- automatic compensations
    - pipette offset
    - bridge balance
    - cap comp
    - series resistance
- streaming: can we achieve gapless recording / stimulation
- what happens if we send a stimulus before streaming has started? Do we get consistent start times?

Contitions to repeat tests under:
- VC / IC
- All sample rates
- All ADC/DAC ranges
- Compensations enabled / disabled




"""


import time

import neuroanalysis.stimuli as stimuli
from sensapex import UMP
from sensapex.uma import UMA


UMP.set_debug_mode(True)
um = UMP.get_ump()
dev1 = um.get_device(1)
uma = UMA(dev1)



def send_receive_stimulus(uma, stimulus, duration, clamp_mode, **parameters):
    """Configure a uMa device, send a stimulus, and return the recorded result.
    """
    with uma.lock:
        uma.stop_receiving()

        uma.set_clamp_mode(clamp_mode)
        uma.set_params(parameters)
        all_parameters = uma.get_params()

        sample_rate = uma.get_sample_rate()
        n_samples = int(sample_rate * duration)
        stim_array = stimulus.eval(sample_rate=sample_rate, n_pts=n_samples)

        with uma.stream() as stream:
            fut = uma.send_stimulus_scaled(stim_array.data)
            fut.result()

    return {
        'data': stream.data[fut.start_index:fut.end_index],
        'parameters': all_parameters,
        'stimulus': stimulus,
        'stimulus_samples': stim_array.data,
    }


def send_receive_square_pulse_sequence(uma, clamp_mode, start_time, pulse_duration, amplitudes, **parameters):
    total_duration = pulse_duration + 2 * start_time
    for amp in amplitudes:
        stim = stimuli.SquarePulse(start_time=start_time, duration=pulse_duration, amplitude=amp)
        yield send_receive_stimulus(uma, stim, total_duration, clamp_mode, **parameters)


def send_receive_pulse_train(uma, ):
    stim = stimuli.SquarePulseTrain(start_time=10e-3, n_pulses=100, pulse_duration=10e-3, interval=50e-3, amplitude=50e-3)


class ResultPlotter:
    def __init__(self):
        self.win = pg.GraphicsLayoutWidget()
        self.v_plot = self.win.addPlot()
        self.i_plot = self.win.addPlot(row=1, col=0)
        self.i_plot.setXLink(self.v_plot)
        self.v_plot.setLabels(left=('voltage', 'V'))
        self.i_plot.setLabels(left=('current', 'A'))
        self.win.show()
        self.result_count = 0

    def clear(self):
        self.v_plot.clear()
        self.i_plot.clear()

    def add_result(self, result, max_results):
        color = pg.intColor(self.result_count, max_results)
        data = result['data']
        t = data['time'] - data['time'][0]
        self.v_plot.plot(t, data['voltage'], pen=color)
        self.i_plot.plot(t, data['current'], pen=color)

        if result['parameters']['clamp_mode'] == 'VC':
            self.v_plot.plot(t, result['stimulus_samples'], pen={'color': color, 'style': pg.QtCore.Qt.DashLine})
        else:
            self.i_plot.plot(t, result['stimulus_samples'], pen={'color': color, 'style': pg.QtCore.Qt.DashLine})

        self.result_count += 1


if __name__ == '__main__':
    import pyqtgraph as pg
    import numpy as np

    app = pg.mkQApp()

    UMP.set_debug_mode(True)
    um = UMP.get_ump()
    dev1 = um.get_device(1)
    uma = UMA(dev1)

    plotter = ResultPlotter()
    amps = np.linspace(-0.05, 0.05, 11)
    for result in send_receive_square_pulse_sequence(uma, 'VC', start_time=100e-3, pulse_duration=200e-3, amplitudes=amps):
        plotter.add_result(result, max_results=len(amps))
        app.processEvents()
