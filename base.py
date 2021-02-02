import numpy as np
import scipy.signal as scs
from time_series_buffer import TimeSeriesBuffer as buf
from time_series_metadata.scheme import MetaData as meta


class PhysicalPhenomenon:
    def __init__(self):

        # build the base chirp signal
        times = np.linspace(0, 5, 200)
        chirp = scs.chirp(times, f0=1, t1=5, f1=3)

        self.base_signal = 1 * np.concatenate((chirp, chirp[::-1]))
        self.counter = 0
        self.p_new_offset = 0.02
        self.offset = 10

    def value(self, time):
        value = self.base_signal[self.counter] + self.offset

        # change offset sometimes
        if np.random.random() < self.p_new_offset:
            self.offset = np.random.randn()

        # increase cyclic counter
        self.counter += 1
        if self.counter >= self.base_signal.size:
            self.counter = 0

        return value
        # return 10 + 0.01 * np.random.randn()


class Sensor:
    def __init__(
        self,
        transfer_model,
        estimated_transfer_model=None,
        estimated_compensation_model=None,
    ):
        self.transfer_model = transfer_model  # simulation model
        self.estimated_transfer_model = estimated_transfer_model  # calibration model
        self.estimated_compensation_model = (
            estimated_compensation_model  # compensation model
        )

    def indicated_value(self, physical_phenomenon_value):
        value, value_unc = self.transfer_model.apply(physical_phenomenon_value, 0)
        return value, value_unc

    def estimated_value(self, indicated_value, indicated_uncertainty):
        value, value_unc = self.estimated_compensation_model.apply(
            indicated_value, indicated_uncertainty
        )
        return value, value_unc
