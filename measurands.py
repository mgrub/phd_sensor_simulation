import numpy as np
import scipy.signal as scs


class SinusoidalMeasurand:
    def __init__(self, sigma_x=0.01, static_omega=2*np.pi, amplitude=1.0, phase_offset=0.0, value_offset=0.0):
        self.static_omega = static_omega
        self.sigma_x = sigma_x
        self.amplitude = amplitude
        self.phase_offset = phase_offset
        self.value_offset = value_offset

    def omega(self, time, f_min=1, f_max=10, period=60):

        if isinstance(self.static_omega, (float, int)):
            return self.static_omega

        else:
            a = (np.log(f_max) - np.log(f_min)) / 2
            b = a + np.log(f_min)
            omega = 2 * np.pi * np.exp(a * (np.sin(2 * np.pi * time / period)) + b)
            return omega

    def value(self, time):
        value = self.amplitude * np.sin(self.omega(time) * time + self.phase_offset) + self.value_offset
        result = {"time": time, "quantity": value + self.sigma_x * np.random.randn()}
        return result


class JumpingMeasurand:
    def __init__(self, sigma_x=0.01, random_jumps=False):
        self.sigma_x = sigma_x
        self.random_jumps = random_jumps

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
        if self.random_jumps and np.random.random() < self.p_new_offset:
            self.offset = np.random.randn()

        # increase cyclic counter
        self.counter += 1
        if self.counter >= self.base_signal.size:
            self.counter = 0

        result = {"time": time, "quantity": value + self.sigma_x * np.random.randn()}
        return result


def return_measurand_object(type, kwargs):
    if type == "SinusoidalMeasurand":
        return SinusoidalMeasurand(**kwargs)
    elif type == "JumpingMeasurand":
        return JumpingMeasurand(**kwargs)
    else:
        raise ValueError(f"Unsupported measurand type <{type}>.")


def return_timestamps(time_start=0.0, time_end=10.0, dt=0.1, jitter=0.0):

    if jitter == 0.0:
        time = np.arange(time_start, time_end, step=dt)
    else:
        tmp = np.random.normal(loc=dt, scale=jitter, size=int((time_end * 1.2) // dt))  # 20% more than required ideally 
        tmp = tmp[tmp > 0]
        time = np.cumsum(tmp)
        time = time[time <= time_end]

    return time
