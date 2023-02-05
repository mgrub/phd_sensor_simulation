import numpy as np
import scipy.signal as scs
import scipy.interpolate as sci


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

        self.p_new_offset = 0.02
        self.previous_offset = 1

    def value(self, time):
        # chirp signal
        t1 = 5
        t_adj = np.abs(t1 * scs.sawtooth(2*np.pi*time))
        value = scs.chirp(t_adj, f0=1, t1=t1, f1=3)

        # change offset sometimes
        if self.random_jumps:
            self.offset = np.random.randn()
            jumps = np.random.random(size=len(time)) < self.p_new_offset
            time_jumps_at = time[jumps]
            height_after_jumps = self.previous_offset + np.random.randn(jumps.sum())
            
            f = sci.interp1d(np.r_[time[0]-1, time_jumps_at], np.r_[self.previous_offset, height_after_jumps], kind="previous", fill_value="extrapolate")
            offset = f(time)
            self.previous_offset = offset[-1]
            value += offset
        else:
            value += self.previous_offset       

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
