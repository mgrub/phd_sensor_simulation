import json
import random
import numpy as np
import scipy.signal as scs
from time_series_buffer import TimeSeriesBuffer
from models import LinearAffineModel


class DeterministicPhysicalPhenomenon:
    def __init__(self, tmin=0, tmax=10):

        n_jumps = random.choice([0,2,5])


    def omega(self, time, f_min=1, f_max=10, period=60):

        a = (np.log(f_max) - np.log(f_min)) / 2
        b = a + np.log(f_min)

        omega = 2*np.pi * np.exp(a * (np.sin(2*np.pi*time/period)) + b)

        return omega

    def value(self, time):
        return np.sin(self.omega(time) * time)


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


class SimulationHelper:
    def generate_sensors(self, n_neighbors=1):
        sensors = {}
        for i in range(n_neighbors+1):
            sensor_name = f"sensor_{i}"
            provides_reference = i < n_neighbors

            # select random model parameters
            a = 1 + 0.5 * np.random.randn()
            b = 5 + 0.5 * np.random.randn()
            ua = 0.1 * (1 + np.random.random())
            ub = 0.1 * (1 + np.random.random())

            # use in model
            transfer = LinearAffineModel(a=a, b=b, ua=ua, ub=ub)
            params = transfer.get_params()

            # get inverse model
            if provides_reference:
                # take into account that estimated transfer model is not the same as actual transfer model
                transfer_estimate = LinearAffineModel(a=a+ua*np.random.randn(), b=b+ub*np.random.randn(), ua=ua, ub=ub)
                params_est = transfer_estimate.get_params()
                params_inv = transfer_estimate.inverse_model_parameters()
            else:
                params_est = {"a": 1.0, "b": 0.0, "ua": 1.0, "ub": 1.0, "uab": 0.0}
                params_inv = {"a": 1.0, "b": 0.0, "ua": 1.0, "ub": 1.0, "uab": 0.0}

            sensors[sensor_name] = {
                "transfer_behavior_type": "LinearAffineModel",
                "transfer_behavior_params": params,
                "estimated_transfer_behavior_type": "LinearAffineModel",
                "estimated_transfer_behavior_params": params_est,
                "estimated_correction_model_type": "LinearAffineModel",
                "estimated_correction_model_params": params_inv,
                "provides_reference": provides_reference,
            }

        return json.dumps(sensors)

    def init_sensors(self, path=None, jsonstring=None, maxlen=None):
        """ load a bunch of sensors from json file/string """
        if path is not None:
            f = open(path, "r")
            jsonstring = f.read()

        sensors = json.loads(jsonstring)

        reference_sensors = []
        non_reference_sensors = []

        for sensor_params in sensors.values():
            # load transfer behavior
            transfer_type = sensor_params["transfer_behavior_type"]
            if transfer_type == "LinearAffineModel":
                transfer = LinearAffineModel(
                    **sensor_params["transfer_behavior_params"]
                )
            else:
                raise NotImplementedError(
                    f"Transfer Model type {transfer_type} not supported."
                )

            # load inverse behavior
            inverse_type = sensor_params["estimated_correction_model_type"]
            if inverse_type == "LinearAffineModel":
                inverse = LinearAffineModel(
                    **sensor_params["estimated_correction_model_params"]
                )
            else:
                raise NotImplementedError(
                    f"Transfer Model type {inverse_type} not supported."
                )

            # init sensor
            sensor = Sensor(
                transfer_model=transfer, estimated_compensation_model=inverse
            )
            if maxlen:
                buffer_indication = TimeSeriesBuffer(maxlen=maxlen, return_type="arrays")
                buffer_estimation = TimeSeriesBuffer(maxlen=maxlen, return_type="arrays")
            else:
                buffer_indication = None
                buffer_estimation = None

            tmp = {
                "sensor": sensor,
                "buffer_indication": buffer_indication,
                "buffer_estimation": buffer_estimation,
            }

            if sensor_params["provides_reference"]:
                reference_sensors.append(tmp)
            else:
                non_reference_sensors.append(tmp)

        return reference_sensors, non_reference_sensors
