import json
import numpy as np
from time_series_buffer import TimeSeriesBuffer
from models import LinearAffineModel


def generate_sensors(type, args, draw=False, n=1):
    sensors = {}
    for i in range(n):
        sensor_name = f"sensor_{i}"
        sensor = generate_sensor(type, args, draw=draw)
        sensors[sensor_name] = sensor
    return sensors


def generate_sensor(type, kwargs, draw=False):
    if type == "LinearAffineModel":

        transfer = LinearAffineModel(**kwargs)
        params = transfer.get_params()

        if draw:
            # take into account that estimated transfer model is not the same as actual transfer model
            # select random model parameters
            a = kwargs["a"] + kwargs["ua"] * np.random.randn()
            b = kwargs["b"] + kwargs["ub"] * np.random.randn()
            ua = kwargs["ua"] * (0.5 + np.random.random())
            ub = kwargs["ub"] * (0.5 + np.random.random())

            transfer_estimate = LinearAffineModel(a=a, b=b, ua=ua, ub=ub)
            params_est = transfer_estimate.get_params()
            params_inv = transfer_estimate.inverse_model_parameters()
        
        else:
            params_est = params
            params_inv = transfer.inverse_model_parameters()
    
    else:
        raise ValueError(f"Unsupported model type <{type}>.")

    sensor = {
        "transfer_behavior_type": type,
        "transfer_behavior_params": params,
        "estimated_transfer_behavior_type": type,
        "estimated_transfer_behavior_params": params_est,
        "estimated_correction_model_type": type,
        "estimated_correction_model_params": params_inv
    }

    return sensor
    


def init_sensors(path=None, jsonstring=None, maxlen=None):
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
