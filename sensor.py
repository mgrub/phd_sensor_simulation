import numpy as np
from models import LinearAffineModel


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

    def indicated_value(self, measurand_value):
        value, value_unc = self.transfer_model.apply(measurand_value, 0)
        return value, value_unc

    def estimated_value(self, indicated_value, indicated_uncertainty):
        value, value_unc = self.estimated_compensation_model.apply(
            indicated_value, indicated_uncertainty
        )
        return value, value_unc


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
    


def init_sensor_objects(sensor_descriptions):
    """ load a bunch of sensors from description """

    sensors = {}

    for sensor_name, sensor_params in sensor_descriptions.items():
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

        sensors[sensor_name] = sensor

    return sensors
