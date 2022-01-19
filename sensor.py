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
        "transfer_model_type": type,
        "transfer_model_params": params,
        "estimated_transfer_model_type": type,
        "estimated_transfer_model_params": params_est,
        "estimated_compensation_model_type": type,
        "estimated_compensation_model_params": params_inv
    }

    return sensor
    


def init_sensor_objects(sensor_descriptions):
    """ load a bunch of sensors from description """

    sensors = {}

    for sensor_name, sensor_params in sensor_descriptions.items():

        # init the three different internally used models
        sensor_args = {
            "transfer_model" : None, 
            "estimated_transfer_model" : None,
            "estimated_compensation_model" : None, 
        }

        for model_name in sensor_args.keys():
            model_type = sensor_params[f"{model_name}_type"]

            # load transfer behavior
            if model_type == "LinearAffineModel":
                model = LinearAffineModel(
                    **sensor_params[f"{model_name}_params"]
                )
            else:
                raise NotImplementedError(
                    f"Transfer Model type {model_type} not supported."
                )

            sensor_args[model_name] = model

        # init sensor
        sensor = Sensor(**sensor_args)

        sensors[sensor_name] = sensor

    return sensors
