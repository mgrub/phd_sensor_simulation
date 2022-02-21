import numpy as np
from models import LinearAffineModel


class CalibratedSensor:
    def __init__(
        self,
        transfer_model,
        estimated_transfer_model=None,
        estimated_compensation_model=None,
        outlier_rate=0.0,
        dropout_rate=0.0,
    ):
        self.transfer_model = transfer_model  # simulation model
        self.estimated_transfer_model = estimated_transfer_model  # calibration model
        self.estimated_compensation_model = (
            estimated_compensation_model  # compensation model
        )

        self.outlier_rate = outlier_rate
        self.dropout_rate = dropout_rate

    def indicated_value(self, measurand_value):
        value, value_unc = self.transfer_model.apply(measurand_value, 0)

        if self.outlier_rate:
            outliers = np.random.random(size=len(measurand_value)) < self.outlier_rate
            value[outliers] = np.random.uniform(1e2, 1e8, size=outliers.sum())

        if self.dropout_rate:
            dropouts = np.random.random(size=len(measurand_value)) < self.dropout_rate
            value[dropouts] = np.nan
            value_unc[dropouts] = np.nan

        return value, value_unc

    def estimated_value(self, indicated_value, indicated_uncertainty):
        value, value_unc = self.estimated_compensation_model.apply(
            indicated_value, indicated_uncertainty
        )
        return value, value_unc


def generate_sensor_descriptions(sensor_args, draw=False, number=1):
    sensors = {}
    for i in range(number):
        sensor_name = f"sensor_{i}"
        sensor = generate_sensor_description(sensor_args, draw=draw)
        sensors[sensor_name] = sensor
    return sensors


def generate_sensor_description(sensor_args, draw=False):
    model_type = sensor_args["model"]["type"]
    if model_type == "LinearAffineModel":
        params = sensor_args["model"]["params"]

        transfer = LinearAffineModel(**params)
        params = transfer.get_params()

        if draw:
            # take into account that estimated transfer model is not the same as actual transfer model
            # select random model parameters
            a = params["a"] + params["ua"] * np.random.randn()
            b = params["b"] + params["ub"] * np.random.randn()
            ua = params["ua"] * (0.5 + np.random.random())
            ub = params["ub"] * (0.5 + np.random.random())

            transfer_estimate = LinearAffineModel(a=a, b=b, ua=ua, ub=ub)
            params_est = transfer_estimate.get_params()
            params_inv = transfer_estimate.inverse_model_parameters()
        
        else:
            params_est = params
            params_inv = transfer.inverse_model_parameters()
    
    else:
        raise ValueError(f"Unsupported model type <{model_type}>.")

    sensor_description = {
        "hasSimulationModel" : {
            "type" : model_type,
            "params" : params,
        },
        "hasCalibrationModel" : {
            "type" : model_type,
            "params" : params_est,
        },
        "hasCompensationModel" : {
            "type" : model_type,
            "params" : params_inv,
        },
        "misc" : sensor_args["misc"],
    }

    return sensor_description
    


def init_sensor_objects(sensor_descriptions):
    """ load a bunch of sensors from description """

    sensors = {}
    model_dict = {
        "transfer_model" : "hasSimulationModel", 
        "estimated_transfer_model" : "hasCalibrationModel",
        "estimated_compensation_model" : "hasCompensationModel", 
    }

    for sensor_name, sensor_description in sensor_descriptions.items():

        model_args = {}

        # iterate over the three different internally used models
        for model_name, property_name in model_dict.items():
            model_type = sensor_description[property_name]["type"]

            # load transfer behavior
            if model_type == "LinearAffineModel":
                model = LinearAffineModel(
                    **sensor_description[property_name]["params"]
                )
            else:
                raise NotImplementedError(
                    f"Transfer Model type {model_type} not supported."
                )

            model_args[model_name] = model

        # init sensor
        sensor = CalibratedSensor(**model_args, **sensor_description["misc"])

        sensors[sensor_name] = sensor

    return sensors
