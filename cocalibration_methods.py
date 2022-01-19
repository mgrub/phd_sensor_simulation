from collections import deque
import numpy as np
from models import LinearAffineModel
from time_series_buffer import TimeSeriesBuffer


class CocalibrationMethod:
    def __init__(self):
        pass

    def update_params(self, sensor_readings, device_under_test):
        pass

    def data_conversion(self, sensor_readings, device_under_test):

        device_under_test_name = list(device_under_test.keys())[0]

        references = np.array(
            [
                sensor_readings[sn]["val"]
                for sn in sensor_readings
                if sn is not device_under_test_name
            ]
        ).T
        references_unc = np.array(
            [
                sensor_readings[sn]["val_unc"]
                for sn in sensor_readings
                if sn is not device_under_test_name
            ]
        ).T

        dut_timestamps = np.array(sensor_readings[device_under_test_name]["time"])
        dut_indications = np.array([sensor_readings[device_under_test_name]["val"]]).T
        dut_indications_unc = np.array(
            [sensor_readings[device_under_test_name]["val_unc"]]
        ).T

        return (
            dut_timestamps,
            references,
            references_unc,
            dut_indications,
            dut_indications_unc,
            device_under_test_name,
        )


class Stankovic(CocalibrationMethod):
    def __init__(
        self,
        calc_unc=False,
        use_unc=False,
        use_enhanced_model=False,
        delay=5,
        delta=0.001,
    ):
        self.calc_unc = calc_unc
        self.use_unc = use_unc
        self.use_enhanced_model = use_enhanced_model
        self.delta = delta
        self.delay = delay

        self.buffer_indication = TimeSeriesBuffer(maxlen=delay, return_type="list")

    def update_params(self, sensor_readings, device_under_test):

        (
            dut_timestamps,
            references,
            references_unc,
            dut_indications,
            dut_indications_unc,
            device_under_test_name,
        ) = self.data_conversion(sensor_readings, device_under_test)
        dut_object = device_under_test[device_under_test_name]
        result = []

        # sequentially loop over all timestamps
        for timestamp, neighbor_values, neighbor_uncertainties, y, uy in zip(
            dut_timestamps,
            references,
            references_unc,
            dut_indications,
            dut_indications_unc,
        ):

            # perform next estimation step
            dut_object = self.update_params_single_timestep(
                timestamp, neighbor_values, neighbor_uncertainties, y, uy, dut_object
            )

            # store result
            if isinstance(dut_object.estimated_transfer_model, LinearAffineModel):
                p, up = dut_object.estimated_transfer_model.get_params(
                    separate_unc=True
                )
            result_timestamp = {
                "time": timestamp,
                "params": {
                    "a": {
                        "val": p["a"],
                        "val_unc": up["ua"],
                    },
                    "b": {
                        "val": p["b"],
                        "val_unc": up["ub"],
                    },
                },
            }
            result.append(result_timestamp)

        return result

    def update_params_single_timestep(
        self, timestamp, neighbor_values, neighbor_uncertainties, y, uy, dut
    ):

        # update buffer
        self.buffer_indication.add(time=timestamp, val=y, val_unc=uy)

        # estimated based on most recent parameter estimate
        x_hat, ux_hat = dut.estimated_value(y, uy)

        if len(self.buffer_indication) == self.delay:
            model = dut.estimated_compensation_model
            param = model.parameters

            y_delayed = self.buffer_indication.show(self.delay)[2][0]
            # J = np.sum(np.square(neighbor_values - x_hat) / neighbor_uncertainties)
            if self.use_unc:
                weights = neighbor_uncertainties / np.linalg.norm(
                    neighbor_uncertainties
                )
            else:
                weights = np.ones_like(neighbor_values)

            enhanced_factor = np.ones(2)
            if self.use_enhanced_model:
                enhanced_factor[0] += self.delta * np.square(uy) * np.sum(weights)

            grad_J = np.sum(
                (neighbor_values - x_hat) * np.array([[y_delayed, 1]]).T / weights,
                axis=1,
            )

            new_param = param * enhanced_factor + self.delta * grad_J

            # adjust parameter estimation uncertainties
            if self.calc_unc:
                C = self.build_derivative(
                    param,
                    neighbor_values,
                    weights,
                    y,
                    uy,
                    self.delta,
                    self.use_enhanced_model,
                )
                U = self.build_full_input_uncertainty_matrix(
                    model.parameters_uncertainty, neighbor_uncertainties, uy
                )
                new_param_unc = C @ U @ C.T
            else:
                new_param_unc = None

            model.set_parameters(
                parameters=new_param, parameters_uncertainty=new_param_unc
            )

            # update DUT
            # calculate estimated inverse model
            p_inv, up_inv = model.inverse_model_parameters(separate_unc=True)
            dut.estimated_transfer_model = LinearAffineModel(**p_inv, **up_inv)

        return dut

    def build_derivative(
        self, param, neighbor_value_estimates, weights, y, uy, delta, use_enhanced_model
    ):
        nn = neighbor_value_estimates.size
        shape = (2, 2 + nn + 1)
        C = np.zeros(shape)
        a = param[0]
        b = param[1]

        # derivatives
        C[0, 0] = 1 + delta * np.sum(weights * (-np.square(y)))
        C[0, 1] = 0 + delta * np.sum(weights * (-y))
        C[0, 2 : 2 + nn] = 0 + delta * weights * y
        C[0, -1] = 0 + delta * np.sum(
            weights * (-2 * a * y + neighbor_value_estimates - b)
        )

        C[1, 0] = 0 + delta * np.sum(weights * (-y))
        C[1, 1] = 1 + delta * np.sum(weights * (-1))
        C[1, 2 : 2 + nn] = 0 + delta * weights
        C[1, -1] = 0 + delta * np.sum(weights * (-a))

        if use_enhanced_model:
            C[0, 0] += delta * np.square(uy) * np.sum(weights)

        return C

    def build_full_input_uncertainty_matrix(
        self, param_unc, neighbor_uncertainty_estimates, uy
    ):
        main_diag = np.hstack((np.diag(param_unc), neighbor_uncertainty_estimates, uy))
        U = np.diag(main_diag)
        U[:2, :2] = param_unc

        return U


class GibbsPosterior(CocalibrationMethod):
    def __init__(
        self,
        gibbs_runs=100,
        burn_in=10,
        use_every=5,
        sigma_y_is_given=False,
        no_error_in_variables_model=False,
        use_robust_statistics=False,
    ):
        self.gibbs_runs = gibbs_runs

    def update_params(self, sensor_readings, device_under_test):
        (
            dut_timestamps,
            references,
            references_unc,
            dut_indications,
            dut_indications_unc,
            device_under_test_name,
        ) = self.data_conversion(sensor_readings, device_under_test)
        result = []

        # cox fusion of reference

        # run MCM

    def cox_fusion(self, timestamps, reference_sensor_values, reference_sensor_uncs):
        pass


class DirectPosterior(CocalibrationMethod):
    def __init__(self, gibbs_runs=1000, further_settings=None):
        self.gibbs_runs = gibbs_runs

    def update_params(self, timestamps, reference_sensor_values):
        pass

    def cox_fusion(self, timestamps, reference_sensor_values, reference_sensor_uncs):
        pass
