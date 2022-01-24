from collections import deque
from os import remove

import numpy as np
from scipy.stats import chi2, iqr
from time_series_buffer import TimeSeriesBuffer

from models import LinearAffineModel
from co_calibration_gibbs_sampler_expressions import (
    posterior_Xa_explicit,
    posterior_a_explicit,
    posterior_b_explicit,
    posterior_sigma_y_explicit,
)

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


class Gruber(CocalibrationMethod):
    def cox_fusion(self, reference_sensor_values, reference_sensor_uncs):

        # def fuse(self, value_uncs, values):

        weights = 1 / np.square(reference_sensor_uncs)
        val, val_unc = self.weighted_mean(
            reference_sensor_values, reference_sensor_uncs, weights
        )

        # chi-square test for outliers
        chi2_obs = np.sum(
            np.square((reference_sensor_values - val[:, None]) / reference_sensor_uncs),
            axis=1,
        )
        dof = reference_sensor_values.shape[1] - 1
        p = chi2.sf(chi2_obs, dof)

        # remove outliers
        for i in range(len(p)):
            val[i], val_unc[i] = self.outlier_removal(
                p[i],
                val[i],
                val_unc[i],
                weights[i],
                reference_sensor_values[i],
                reference_sensor_uncs[i],
            )

        return val, val_unc

    def weighted_mean(self, values, value_uncs, weights):
        k = np.sum(weights, axis=1)

        # calculate the weighted mean
        val = np.sum(weights * values, axis=1) / k

        # uncertainty according GUM for uncorrelated inputs
        val_unc = np.linalg.norm(weights / k[:, None] * value_uncs, ord=2, axis=1)

        return val, val_unc

    def outlier_removal(
        self, p_row, val_row, val_unc_row, weights_row, values_row, value_uncs_row
    ):
        if p_row < 0.05:
            d = values_row - val_row
            ud = np.square(value_uncs_row) - np.square(val_unc_row)
            within_limits = np.abs(d) <= 2 * ud

            if np.any(within_limits):
                values_inside = values_row[within_limits][:, None]
                value_uncs_inside = value_uncs_row[within_limits][:, None]
                weights_inside = weights_row[within_limits][:, None]
                # [:,None] -> add singelton dimension

                # recalculate the fusion value without outliers
                val, val_unc = self.weighted_mean(
                    values_inside, value_uncs_inside, weights_inside
                )

            else:  # if all values are rejected, take the median
                median_index = np.argsort(values_row)[len(values_row) // 2]
                val_row = values_row[median_index]
                val_unc_row = value_uncs_row[median_index]

        return val_row, val_unc_row


class GibbsPosterior(Gruber):
    def __init__(
        self,
        gibbs_runs=100,
        burn_in=10,
        use_every=5,
        sigma_y_is_given=False,
        no_error_in_variables_model=False,
        use_robust_statistics=False,
        prior=None,
        sigma_y_true=0.2,
    ):
        self.gibbs_runs = gibbs_runs
        self.burn_in = burn_in
        self.use_every = use_every
        self.sigma_y_is_given = sigma_y_is_given
        self.no_error_in_variables_model = no_error_in_variables_model
        self.use_robust_statistics = use_robust_statistics
        self.prior = prior
        self.sigma_y_true = sigma_y_true

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
        fused_reference, fused_reference_unc = self.cox_fusion(
            references, references_unc
        )

        # run MCM
        Uxx = np.diag(np.square(fused_reference_unc))
        posterior = self.gibbs_routine(fused_reference, Uxx, np.squeeze(dut_indications))

        # return estimate
        result.append(
            {
                "time": dut_timestamps[-1],
                "params": posterior,
            }
        )

        return result


    def gibbs_routine(self, xx_observed, Uxx, yy):

        Uxx_inv = np.linalg.inv(Uxx)

        # shortcuts for prior
        mu_a = self.prior["a"]["mu"]
        mu_b = self.prior["b"]["mu"]
        mu_sigma_y = self.prior["sigma_y"]["mu"]

        sigma_a = self.prior["a"]["sigma"]
        sigma_b = self.prior["b"]["sigma"]
        sigma_sigma_y = self.prior["sigma_y"]["sigma"]

        # update posteriors using (block-)Gibbs sampling
        samples = []
        # initital sample from best guess (or any other method?)
        initial_sample = {
            "Xa" : xx_observed,
            "a" : mu_a,
            "b" : mu_b,
            "sigma_y" : mu_sigma_y,
        }
        samples.append(initial_sample)

        # init variables
        ps = samples[0]
        Xa_gibbs = ps["Xa"]
        a_gibbs = ps["a"]
        b_gibbs = ps["b"]
        sigma_y_gibbs = ps["sigma_y"]

        for i_run in range(1, self.gibbs_runs): # gibbs runs

            if self.no_error_in_variables_model:
                Xa_gibbs = xx_observed
            else:
                # sample from posterior of Xa
                Xa_gibbs = posterior_Xa_explicit(None, a_gibbs, b_gibbs, sigma_y_gibbs, yy, xx_observed, Uxx_inv)

            # sample from posterior of a
            a_gibbs = posterior_a_explicit(None, b_gibbs, Xa_gibbs, sigma_y_gibbs, yy, mu_a, sigma_a)

            # sample from posterior of b
            b_gibbs = posterior_b_explicit(None, a_gibbs, Xa_gibbs, sigma_y_gibbs, yy, mu_b, sigma_b)

            # sample from posterior of sigma_y
            if self.sigma_y_is_given:
                sigma_y_gibbs = self.sigma_y_true
            else:
                sigma_y_gibbs = posterior_sigma_y_explicit(None, a_gibbs, b_gibbs, Xa_gibbs, yy, mu_sigma_y, sigma_sigma_y)

            # 
            samples.append({
            "Xa" : Xa_gibbs,
            "a" : a_gibbs,
            "b" : b_gibbs,
            "sigma_y" : sigma_y_gibbs,
        })

        # estimate posterior from (avoid burn-in and take only every nth sample to avoid autocorrelation)

        considered_samples = samples[self.burn_in::self.use_every]
        AA = [sample["a"] for sample in considered_samples]
        BB = [sample["b"] for sample in considered_samples]
        SY = [sample["sigma_y"] for sample in considered_samples]

        if self.use_robust_statistics:
            posterior = {
                "a" : {
                    "mu" : np.median(AA),
                    "sigma" : iqr(AA),
                },
                "b" : {
                    "mu" : np.median(BB),
                    "sigma" : iqr(BB),
                },
                "sigma_y" : {
                    "mu" : np.median(SY),
                    "sigma" : iqr(SY),
                }
            }
        else:
            posterior = {
                "a" : {
                    "mu" : np.mean(AA),
                    "sigma" : np.std(AA),
                },
                "b" : {
                    "mu" : np.mean(BB), 
                    "sigma" : np.std(BB),
                },
                "sigma_y" : {
                    "mu" : np.mean(SY),
                    "sigma" : np.std(SY),
                }
            }

        # prepare next cycle
        self.prior = posterior

        return posterior




class AnalyticalDiscretePosterior(Gruber):
    def __init__(
        self,
        prior=None,
    ):
        self.prior = prior

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
        fused_reference, fused_reference_unc = self.cox_fusion(
            references, references_unc
        )

        # run MCM
        Uxx = np.diag(np.square(fused_reference_unc))
        posterior = self.update_discrete_posterior(fused_reference, Uxx, np.squeeze(dut_indications))

        # return estimate
        result.append(
            {
                "time": dut_timestamps[-1],
                "params": posterior,
            }
        )

        return result

    def update_discrete_posterior(self, xx_observed, Uxx, yy):
        return self.prior
