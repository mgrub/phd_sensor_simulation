import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, LinearNDInterpolator
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, invgamma, iqr, norm
from time_series_buffer import TimeSeriesBuffer

from co_calibration_gibbs_sampler_expressions import (
    log_likelidhood_a_b_sigma_y_without_Xa, posterior_a_explicit,
    posterior_b_explicit, posterior_sigma_y_explicit, posterior_Xa_explicit)
from models import LinearAffineModel


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
        k = np.nansum(weights, axis=1)

        # calculate the weighted mean
        val = np.nansum(weights * values, axis=1) / k

        # uncertainty according GUM for uncorrelated inputs
        # not suitable if nans are expected:
        # val_unc = np.linalg.norm(weights / k[:, None] * value_uncs, ord=2, axis=1)
        # use this to achieve robustness against nans
        tmp = weights / k[:, None] * value_uncs
        val_unc = np.sqrt(np.nansum(np.square(tmp), axis=1))

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


    def laplace_approximation_1d(self, x, log_y):
        """ Given a log-PDF y given at discrete positions x, 
            estimate the laplace approximation (fit a gaussian
            around the maximum a-posteriori probability).

            Uses spline interpolation if possible to get position of
            maximum and an estimate of the hessian at that position.
        """
        
        ## interpolate
        logging.info(log_y)
        finite_entries = np.logical_not(np.isneginf(log_y))
        
        # only interpolate, if enough finite datapoints
        if finite_entries.sum() > 2:
            x_finite = x[finite_entries]
            y_interp = CubicSpline(x_finite, - log_y[finite_entries])
            y_interp_second_order_derivate = y_interp.derivative(2)
            
            ## find minimum and second order derivative at minimum
            result = minimize_scalar(y_interp, method="Bounded", bounds=[x_finite.min(), x_finite.max()])
            laplace_mean = result.x
            laplace_hess = y_interp_second_order_derivate(laplace_mean)
            laplace_std = 1 / np.sqrt(laplace_hess)

        else: # if no interpolation possible, fall back to grid specs
            i_max = np.argmax(log_y)
            laplace_mean = x[i_max]
            laplace_std = (x[i_max+1] - x[i_max-1]) / 2
        
        return laplace_mean, laplace_std


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

        Uxx = np.diag(np.square(fused_reference_unc))

        # only run if at least a single valid datapoint in the reference
        if not np.all(np.isnan(fused_reference)):
            # run MCM
            posterior = self.gibbs_routine(fused_reference, Uxx, np.squeeze(dut_indications, axis=-1))

            # return estimate
            result.append(
                {
                    "time": dut_timestamps[-1],
                    "params": posterior,
                }
            )

        return result


    def gibbs_routine(self, xx_observed, Uxx, yy):
        # handle nans (dropouts)
        mask = np.isfinite(xx_observed)
        xx_observed = xx_observed[mask]
        yy = yy[mask]
        Uxx = Uxx[:,mask][mask]

        Uxx_inv = np.linalg.inv(Uxx)

        # shortcuts for prior
        mu_a = self.prior["a"]["params"]["loc"]
        sigma_a = self.prior["a"]["params"]["scale"]

        mu_b = self.prior["b"]["params"]["loc"]
        sigma_b = self.prior["b"]["params"]["scale"]

        shape_sigma_y = self.prior["sigma_y"]["params"]["a"]
        loc_sigma_y = self.prior["sigma_y"]["params"]["loc"]
        scale_sigma_y = self.prior["sigma_y"]["params"]["scale"]
        mode_sigma_y = scale_sigma_y / (1 + shape_sigma_y)   # wikipedia


        # update posteriors using (block-)Gibbs sampling
        samples = []
        # initital sample from best guess (or any other method?)
        initial_sample = {
            "Xa" : xx_observed,
            "a" : mu_a,
            "b" : mu_b,
            "sigma_y" : mode_sigma_y,
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
                sigma_y_gibbs = posterior_sigma_y_explicit(None, a_gibbs, b_gibbs, Xa_gibbs, yy, shape_sigma_y, scale_sigma_y, loc_sigma_y)

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

        # estimate a dist to the discrete hist (to be used as next prior)
        a_norm_fit, a_norm_fit_std = norm.fit(AA)
        b_norm_fit, b_norm_fit_std = norm.fit(BB)
        sigma_invgamma_a, sigma_invgamma_loc, sigma_invgamma_scale = invgamma.fit(SY, floc=0.0)

        fitted_posterior = {
                "a" : {
                    "type" : "norm",
                    "params" : {
                        "loc" : a_norm_fit,
                        "scale" : a_norm_fit_std,
                    }
                },
                "b" : {
                    "type" : "norm",
                    "params" : {
                        "loc" : b_norm_fit,
                        "scale" : b_norm_fit_std,
                    }
                }, 
                "sigma_y" : {
                    "type" : "invgamma",
                    "params" : {
                        "a" : sigma_invgamma_a,
                        "loc" : sigma_invgamma_loc,
                        "scale" : sigma_invgamma_scale,
                    }
                },
            }

        # laplace approximatino for result / output
        a_dist, a_hist_range = np.histogram(AA, density=True)
        a_laplace_approx, a_laplace_approx_std = self.laplace_approximation_1d(a_hist_range[:-1], np.log(a_dist))

        b_dist, b_hist_range = np.histogram(BB, density=True)
        b_laplace_approx, b_laplace_approx_std = self.laplace_approximation_1d(b_hist_range[:-1], np.log(b_dist))

        sigma_y_dist, sigma_y_hist_range = np.histogram(SY, density=True)
        sigma_y_laplace_approx, sigma_y_laplace_approx_std = self.laplace_approximation_1d(sigma_y_hist_range[:-1], np.log(sigma_y_dist))

        posterior_laplace_approximation = {
            "a" : {
                "val" : a_laplace_approx,
                "val_unc" : a_laplace_approx_std,
            },
            "b" : {
                "val" : b_laplace_approx,
                "val_unc" : b_laplace_approx_std,
            },
            "sigma_y" : {
                "val" : sigma_y_laplace_approx,
                "val_unc" : sigma_y_laplace_approx_std,
            }
        }

        # visualize for DEBUGGING
        # fix, ax = plt.subplots(3,1)
        # a_range = np.linspace(min(AA), max(AA))
        # ax[0].hist(AA, 20, density=True);
        # ax[0].plot(a_range, norm.pdf(a_range, a_laplace_approx, a_laplace_approx_std));
        # ax[0].plot(a_range, norm.pdf(a_range, a_norm_fit, a_norm_fit_std));

        # b_range = np.linspace(min(BB), max(BB))
        # ax[1].hist(BB, 20, density=True);
        # ax[1].plot(b_range, norm.pdf(b_range, b_laplace_approx, b_laplace_approx_std));
        # ax[1].plot(b_range, norm.pdf(b_range, b_norm_fit, b_norm_fit_std));

        # sigma_y_range = np.linspace(min(SY), max(SY))
        # ax[2].hist(SY, 20, density=True);
        # ax[2].plot(sigma_y_range, norm.pdf(sigma_y_range, sigma_y_laplace_approx, sigma_y_laplace_approx_std));
        # ax[2].plot(sigma_y_range, invgamma.pdf(sigma_y_range, sigma_invgamma_a, sigma_invgamma_loc, sigma_invgamma_scale));
        
        # plt.show()

        # prepare next cycle
        self.prior = fitted_posterior

        return posterior_laplace_approximation


class AnalyticalDiscretePosterior(Gruber):

    def __init__(
        self,
        prior=None,
        use_adaptive_grid=False,
        grid_resolution=15,
    ):
        # discrete grid to evaluate the posterior on
        a_low = prior["a"]["params"]["loc"] - 2 * prior["a"]["params"]["scale"]
        a_high = prior["a"]["params"]["loc"] + 2 * prior["a"]["params"]["scale"]
        b_low = prior["b"]["params"]["loc"] - 2 * prior["b"]["params"]["scale"]
        b_high = prior["b"]["params"]["loc"] + 2 * prior["b"]["params"]["scale"]
        sigma_y_low = 1e-3
        sigma_y_high = 2e0

        self.a_range = np.linspace(a_low, a_high, num=grid_resolution)
        self.b_range = np.linspace(b_low, b_high, num=grid_resolution)
        self.sigma_y_range = np.logspace(np.log10(sigma_y_low), np.log10(sigma_y_high), num=grid_resolution)

        self.a_grid, self.b_grid, self.sigma_y_grid = np.meshgrid(self.a_range, self.b_range, self.sigma_y_range)
        self.discrete_log_posterior = self.init_informative_prior(prior)

        self.use_adaptive_grid = use_adaptive_grid
        self.grid_resolution = grid_resolution
        # debug
        #self.plot_discrete_distribution(self.discrete_posterior)


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

        # only run if at least a single valid datapoint in the reference
        if not np.all(np.isnan(fused_reference)):
            # update posterior
            Uxx = np.diag(np.square(fused_reference_unc))
            self.update_discrete_log_posterior(fused_reference, Uxx, np.atleast_1d(np.squeeze(dut_indications)))
            posterior = self.laplace_approximation_posterior()
            if self.use_adaptive_grid:
                self.update_grid()

            # return estimate
            result.append(
                {
                    "time": dut_timestamps[-1],
                    "params": posterior,
                }
            )

        return result


    def update_discrete_log_posterior(self, xx_observed, Uxx, yy):
        # handle nans (dropouts)
        mask = np.isfinite(xx_observed)
        xx_observed = xx_observed[mask]
        yy = yy[mask]
        Uxx = Uxx[:,mask][mask]

        # shortcuts
        A = self.a_grid
        B = self.b_grid
        SIGMA = self.sigma_y_grid
        Uxx_inv = np.linalg.inv(Uxx)

        # calculate likelihood
        # alternatively: functools.partial??? 
        f = lambda a, b, sigma_y : log_likelidhood_a_b_sigma_y_without_Xa(a, b, sigma_y, Xo = xx_observed, UXo_inv = Uxx_inv, Y = yy)
        fv = np.vectorize(f)
        log_likelihood = fv(A, B, SIGMA)

        # actual update
        self.discrete_log_posterior = self.discrete_log_posterior + log_likelihood
        
        # condition log-posterior to avoid numerical issues (i.e. C=-np.inf) from integration
        conditioner = self.discrete_log_posterior.max()
        self.discrete_log_posterior -= conditioner

        # normalize
        a = self.a_range
        b = self.b_range
        sigma = self.sigma_y_range
        C = self.integrate_discrete_log_distribution(self.discrete_log_posterior, axes= [a, b, sigma])
        self.discrete_log_posterior -= C


    def init_informative_prior(self, prior):
        
        # shortcuts
        A = self.a_grid
        B = self.b_grid
        SIGMA = self.sigma_y_grid

        a = self.a_range
        b = self.b_range
        sigma = self.sigma_y_range
        
        discrete_log_prior = norm.logpdf(A, **prior["a"]["params"])
        discrete_log_prior += norm.logpdf(B, **prior["b"]["params"])
        discrete_log_prior += invgamma.logpdf(SIGMA, **prior["sigma_y"]["params"])

        # normalize
        C = self.integrate_discrete_log_distribution(discrete_log_prior, axes = [a, b, sigma])

        return discrete_log_prior - C


    def integrate_discrete_log_distribution(self, discrete_log_distribution, axes, return_log=True):
        # axes = a_range, b_range, sigma_y_range
        tmp = np.exp(discrete_log_distribution)

        # what to integrate
        axis_offset = 0
        axes_ids = []
        axes_ranges = []
        for i, given_range in enumerate(axes):
            if given_range is not None:
                axes_ids.append(axis_offset)
                axes_ranges.append(given_range)
            else:
                axis_offset += 1
                
        # integrate over all axes in reverse order
        for axis_id, axis_range in zip(axes_ids, axes_ranges):
            tmp = np.trapz(y=tmp, x=axis_range, axis=axis_id)

        if return_log:
            return np.log(tmp)
        else:
            return tmp


    def laplace_approximation_posterior(self):
        # shortcuts
        a = self.a_range
        b = self.b_range
        sigma = self.sigma_y_range

        # a 
        a_log_dist = self.integrate_discrete_log_distribution(self.discrete_log_posterior, axes = [None, b, sigma])
        a_mean, a_std = self.laplace_approximation_1d(a, a_log_dist)

        # b
        b_log_dist = self.integrate_discrete_log_distribution(self.discrete_log_posterior, axes = [a, None, sigma])
        b_mean, b_std = self.laplace_approximation_1d(b, b_log_dist)

        # sigma_y
        sigma_log_dist = self.integrate_discrete_log_distribution(self.discrete_log_posterior, axes = [a, b, None])
        sigma_mean, sigma_std = self.laplace_approximation_1d(sigma, sigma_log_dist)


        laplace_approximation = {
            "a" : {
                "val" : a_mean,
                "val_unc" : a_std,
            },
            "b" : {
                "val" : b_mean, 
                "val_unc" : b_std,
            },
            "sigma_y" : {
                "val" : sigma_mean,
                "val_unc" : sigma_std,
            }
        }
        logging.info(laplace_approximation)
    
        return laplace_approximation


    def update_grid(self, log_threshold=-600, zoom_out=0.2):

        # find bounding box of region that is above threshold (zoom in into relevant parts)
        relevant_part_of_dist = self.discrete_log_posterior > log_threshold
        a_above_limit = np.any(relevant_part_of_dist, axis=(1,2))
        b_above_limit = np.any(relevant_part_of_dist, axis=(0,2))
        sigma_y_above_limit = np.any(relevant_part_of_dist, axis=(0,1))

        a_min_index, a_max_index = np.where(a_above_limit)[0][[0, -1]]
        b_min_index, b_max_index = np.where(b_above_limit)[0][[0, -1]]
        sigma_y_min_index, sigma_y_max_index = np.where(sigma_y_above_limit)[0][[0, -1]]

        # ensure that bounding box never has min==max
        if a_min_index == a_max_index:
            a_min_index = max(0, a_min_index - 2)
            a_max_index = min(len(self.a_range), a_max_index + 2)
        if b_min_index == b_max_index:
            b_min_index = max(0, b_min_index - 2)
            b_max_index = min(len(self.b_range), b_max_index + 2)
        if sigma_y_min_index == sigma_y_max_index:
            sigma_y_min_index = max(0, sigma_y_min_index - 2)
            sigma_y_max_index = min(len(self.sigma_y_range), sigma_y_max_index + 2)
        
        # actual boundaries of bounding box
        a_min_box, a_max_box = self.a_range[[a_min_index, a_max_index]]
        b_min_box, b_max_box = self.b_range[[b_min_index, b_max_index]]
        log_sigma_y_min_box, log_sigma_y_max_box = np.log10(self.sigma_y_range[[sigma_y_min_index, sigma_y_max_index]])

        # make box bigger in every direction (zoom out to provide room for updates)
        da = min(0.5, a_max_box - a_min_box)
        db = min(0.5, b_max_box - b_min_box)
        log_dsigma_y = min(0.5, log_sigma_y_max_box - log_sigma_y_min_box)
        zoom_out = 1.0

        a_min_new, a_max_new = a_min_box - da * zoom_out, a_max_box + da * zoom_out
        b_min_new, b_max_new = b_min_box - db * zoom_out, b_max_box + db * zoom_out
        log_sigma_y_min_new, log_sigma_y_max_new = log_sigma_y_min_box - log_dsigma_y * zoom_out, log_sigma_y_max_box + log_dsigma_y * zoom_out

        # generate new grid
        a_range_new = np.linspace(a_min_new, a_max_new, num=self.grid_resolution)
        b_range_new = np.linspace(b_min_new, b_max_new, num=self.grid_resolution)
        sigma_y_range_new = np.logspace(log_sigma_y_min_new, log_sigma_y_max_new, num=self.grid_resolution)

        logging.info(a_range_new)
        logging.info(b_range_new)
        logging.info(sigma_y_range_new)

        a_grid_new, b_grid_new, sigma_y_grid_new = np.meshgrid(self.a_range, self.b_range, self.sigma_y_range)

        # interpolate old distrubtion onto new grid
        # (need to turn grid into list of points first)
        points = np.vstack([self.a_grid.ravel(), self.b_grid.ravel(), self.sigma_y_grid.ravel()])
        values = self.discrete_log_posterior.ravel()
        fill_value = log_threshold #values[np.logical_not(np.isinf(values))].min()
        interp_dist = LinearNDInterpolator(points.T, values, fill_value = fill_value)
        discrete_log_posterior_new = interp_dist(a_grid_new, b_grid_new, sigma_y_grid_new)

        # update variables
        self.a_range = a_range_new
        self.b_range = b_range_new
        self.sigma_y_range = sigma_y_range_new

        self.a_grid = a_grid_new
        self.b_grid = b_grid_new
        self.sigma_y_grid = sigma_y_grid_new

        self.discrete_log_posterior = discrete_log_posterior_new


    def plot_discrete_distribution(self, discrete_distribution):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        p = ax.scatter(self.a_grid, self.b_grid, np.log10(self.sigma_y_grid), c=discrete_distribution)
        fig.colorbar(p)
        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.set_zlabel("sigma_y")
        fig.colorbar()
        plt.show()
