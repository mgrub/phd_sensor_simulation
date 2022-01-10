import numpy as np

from models import LinearAffineModel


class StankovicMethod:
    def simulate_sensor_reading(self, timestamp, measurand_value, sensors):
        for s in sensors:
            y, uy = s["sensor"].indicated_value(measurand_value)
            y = y + uy * np.random.randn()
            s["buffer_indication"].add(data=[[timestamp, y, uy]])

            x_hat, ux_hat = s["sensor"].estimated_value(y, uy)
            s["buffer_estimation"].add(data=[[timestamp, x_hat, ux_hat]])

    def update_single_sensor(
        self, sensor, neighbors, delay=5, delta=0.001, calc_unc=False, use_unc=False, use_enhanced_model=False
    ):
        # define some shorthand notation
        neighbor_value_estimates = np.squeeze(
            [n["buffer_estimation"].show(1)[2] for n in neighbors]
        )
        neighbor_uncertainty_estimates = np.squeeze(
            [n["buffer_estimation"].show(1)[3] for n in neighbors]
        )
        timestamp = sensor["buffer_estimation"].show(1)[0]
        x_hat = sensor["buffer_estimation"].show(1)[2]
        y, uy = sensor["buffer_indication"].show(1)[2:]

        if len(sensor["buffer_indication"]) > delay:
            model = sensor["sensor"].estimated_compensation_model
            param = model.parameters

            y_delayed = sensor["buffer_indication"].show(delay)[2][0]
            # J = np.sum(np.square(neighbor_value_estimates - x_hat) / neighbor_uncertainty_estimates)
            if use_unc:
                weights = neighbor_uncertainty_estimates / np.linalg.norm(
                    neighbor_uncertainty_estimates
                )
            else:
                weights = np.ones_like(neighbor_value_estimates)

            enhanced_factor = np.ones(2)
            if use_enhanced_model:
                enhanced_factor[0] += delta * np.square(uy) * np.sum(weights) 

            grad_J = np.sum(
                (neighbor_value_estimates - x_hat)
                * np.array([[y_delayed, 1]]).T
                / weights,
                axis=1,
            )

            new_param = param * enhanced_factor + delta * grad_J

            # adjust parameter estimation uncertainties
            if calc_unc:
                C = self.build_derivative(
                    param, neighbor_value_estimates, weights, y, uy, delta, use_enhanced_model
                )
                U = self.build_full_input_uncertainty_matrix(
                    model.parameters_uncertainty, neighbor_uncertainty_estimates, uy
                )
                new_param_unc = C @ U @ C.T
            else:
                new_param_unc = None

            model.set_parameters(
                parameters=new_param, parameters_uncertainty=new_param_unc
            )

            # calculate estimated inverse model
            p_inv, up_inv = model.inverse_model_parameters(separate_unc=True)
            sensor["sensor"].estimated_transfer_model = LinearAffineModel(**p_inv, **up_inv)

            return [timestamp, p_inv, up_inv]
        else:
            return None

    def build_derivative(self, param, neighbor_value_estimates, weights, y, uy, delta, use_enhanced_model):
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
