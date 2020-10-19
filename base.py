import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scs
from time_series_buffer import TimeSeriesBuffer as buf
from time_series_metadata.scheme import MetaData as meta


class PhysicalPhenomenon:
    def __init__(self):

        # build the base chirp signal
        times = np.linspace(0, 5, 200)
        chirp = scs.chirp(times, f0=1, t1=5, f1=3)

        self.base_signal = 0.1 * np.concatenate((chirp, chirp[::-1]))
        self.counter = 0
        self.p_new_offset = 0.02
        self.offset = 0

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


class Sensor:
    def __init__(self, transfer_model, estimated_transfer_model=None, estimated_compensation_model=None):
        self.transfer_model = transfer_model  # simulation model
        self.estimated_transfer_model = estimated_transfer_model  # calibration model
        self.estimated_compensation_model = estimated_compensation_model  # compensation model

    def indicated_value(self, physical_phenomenon_value):
        value, value_unc = self.transfer_model.apply(physical_phenomenon_value, 0)
        return value, value_unc

    def estimated_value(self, indicated_value, indicated_uncertainty):
        value, value_unc = self.estimated_compensation_model.apply(
            indicated_value, indicated_uncertainty
        )
        return value, value_unc


class ParametricModel:
    """Generic parametric model class

    Allows to define a structure with parametrization, that can then be applied to some input(-signal).
    """

    def __init__(self, evaluation_function, parameters, parameters_uncertainty):
        self.evaluation_function = evaluation_function
        self.parameters = parameters
        self.parameters_uncertainty = parameters_uncertainty
        self.state = None

    def apply(self, input, input_unc):
        output, output_uncertainty, self.state = self.evaluation_function(
            input, input_unc, self.parameters, self.parameters_uncertainty
        )
        return output, output_uncertainty

    def inverse_model_parameters(self):
        return None
    
    def set_parameters(self, parameters=None, parameters_uncertainty=None):
        if parameters is not None:
            self.parameters = parameters
        
        if parameters_uncertainty is not None:
            self.parameters_uncertainty = parameters_uncertainty

class LinearAffineModel(ParametricModel):
    def __init__(self, a=1.0, b=0.0, ua=0.0, ub=0.0):
        """Initialize linear affine model with uncorrelated uncertainty evaluation
        y = a * x + b
        uy = sqrt(x^2 * ua^2 + a^2 * ux^2 + ub^2)

        :param a: gain
        :type a: float
        :param b: offset
        :type b: float
        :param ua: gain uncertainty, defaults to 0.0
        :type ua: float, optional
        :param ub: offset uncertainty, defaults to 0.0
        :type ub: float, optional
        """

        evaluation_function = self.equation
        parameter = [a, b]
        parameters_uncertainty = [ua, ub]

        super().__init__(evaluation_function, parameter, parameters_uncertainty)
    
    def __repr__(self):
        # shortcuts
        a = self.parameters[0]
        b = self.parameters[1]

        s = f"<LinearAffineModel: {a} * x + {b}>"
        return s

    def equation(self, x, ux, p, up):
        # shortcuts
        a = p[0]
        b = p[1]
        ua = up[0]
        ub = up[1]

        # output equation
        val = a * x + b
        unc = np.sqrt(np.square(x * ua) + np.square(a * ux) + np.square(ub))
        state = None

        return val, unc, state

    def inverse_model_parameters(self):
        # shortcuts
        a = self.parameters[0]
        b = self.parameters[1]
        ua = self.parameters_uncertainty[0]
        ub = self.parameters_uncertainty[1]

        # inverse (unc according GUM)
        a_inv = 1 / a
        b_inv = -b / a
        ua_inv = ua / np.square(a)
        ub_inv = np.sqrt(
            1 / np.square(a) * np.square(ub)
            + np.square(b / np.square(a)) * np.square(ua)
        )

        # pack result
        parameters_inv = {"a": a_inv, "b": b_inv}
        parameters_uncertainty_inv = {"ua": ua_inv, "ub": ub_inv}

        return parameters_inv, parameters_uncertainty_inv


class LinearTimeInvariantModel(ParametricModel):
    pass

