import numpy as np


class ParametricModel:
    """Generic parametric model class

    Allows to define a structure with parametrization, that can then be applied to some input(-signal).
    """

    def __init__(
        self, evaluation_function, parameters, parameters_uncertainty, state=None
    ):
        self.evaluation_function = evaluation_function
        self.parameters = parameters
        self.parameters_uncertainty = parameters_uncertainty
        self.state = state

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
    def __init__(self, a=1.0, b=0.0, ua=0.0, ub=0.0, uab=0.0):
        """Initialize linear affine model with uncorrelated uncertainty evaluation
        y = a * x + b
        uy = sqrt(C * U * C^T)

        with
        C = [dy/da, dy/db, dy/dx]^T = [x, 1, a]^T
        U = [[ua^2, uab,  0   ]
             [uab,  ub^2, 0   ]
             [0,    0,    ux^2]]

        :param a: gain
        :type a: float
        :param b: offset
        :type b: float
        :param ua: gain uncertainty, defaults to 0.0
        :type ua: float, optional
        :param ub: offset uncertainty, defaults to 0.0
        :type ub: float, optional
        :param uab: correlated uncertainty between a and b, defaults to 0.0
        :type ub: float, optional
        """

        evaluation_function = self.equation
        parameter = [a, b]
        parameters_uncertainty = np.array(
            [[np.square(ua), uab], [uab, np.square(ub)]]
        )  # covariance matrix

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
        ua2 = up[0, 0]
        ub2 = up[1, 1]
        uab = up[0, 1]

        # sensitivities
        C = np.array([x, 1, a])

        # uncertainty matrix
        U = np.array([[ua2, uab, 0], [uab, ub2, 0], [0, 0, np.square(ux)]])

        # output equation
        val = a * x + b
        unc = np.sqrt(C @ U @ C.T)
        state = None

        return val, unc, state

    def inverse_model_parameters(self):
        # shortcuts
        a = self.parameters[0]
        b = self.parameters[1]
        ua2 = self.parameters_uncertainty[0, 0]
        ub2 = self.parameters_uncertainty[1, 1]
        uab = self.parameters_uncertainty[0, 1]

        # inverse (unc according GUM)
        a_inv = 1 / a
        b_inv = -b / a

        C = np.array([[-1 / np.square(a), 0], [b / np.square(a), -1 / a]])
        U = np.array([[ua2, uab], [uab, ub2]])
        U_inv = C @ U @ C.T

        # pack result
        ua_inv = np.sqrt(U_inv[0, 0])
        ub_inv = np.sqrt(U_inv[1, 1])
        uab_inv = U_inv[0, 1]
        parameters_inv = {"a": a_inv, "b": b_inv}
        parameters_uncertainty_inv = {"ua": ua_inv, "ub": ub_inv, "uab": uab_inv}

        return parameters_inv, parameters_uncertainty_inv


class LinearTimeInvariantModel(ParametricModel):
    pass
