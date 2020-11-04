"""
Implementation of Stankovic 2018 for further comparison.
"""

import matplotlib.pyplot as plt
import numpy as np
from time_series_buffer import TimeSeriesBuffer

from base import LinearAffineModel, PhysicalPhenomenon, Sensor

# init "existing" already calibrated sensors
n_neighbors = 5
maxlen = 1000
neighbors = []
delta = 0.0001


# WIP: these functions should probably be defined somewhere else inside base?
def build_derivative(param, neighbor_value_estimates, weights, y, delta):
    nn = len(neighbor_value_estimates)
    shape = (2, 2 + nn + 1)
    C = np.zeros(shape)
    a = param[0]
    b = param[1]

    # derivatives
    C[0, 0] = 1 + delta * np.sum(weights * (-np.square(y)))
    C[0, 1] = 0 + delta * np.sum(weights * (-y))
    C[0, 2 : 2 + nn] = 0 + delta * weights * y
    C[0, -1] = 0 + delta * np.sum(weights * (-2 * a * y + neighbor_value_estimates - b))

    C[1, 0] = 0 + delta * np.sum(weights * (-y))
    C[1, 1] = 1 + delta * np.sum(weights * (-1))
    C[1, 2 : 2 + nn] = 0 + delta * weights
    C[1, -1] = 0 + delta * np.sum(weights * (-a))

    return C


def build_full_input_uncertainty_matrix(param_unc, neighbor_uncertainty_estimates, uy):
    main_diag = np.hstack((np.diag(param_unc), neighbor_uncertainty_estimates, uy))
    U = np.diag(main_diag)
    U[:2, :2] = param_unc

    return U


for i in range(n_neighbors):

    # select random model parameters
    a = 1 + 0.5 * np.random.randn()
    b = 5 + 0.5 * np.random.randn()
    ua = 0.1 * (1 + np.random.random())
    ub = 0.1 * (1 + np.random.random())
    transfer = LinearAffineModel(a=a, b=b, ua=ua, ub=ub)

    # get inverse model
    p, up = transfer.inverse_model_parameters()
    inverse = LinearAffineModel(**p, **up)

    # init sensor
    sensor = Sensor(transfer_model=transfer, estimated_compensation_model=inverse)
    buffer_indication = TimeSeriesBuffer(maxlen=maxlen, return_type="arrays")
    buffer_estimation = TimeSeriesBuffer(maxlen=maxlen, return_type="arrays")
    neighbors.append(
        {
            "sensor": sensor,
            "buffer_indication": buffer_indication,
            "buffer_estimation": buffer_estimation,
        }
    )

# introduce new sensor
transfer = LinearAffineModel(a=1.2, b=7.7)
inverse = LinearAffineModel(a=1, b=-5, ua=1.0, ub=1.0, uab=0)
sensor = Sensor(transfer_model=transfer, estimated_compensation_model=inverse)
buffer_indication = TimeSeriesBuffer(maxlen=maxlen, return_type="arrays")
buffer_estimation = TimeSeriesBuffer(maxlen=maxlen, return_type="arrays")
new_neighbor = {
    "sensor": sensor,
    "buffer_indication": buffer_indication,
    "buffer_estimation": buffer_estimation,
}
buffer_parameters = TimeSeriesBuffer(maxlen=maxlen, return_type="list")

# init physical phenomenon that will be observed by sensors
pp = PhysicalPhenomenon()
pp_buffer = TimeSeriesBuffer(maxlen=maxlen, return_type="arrays")

# observe the signal for some time
time = np.linspace(0, 20, 500)
for t in time:
    # actual value of physical phenomenon
    real_value = pp.value(t)
    pp_buffer.add(data=[[t, real_value]])

    # simulate sensor readings
    for n in neighbors + [new_neighbor]:
        y, uy = n["sensor"].indicated_value(real_value)
        # y = y + uy * np.random.randn()
        n["buffer_indication"].add(data=[[t, y, uy]])

        x_hat, ux_hat = n["sensor"].estimated_value(y, uy)
        n["buffer_estimation"].add(data=[[t, x_hat, ux_hat]])

    # adjust/compensate new sensor by adjusting its parameters based on gradient
    neighbor_value_estimates = np.squeeze(
        [n["buffer_estimation"].show(1)[2] for n in neighbors]
    )
    neighbor_uncertainty_estimates = np.squeeze(
        [n["buffer_estimation"].show(1)[3] for n in neighbors]
    )

    delay = 5
    if len(new_neighbor["buffer_indication"]) > delay:
        y_delayed = new_neighbor["buffer_indication"].show(delay)[2][0]
        # J = np.sum(np.square(neighbor_value_estimates - x_hat) / neighbor_uncertainty_estimates)
        weights = neighbor_uncertainty_estimates / np.linalg.norm(
            neighbor_uncertainty_estimates
        )
        grad_J = np.sum(
            (neighbor_value_estimates - x_hat) * np.array([[y_delayed, 1]]).T / weights,
            axis=1,
        )
        model = new_neighbor["sensor"].estimated_compensation_model
        param = model.parameters
        new_param = param + delta * grad_J

        # adjust parameter estimation uncertainties
        C = build_derivative(
            param, neighbor_value_estimates, weights, y, delta
        )
        U = build_full_input_uncertainty_matrix(
            model.parameters_uncertainty, neighbor_uncertainty_estimates, uy
        )
        new_param_unc = C @ U @ C.T
        # print(param, J)

        model.set_parameters(parameters=new_param, parameters_uncertainty=new_param_unc)

        # calculate estimated inverse model
        p_inv, up_inv = model.inverse_model_parameters()
        new_neighbor["sensor"].estimated_transfer_model = LinearAffineModel(
            **p_inv, **up_inv
        )
        # print(new_neighbor["sensor"].estimated_transfer_model.parameters)
        buffer_parameters.add(data=[[t, p_inv, up_inv]])


# plotting
tp, _, vp, _ = pp_buffer.show(-1)

fig, ax = plt.subplots(nrows=2, sharex=True)

ax[0].plot(tp, vp, label="ground truth")
for i, n in enumerate(neighbors + [new_neighbor]):
    tn, utn, vn, uvn = n["buffer_estimation"].show(-1)
    ax[0].errorbar(
        tn, vn, xerr=utn, yerr=uvn, errorevery=10, capsize=3, label=str(i)
    )  # , ecolor="grey")
ax[0].legend()

# show parameter development
params_list = buffer_parameters.show(-1)
t_par = [p[0] for p in params_list]
par = np.array([(p[2]["a"], p[2]["b"]) for p in params_list])
upar = np.array([(p[3]["ua"], p[3]["ub"]) for p in params_list])

ax[1].plot(t_par, par, label="estimates")
ax[1].errorbar(t_par, par[:,0], yerr=upar[:,0], errorevery=10, capsize=3, label="a")
ax[1].errorbar(t_par, par[:,1], yerr=upar[:,1], errorevery=10, capsize=3, label="b")
ax[1].hlines(new_neighbor["sensor"].transfer_model.parameters, tp.min(), tp.max())
ax[1].legend()

plt.show()
