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
inverse = LinearAffineModel()
sensor = Sensor(transfer_model=transfer, estimated_compensation_model=inverse)
buffer_indication = TimeSeriesBuffer(maxlen=maxlen, return_type="arrays")
buffer_estimation = TimeSeriesBuffer(maxlen=maxlen, return_type="arrays")
new_neighbor = {
    "sensor": sensor,
    "buffer_indication": buffer_indication,
    "buffer_estimation": buffer_estimation,
}

# init physical phenomenon that will be observed by sensors
pp = PhysicalPhenomenon()
pp_buffer = TimeSeriesBuffer(maxlen=maxlen, return_type="arrays")

# observe the signal for some time
time = np.linspace(0, 20, 1000)
for t in time:
    # actual value of physical phenomenon
    real_value = pp.value(t)
    pp_buffer.add(data=[[t, real_value]])

    # simulate sensor readings
    for n in neighbors + [new_neighbor]:
        y, uy = n["sensor"].indicated_value(real_value)
        y = y + uy * np.random.randn()
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
    J = np.sum(
        np.square(neighbor_value_estimates - x_hat) / neighbor_uncertainty_estimates
    )
    grad_J = np.sum(
        (neighbor_value_estimates - x_hat)
        * np.array([[y, 1]]).T
        / neighbor_uncertainty_estimates,
        axis=1,
    )
    model = new_neighbor["sensor"].estimated_compensation_model
    param = model.parameters
    new_param = param + 0.001 * grad_J
    print(param, J)

    # todo: adjust parameter estimation uncertainties

    model.set_parameters(parameters = new_param)


# plotting
tp, _, vp, _ = pp_buffer.show(-1)
plt.plot(tp, vp, label="ground truth")
for i, n in enumerate(neighbors + [new_neighbor]):
    tn, utn, vn, uvn = n["buffer_estimation"].show(-1)
    plt.errorbar(
        tn, vn, xerr=utn, yerr=uvn, errorevery=10, capsize=3, label=str(i)
    )  # , ecolor="grey")

plt.legend()
plt.show()
