"""
Implementation of Stankovic 2018 for further comparison.
"""

from stankovic import StankovicMethod
import matplotlib.pyplot as plt
import numpy as np
from time_series_buffer import TimeSeriesBuffer

from models import LinearAffineModel
from base import PhysicalPhenomenon, Sensor, SimulationHelper, DeterministicPhysicalPhenomenon

n_neighbors = 5
maxlen = 1000
delta = 0.001
delay = 5

sh = SimulationHelper()

# init "existing" neighbors and new sensors
sensor_config = sh.generate_sensors(n_neighbors = n_neighbors)
neighbors, new_neighbors = sh.init_sensors(jsonstring=sensor_config, maxlen=maxlen)

# init a buffer to store estimated parameters of new sensor(s)
buffer_parameters = TimeSeriesBuffer(maxlen=maxlen, return_type="list")

# init physical phenomenon that will be observed by sensors
#pp = DeterministicPhysicalPhenomenon(static_omega=False)
pp = PhysicalPhenomenon(sigma_x=0.01)
pp_buffer = TimeSeriesBuffer(maxlen=maxlen, return_type="arrays")

# init stankovic method
sm = StankovicMethod()
 
# observe the signal for some time
time = np.linspace(0, 40, 1000)
for t in time:
    # actual value of physical phenomenon
    real_value = pp.value(t)
    pp_buffer.add(data=[[t, real_value]])

    # simulate sensor readings
    sm.simulate_sensor_reading(t, np.array([real_value]), sensors = neighbors + new_neighbors)

    # adjust/compensate new sensor by adjusting its parameters based on gradient
    estimated_inverse_params = sm.update_single_sensor(new_neighbors[0], neighbors, delta=delta, calc_unc=True, use_unc=True, use_enhanced_model=True)
    
    # store estimated new parameters for later visualization
    if estimated_inverse_params is not None:
        buffer_parameters.add(data=[estimated_inverse_params])


# plotting
tp, _, vp, _ = pp_buffer.show(-1)

fig, ax = plt.subplots(nrows=2, sharex=True)

ax[0].plot(tp, vp, label="ground truth")
for i, n in enumerate(neighbors + new_neighbors):
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

par = np.squeeze(par)

ax[1].plot(t_par, par, label="estimates")
ax[1].errorbar(t_par, par[:,0], yerr=upar[:,0], errorevery=10, capsize=3, label="a")
ax[1].errorbar(t_par, par[:,1], yerr=upar[:,1], errorevery=10, capsize=3, label="b")
ax[1].hlines(new_neighbors[0]["sensor"].transfer_model.parameters, tp.min(), tp.max())
ax[1].legend()

plt.show()
