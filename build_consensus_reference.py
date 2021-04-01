import random

import matplotlib.pyplot as plt
import numpy as np

from base import DeterministicPhysicalPhenomenon, Sensor, SimulationHelper
from models import LinearAffineModel

dpp = DeterministicPhysicalPhenomenon()
sh = SimulationHelper()

# init "existing" sensors
sensor_config = sh.generate_sensors(n_neighbors = 5)
sensors, _ = sh.init_sensors(jsonstring=sensor_config, maxlen=None)

#
time_limits = [10, 130]  # s
freqs = [0.1, 1, 2, 5, 10, 50, 100]  # Hz
time_unc = 5e-3  # e-3 -> ms


# plot ground truth signal for reference
time_ground_truth = np.arange(start=time_limits[0], stop=time_limits[1], step=1/max(freqs))
ground_truth = dpp.value(time_ground_truth)
plt.plot(time_ground_truth, ground_truth, "--k")

# generate time signal of every sensor

for sensor in sensors:
    s = sensor["sensor"]

    dt = 1.0 / random.choice(freqs)
    time_len = int((time_limits[1] - time_limits[0])//dt + 1)
    time = time_limits[0] + np.cumsum(np.random.normal(loc=dt, scale=time_unc, size=time_len))

    ground_truth = dpp.value(time)
    indication, indication_unc = s.indicated_value(ground_truth)
    estimate, estimate_unc = s.estimated_value(indication, indication_unc)


    #plt.plot(time, estimate, "k")
    #plt.fill_between(time, y1=estimate-estimate_unc, y2=estimate+estimate_unc, color="k", alpha=0.3)
    plt.errorbar(time, estimate, estimate_unc, xerr=time_unc, fmt="o", capsize=2, linewidth=0, elinewidth=1)


plt.show()
