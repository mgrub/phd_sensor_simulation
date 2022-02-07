import argparse
import glob
import json
import os
import sys
from unittest import result
import matplotlib.pyplot as plt
import numpy as np

# provide CLI that accepts a configuration file
parser = argparse.ArgumentParser("Visualize the results of an evaluated cocalibration scenario.")
parser.add_argument(
    "--scenario",
    type=str,
    help="Path to a scenario",
    default="experiments/scenario_A/",
)
args = parser.parse_args()

#######################################
# load files ##########################
#######################################

# where to search
working_directory = os.path.join(args.scenario)
results_directory = os.path.join(working_directory, "results")

device_under_test_path = os.path.join(working_directory, "device_under_test.json")
cocalibration_path = os.path.join(working_directory, "cocalibration.json")
measurand_path = os.path.join(working_directory, "measurand.json")
sensor_readings_path = os.path.join(working_directory, "sensor_readings.json")
method_results_paths = glob.glob(os.path.join(results_directory, f"*.json"))

# check if any results available
if not os.path.exists(results_directory):
    raise FileNotFoundError(f"Path <{args.config}> does not store any 'results'. Exiting.\n")


# load scenario and result data
device_under_test = json.load(open(device_under_test_path, "r"))
device_under_test_name = list(device_under_test.keys())[0]
cocalibration = json.load(open(cocalibration_path, "r"))
measurand = json.load(open(measurand_path, "r"))
sensor_readings = json.load(open(sensor_readings_path, "r"))
results = {}
for method_result_path in method_results_paths:
    method_name = os.path.splitext(os.path.basename(method_result_path))[0]
    results[method_name] = json.load(open(method_result_path, "r"))

#######################################
# visualize reference readings ########
#######################################

# colors to cycle through
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig_ref, ax_ref = plt.subplots(nrows=2, sharex=True, sharey=True)

# true measurand
t = measurand["time"]
v = measurand["quantity"]

ax_ref[0].plot(t, v, linewidth=3, label="measurand (unknown)", marker="o", color="red", zorder=20)

# sensor readings + unc-tube
for i, (sensor_name, sensor_reading) in enumerate(sensor_readings.items()):
    # plot device under test into separate axis
    if sensor_name == device_under_test_name:
        ax = ax_ref[1]
    else:
        ax = ax_ref[0]
    
    # choose color
    color = colors[i % len(colors)]

    # shortcuts
    t = np.array(sensor_reading["time"])
    v = np.array(sensor_reading["val"])
    uv = np.array(sensor_reading["val_unc"])

    # plot blocks if available, otherwise just plot whole timeseries
    if "split_indices" in cocalibration.keys():
        s_old = 0

        for i, split_index in enumerate(cocalibration["split_indices"] + [len(t)]):
            if i == 0:
                label = f"{sensor_name} (blocks)"
            else:
                label = None

            # plot block-sequence
            tt = t[s_old:split_index]
            vv = v[s_old:split_index]
            uvv = uv[s_old:split_index]
            ax.plot(tt, vv, label=label, color=color, marker="o", markersize=2)
            ax.fill_between(tt, vv-uvv, vv+uvv, alpha=0.3, color=color)

            # next block
            s_old = split_index

    else:
        # plot value + unc-tube
        ax.plot(t, v, label=sensor_name, color=color, marker="o", markersize=2)
        ax.fill_between(t, v-uv, v+uv, alpha=0.3, color=color)

ax_ref[0].legend()
ax_ref[1].legend()

# fused value?

#######################################
# visualize method results ############
#######################################

fig_params, ax_params = plt.subplots()

# true values

# for every method

    # estimates + unc-tube

    # unc separate


plt.show()