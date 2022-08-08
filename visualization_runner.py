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
    "scenario",
    nargs="?",
    type=str,
    default="experiments/scenario_A/",
    help="Path to a scenario",
)
parser.add_argument("--show", action='store_true', help="Show plot, rather than generating PNGs",
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
if not any(method_results_paths):
    raise FileNotFoundError(f"Path <{args.scenario}> does not store any method's results. Exiting.\n")


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

fig_ref, ax_ref = plt.subplots(nrows=2, sharex=True, sharey=False)
fig_ref.set_size_inches(18.5, 10.5)

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
            if sensor_name is not device_under_test_name:
                ax.fill_between(tt, vv-uvv, vv+uvv, alpha=0.3, color=color)

            # next block
            s_old = split_index

    else:
        # plot value + unc-tube
        ax.plot(t, v, label=sensor_name, color=color, marker="o", markersize=2)
        if sensor_name is not device_under_test_name:
            ax.fill_between(t, v-uv, v+uv, alpha=0.3, color=color)

ax_ref[0].legend()
ax_ref[1].legend()

# fused value?


#######################################
# visualize method results ############
#######################################

fig_params, ax_params = plt.subplots(nrows=3)
fig_params_unc, ax_params_unc = plt.subplots(nrows=3)
fig_params.set_size_inches(18.5, 15.5)
fig_params_unc.set_size_inches(18.5, 15.5)
ax_params[0].set_title("parameter a estimates")
ax_params[1].set_title("parameter b estimates")
ax_params[2].set_title("parameter sigma_y estimates")

ax_params_unc[0].set_title("parameter a uncertainties")
ax_params_unc[1].set_title("parameter b uncertainties")
ax_params_unc[2].set_title("parameter sigma_y uncertainties")

true_dut_model =  device_under_test[device_under_test_name]["hasSimulationModel"]

# true values
a_true = true_dut_model["params"]["a"]
ua_true = true_dut_model["params"]["ua"]

b_true = true_dut_model["params"]["b"]
ub_true = true_dut_model["params"]["ub"]

sigma_y_true_time = np.array(sensor_readings[device_under_test_name]["time"])
sigma_y_true = np.array(sensor_readings[device_under_test_name]["val_unc"])

# plot true values
ax_params[0].hlines(a_true, sigma_y_true_time.min(), sigma_y_true_time.max(), colors="k", label="true")
ax_params[1].hlines(b_true, sigma_y_true_time.min(), sigma_y_true_time.max(), colors="k", label="true")
ax_params[2].plot(sigma_y_true_time, sigma_y_true, color="k")


# for every method
for i, (method_name, method_result) in enumerate(results.items()):# choose color

    color = colors[i % len(colors)]
    sigma_y_was_estimated = "sigma_y" in method_result[0][0]["params"].keys()

    t = np.array([item[-1]["time"] for item in method_result])
    a =  np.array([item[-1]["params"]["a"]["val"] for item in method_result])
    ua = np.array([item[-1]["params"]["a"]["val_unc"] for item in method_result])
    b =  np.array([item[-1]["params"]["b"]["val"] for item in method_result])
    ub = np.array([item[-1]["params"]["b"]["val_unc"] for item in method_result])
    if sigma_y_was_estimated:
        sigma =  np.array([item[-1]["params"]["sigma_y"]["val"] for item in method_result])
        usigma = np.array([item[-1]["params"]["sigma_y"]["val_unc"] for item in method_result])

    # estimates + unc-tube
    ax_params[0].plot(t, a, color=color, marker="o", label=method_name)
    ax_params[0].fill_between(t, a-ua, a+ua, alpha=0.3, color=color)
    
    ax_params[1].plot(t, b, color=color, marker="o", label=method_name)
    ax_params[1].fill_between(t, b-ub, b+ub, alpha=0.3, color=color)
    
    if sigma_y_was_estimated:
        ax_params[2].plot(t, sigma, color=color, marker="o", label=method_name)
        ax_params[2].fill_between(t, sigma-usigma, sigma+usigma, alpha=0.3, color=color)
        #ax_params[4].set_yscale("log")

    # unc separate
    ax_params_unc[0].semilogy(t, ua, color=color, label=method_name)
    ax_params_unc[1].semilogy(t, ub, color=color, label=method_name)
    if sigma_y_was_estimated:
        ax_params_unc[2].semilogy(t, usigma, color=color, label=method_name)

    # sigma_y ???

ax_params[0].legend()
ax_params_unc[0].legend()

if args.show:
    fig_ref.set_tight_layout(True)
    fig_params.set_tight_layout(True)
    fig_params_unc.set_tight_layout(True)
    plt.show()
else:
    fig_ref.savefig(os.path.join(results_directory, "input.png"), bbox_inches="tight")
    fig_params.savefig(os.path.join(results_directory, "params.png"), bbox_inches="tight")
    fig_params_unc.savefig(os.path.join(results_directory, "params_unc.png"), bbox_inches="tight")
