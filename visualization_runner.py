import argparse
import datetime
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from models import LinearAffineModel

# provide CLI that accepts a configuration file
parser = argparse.ArgumentParser("Visualize the results of an evaluated cocalibration scenario.")
parser.add_argument(
    "scenario",
    nargs="?",
    type=str,
    default="experiments/scenario_A/",
    help="Path to a scenario",
)
parser.add_argument("--show", action='store_true', help="Show plot, rather than generating PNGs")
parser.add_argument("--novis", action='store_true', help="skip visualization")
args = parser.parse_args()

#######################################
# load files ##########################
#######################################

# where to search
working_directory = os.path.join(args.scenario)
results_directory = os.path.join(working_directory, "results")

logfile_path = os.path.join(working_directory, "log.txt")
device_under_test_path = os.path.join(working_directory, "device_under_test.json")
cocalibration_path = os.path.join(working_directory, "cocalibration.json")
measurand_path = os.path.join(working_directory, "measurand.json")
sensor_readings_path = os.path.join(working_directory, "sensor_readings.json")
method_results_paths = glob.glob(os.path.join(results_directory, f"*.json"))

# check if any results available
if not any(method_results_paths):
    raise FileNotFoundError(f"Path <{args.scenario}> does not store any method's results. Exiting.\n")

# load scenario and result data
logfile = open(logfile_path, "r").readlines()
device_under_test = json.load(open(device_under_test_path, "r"))
device_under_test_name = list(device_under_test.keys())[0]
cocalibration = json.load(open(cocalibration_path, "r"))
measurand = json.load(open(measurand_path, "r"))
sensor_readings = json.load(open(sensor_readings_path, "r"))
results = {}
for method_result_path in method_results_paths:
    method_name = os.path.splitext(os.path.basename(method_result_path))[0]
    results[method_name] = json.load(open(method_result_path, "r"))


if not args.novis:
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


    # true values
    true_dut_model =  device_under_test[device_under_test_name]["hasSimulationModel"]
    a_true = true_dut_model["params"]["a"]
    ua_true = true_dut_model["params"]["ua"]

    b_true = true_dut_model["params"]["b"]
    ub_true = true_dut_model["params"]["ub"]

    params_time = np.array(sensor_readings[device_under_test_name]["time"])
    sigma_y_true = sensor_readings[device_under_test_name]["val_unc"][0]

    # plot true values
    ax_params[0].hlines(a_true, params_time.min(), params_time.max(), colors="k", label="true")
    ax_params[1].hlines(b_true, params_time.min(), params_time.max(), colors="k", label="true")
    ax_params[2].hlines(sigma_y_true, params_time.min(), params_time.max(), color="k", label="true")


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


#######################################
# metrics extraction results ##########
#######################################

metrics = {}

t_measurand = measurand["time"]
v_measurand = measurand["quantity"]

true_dut_model =  device_under_test[device_under_test_name]["hasSimulationModel"]
a_true = true_dut_model["params"]["a"]
ua_true = true_dut_model["params"]["ua"]

b_true = true_dut_model["params"]["b"]
ub_true = true_dut_model["params"]["ub"]

sigma_y_true = np.array(sensor_readings[device_under_test_name]["val_unc"])[-1]

# summarize results for easier access
for i, (method_name, method_result) in enumerate(results.items()):
    print(method_name)

    if method_name not in metrics.keys():
        metrics[method_name] = {}
    
    metrics[method_name]["summary"] = {}
    ms = metrics[method_name]["summary"]

    sigma_y_was_estimated = "sigma_y" in method_result[0][0]["params"].keys()

    ms["timestamp"] = np.array([item[-1]["time"] for item in method_result])[-1]
    ms["a"] =  np.array([item[-1]["params"]["a"]["val"] for item in method_result])[-1]
    ms["ua"] = np.array([item[-1]["params"]["a"]["val_unc"] for item in method_result])[-1]
    ms["b"] =  np.array([item[-1]["params"]["b"]["val"] for item in method_result])[-1]
    ms["ub"] = np.array([item[-1]["params"]["b"]["val_unc"] for item in method_result])[-1]
    if sigma_y_was_estimated:
        ms["sigma"] =  np.array([item[-1]["params"]["sigma_y"]["val"] for item in method_result])[-1]
        ms["usigma"] = np.array([item[-1]["params"]["sigma_y"]["val_unc"] for item in method_result])[-1]
    
    ms["a_true"] = a_true
    ms["b_true"] = b_true
    ms["sigma_y_true"] = sigma_y_true
        

# runtime metric
start_re = re.compile("INFO:root:Starting (\w*) at ([0-9+-:\.T]*)\.")
finish_re = re.compile("INFO:root:Finished (\w*) at ([0-9+-:\.T]*)\.")

log_times = {}
for line in logfile:
    start = re.findall(start_re, line)
    end = re.findall(finish_re, line)

    if start:
        method = start[0][0]
        kind = "start"
        time = datetime.datetime.fromisoformat(start[0][1])

    if end:
        method = end[0][0]
        kind = "end"
        time = datetime.datetime.fromisoformat(end[0][1])

    if start or end:
        if method not in log_times.keys():
            log_times[method] = {}
        
        log_times[method][kind] = time

for method, method_vals in log_times.items():
    if "start" in method_vals.keys() and "end" in method_vals.keys():
        duration = method_vals["end"] - method_vals["start"]
    else:
        duration = datetime.timedelta(seconds=0.0)

    if method not in metrics.keys():
        metrics[method] = {}
    
    metrics[method]["computation_duration"] = duration.total_seconds()


# consistency metrics

for i, (method_name, method_result) in enumerate(results.items()):

    sigma_y_was_estimated = "sigma_y" in method_result[0][0]["params"].keys()

    t = np.array([item[-1]["time"] for item in method_result])
    a =  np.array([item[-1]["params"]["a"]["val"] for item in method_result])
    ua = np.array([item[-1]["params"]["a"]["val_unc"] for item in method_result])
    b =  np.array([item[-1]["params"]["b"]["val"] for item in method_result])
    ub = np.array([item[-1]["params"]["b"]["val_unc"] for item in method_result])
    if sigma_y_was_estimated:
        sigma =  np.array([item[-1]["params"]["sigma_y"]["val"] for item in method_result])
        usigma = np.array([item[-1]["params"]["sigma_y"]["val_unc"] for item in method_result])
    

    metrics[method_name]["consistency"] = {}
    mc = metrics[method_name]["consistency"]

    # parameter estimate consistency metric
    mc["a_normalized_error"] = (a[-1] - a_true) / ua[-1]
    mc["b_normalized_error"] = (b[-1] - b_true) / ub[-1]
    if sigma_y_was_estimated:
        mc["sigma_y_normalized_error"] = (sigma[-1] - sigma_y_true) / usigma[-1]

    # check consistency in terms of output
    mr = method_result[-1][-1]["params"]
    kwargs = {
        "a" : mr["a"]["val"], 
        "ua" : mr["a"]["val_unc"], 
        "b" : mr["b"]["val"], 
        "ub" : mr["b"]["val_unc"], 
    }
    m = LinearAffineModel(**kwargs)
    m_inv = LinearAffineModel(**m.inverse_model_parameters())

    dut_readings = np.array(sensor_readings[device_under_test_name]["val"])
    if sigma_y_was_estimated:
        dut_readings_unc = np.full_like(dut_readings, fill_value=sigma[-1])
    else:
        dut_readings_unc = np.zeros_like(dut_readings)
    
    # readings based on calibrated params
    Xa = np.array(measurand["quantity"])
    Xa_hat, Xa_hat_unc = m_inv.apply(dut_readings, dut_readings_unc)
    normalized_model_error = (Xa_hat - Xa) / Xa_hat_unc
    mc["normalized_model_error_mean"] = np.mean(normalized_model_error)
    mc["normalized_model_error_std"] = np.std(normalized_model_error)
    

# convergence metrics
def convergence_band_after(t0, times, vals, vals_unc = None):
    i = np.argmin(np.abs(times-t0))
    i_max = np.argmax(vals[i:])
    i_min = np.argmin(vals[i:])

    band = vals[i:][i_max] - vals[i:][i_min]
    band_unc = np.sqrt(np.square(vals_unc[i:][i_max]) + np.square(vals_unc[i:][i_min]))

    return band, band_unc

def first_below_threshold(t, u, threshold=0.1):
    indices_below = np.where(u < threshold)[0]
        
    if indices_below.size:
        first_index_below = indices_below[0]
        return t[first_index_below]
    else:
        return np.nan


def stays_below_threshold(t, u, threshold=0.1):

    result = np.nan
    for i in range(u.size):
        if np.all(u[i:] < threshold):
            result = t[i]
            break
    
    return result

for i, (method_name, method_result) in enumerate(results.items()):

    if method_name not in metrics.keys():
        metrics[method_name] = {}
    
    sigma_y_was_estimated = "sigma_y" in method_result[0][0]["params"].keys()

    t = np.array([item[-1]["time"] for item in method_result])
    a =  np.array([item[-1]["params"]["a"]["val"] for item in method_result])
    ua = np.array([item[-1]["params"]["a"]["val_unc"] for item in method_result])
    b =  np.array([item[-1]["params"]["b"]["val"] for item in method_result])
    ub = np.array([item[-1]["params"]["b"]["val_unc"] for item in method_result])
    if sigma_y_was_estimated:
        sigma =  np.array([item[-1]["params"]["sigma_y"]["val"] for item in method_result])
        usigma = np.array([item[-1]["params"]["sigma_y"]["val_unc"] for item in method_result])
    
    # convergence span
    metrics[method_name]["convergence"] = {}
    mc = metrics[method_name]["convergence"]

    mc["a"] = {}
    mc["a"]["band_04s"], mc["a"]["a_band_unc_04s"] = convergence_band_after(4, t, a, ua)
    mc["a"]["band_10s"], mc["a"]["a_band_unc_10s"] = convergence_band_after(10, t, a, ua)
    mc["a"]["band_16s"], mc["a"]["a_band_unc_16s"] = convergence_band_after(16, t, a, ua)

    mc["b"] = {}
    mc["b"]["band_04s"], mc["b"]["band_unc_04"] = convergence_band_after(4, t, b, ub)
    mc["b"]["band_10s"], mc["b"]["band_unc_10"] = convergence_band_after(10, t, b, ub)
    mc["b"]["band_16s"], mc["b"]["band_unc_16"] = convergence_band_after(16, t, b, ub)

    if sigma_y_was_estimated:
        mc["sigma_y"] = {}
        mc["sigma_y"]["band_04s"], mc["sigma_y"]["band_unc_04s"] = convergence_band_after(4, t, sigma, usigma)
        mc["sigma_y"]["band_10s"], mc["sigma_y"]["band_unc_10s"] = convergence_band_after(10, t, sigma, usigma)
        mc["sigma_y"]["band_16s"], mc["sigma_y"]["band_unc_16s"] = convergence_band_after(16, t, sigma, usigma)
    
    # convergence in uncertainty
    mc["a"]["unc_first_below_0.1"] = first_below_threshold(t, ua, 0.1)
    mc["a"]["unc_stays_below_0.1"] = stays_below_threshold(t, ua, 0.1)
    mc["b"]["unc_first_below_0.1"] = first_below_threshold(t, ub, 0.1)
    mc["b"]["unc_stays_below_0.1"] = stays_below_threshold(t, ub, 0.1)
    if sigma_y_was_estimated:
        mc["sigma_y"]["unc_first_below_0.1"] = first_below_threshold(t, usigma, 0.1)
        mc["sigma_y"]["unc_stays_below_0.1"] = stays_below_threshold(t, usigma, 0.1)



# write extracted metrics to file
metrics_results_path = os.path.join(working_directory, "metrics.json")
with open(metrics_results_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("\n")