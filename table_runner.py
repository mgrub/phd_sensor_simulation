import argparse
import glob
import json
import os
import sys
from unittest import result
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# provide CLI that accepts a configuration file
parser = argparse.ArgumentParser(
    "Visualize the results of an evaluated cocalibration scenario."
)
parser.add_argument(
    "scenario",
    nargs="?",
    type=str,
    default="experiments/07_thesis_example/",  # "experiments/scenario_A/"
    help="Path to a scenario",
)
parser.add_argument(
    "--show",
    action="store_true",
    help="Show plot, rather than generating PNGs",
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
    raise FileNotFoundError(
        f"Path <{args.scenario}> does not store any method's results. Exiting.\n"
    )


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
# extract info ########################
#######################################

dut_name = list(device_under_test.keys())[0]
device_under_test[dut_name]

dut_sim_model = device_under_test[dut_name]["hasSimulationModel"]
dut_sim_params = dut_sim_model["params"]

subcats = [
    200,
    500,
    1900,
]
# relate subcats to actual available timestamps
t_measurand = np.array(measurand["time"])
t_avail = np.array([item[-1]["time"] for item in results[list(results.keys())[0]]])
t_wanted = t_measurand[subcats]
t_closest_index = np.searchsorted(t_avail, t_wanted)
print(t_wanted, t_avail[t_closest_index])
subcats = np.searchsorted(t_measurand, t_avail[t_closest_index])

row_iter = [results.keys(), subcats]
row_names = ["method", "estimate with n datapoints"]
row_index = pd.MultiIndex.from_product(row_iter, names=row_names)
row_index = row_index.insert(0, ("true", ""))
col_index = pd.Index(["a", "b", "sigma\\_y"], name="parameter")

# create+fill data frame
df = pd.DataFrame(index=row_index, columns=col_index)
format = ".2e"
np.set_printoptions(precision=2)

for method_name in results.keys():
    for subcat, i in zip(subcats, t_closest_index):
        p = results[method_name][i][-1]["params"]
        a = f"{p['a']['val']:{format}} ± {p['a']['val_unc']:{format}}"
        b = f"{p['b']['val']:{format}} ± {p['b']['val_unc']:{format}}"
        sigma_y = (
            f"{p['sigma_y']['val']:{format}} ± {p['sigma_y']['val_unc']:{format}}"
            if "sigma_y" in p.keys()
            else ""
        )

        df.loc[(method_name, subcat)] = [a, b, sigma_y]

df.loc[("true", "")] = [
    f"{dut_sim_params['a']:{format}}",
    f"{dut_sim_params['b']:{format}}",
    "",
]

# provide LaTeX of dataframe

# overwrite previous index with latex-compatible syntax
latex_col_labels = [
    "\\texttt{" + key.replace("_", "\\_") + "}" for key in results.keys()
]
new_index = pd.MultiIndex.from_product([latex_col_labels, subcats], names=row_names)
new_index = new_index.insert(0, ("true", ""))
df.index = new_index

# generate table
print(df.style.to_latex())  # column_format=">{\ttfamily}lllll"
