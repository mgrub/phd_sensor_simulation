import glob
import json
import os
import numpy


scenarios = [
    "01a_static_input", 
    "01b_sinusoidal_input", 
    "01c_jumping_input", 
    "02a_static_input_noisy", 
    "02b_sinusoidal_input_noisy", 
    "02c_jumping_input_noisy", 
    "03a_variable_blockwise", 
    "03b_smaller_blocks", 
    "03c_larger_blocks", 
    "04a_dropouts", 
    "04b_outliers", 
    "05a_better_references", 
    "05b_equal_references", 
    "05c_worse_references", 
]

methods_to_include = [
    "gibbs_base", 
    "gibbs_minimal", 
    "gibbs_no_EIV", 
    "gibbs_known_sigma_y", 
    "joint_posterior", 
    "joint_posterior_agrid", 
    "stankovic_base", 
    "stankovic_enhanced_unc", 
    "stankovic_base_unc", 
]

# where to search
working_directory = os.path.join(os.getcwd())


for scenario in scenarios:
    print(scenario)

    metrics_path = os.path.join(working_directory, "experiments", scenario, "metrics.json")
    if not os.path.exists(metrics_path):
        print("")
        continue
    metrics = json.load(open(metrics_path, "r"))

    for method in methods_to_include:

        if method not in metrics.keys():
            print("    ", method, "not found")