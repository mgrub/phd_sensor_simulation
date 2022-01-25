import argparse
import json
import os
import sys


# provide CLI that accepts a configuration file
parser = argparse.ArgumentParser("Visualize the results of an evaluated cocalibration scenario.")
parser.add_argument(
    "--scenario",
    type=str,
    help="Path to a scenario",
    default="experiments/scenario_A/",
)
args = parser.parse_args()

# 
working_directory = os.path.join(args.config)
result_directory

if not os.path.exists(working_directory):
    raise FileNotFoundError(f"Path <{args.config}> does not store any 'results'. Exiting.\n")

