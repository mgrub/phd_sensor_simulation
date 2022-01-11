import argparse
import datetime
import json
import logging
import os
import sys

import numpy as np

from io_helper import NumpyEncoder
from sensor import generate_sensors

# provide CLI that accepts a configuration file
parser = argparse.ArgumentParser("Run an evaluation based on a configuration file")
parser.add_argument(
    "--config",
    type=str,
    help="Path to configuration file",
    default="experiments/scenario_A/config.json",
)
args = parser.parse_args()

# load configuration
if not os.path.exists(args.config):
    raise FileNotFoundError(f"File <{args.config}> does not exist. Exiting.\n")

working_directory = os.path.dirname(args.config)
with open(args.config, "r") as config_file:
    config = json.load(config_file)


# init logger
log_path = os.path.join(working_directory, "log.txt")
logging.basicConfig(filename=log_path, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(f"Start: {datetime.datetime.now().isoformat()}")


# random state
path_or_config = config["random_state"]
if isinstance(path_or_config, str):
    with open(path_or_config, "r") as f:
        random_state = json.load(f)
        np.random.set_state(random_state)
        logging.info(f"Read random state from {path_or_config}.")
else:
    random_state = np.random.get_state()

random_state_path = os.path.join(working_directory, "random_state.json")
with open(random_state_path, "w") as f:
    json.dump(random_state, f, cls=NumpyEncoder, indent=4)


# reference sensors
path_or_config = config["reference_sensors"]
if isinstance(path_or_config, str):
    with open(path_or_config, "r") as f:
        reference_sensors = json.load(f)
        logging.info(f"Read reference sensors from {path_or_config}.")
else:
    reference_sensors = generate_sensors(
        type=path_or_config["type"],
        args=path_or_config["args"],
        draw=path_or_config["draw"],
        n=path_or_config["number"],
    )

reference_sensors_path = os.path.join(working_directory, "reference_sensors.json")
with open(reference_sensors_path, "w") as f:
    json.dump(reference_sensors, f, cls=NumpyEncoder, indent=4)


# device under test


# measurand


# sensor readings


# run cocalibration methods


# clean up
logging.info(f"End: {datetime.datetime.now().isoformat()}")
logging.info("=" * 30)
