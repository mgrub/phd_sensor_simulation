import argparse
import datetime
import json
import logging
import os
import sys

import numpy as np

from io_helper import NumpyEncoder
from measurands import return_timestamps, return_measurand_object
from sensor import generate_sensor, generate_sensors

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

# TODO: get git status information for log

# TODO: get pip status information for log


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
        reference_sensors_description = json.load(f)
        logging.info(f"Read reference sensors from {path_or_config}.")
else:
    reference_sensors_description = generate_sensors(
        type=path_or_config["type"],
        args=path_or_config["args"],
        draw=path_or_config["draw"],
        n=path_or_config["number"],
    )
#reference_sensors = init_from(reference_sensors_description)

reference_sensors_path = os.path.join(working_directory, "reference_sensors.json")
with open(reference_sensors_path, "w") as f:
    json.dump(reference_sensors_description, f, cls=NumpyEncoder, indent=4)


# device under test
path_or_config = config["device_under_test"]
if isinstance(path_or_config, str):
    with open(path_or_config, "r") as f:
        device_under_test_description = json.load(f)
        logging.info(f"Read device under test from {path_or_config}.")
else:
    device_under_test_description = generate_sensor(
        type=path_or_config["type"],
        args=path_or_config["args"],
        draw=path_or_config["draw"],
    )
#device_under_test = init_from(device_under_test_description)

device_under_test_path = os.path.join(working_directory, "device_under_test.json")
with open(device_under_test_path, "w") as f:
    json.dump(device_under_test_description, f, cls=NumpyEncoder, indent=4)


# measurand
path_or_config = config["measurand"]
if isinstance(path_or_config, str):
    with open(path_or_config, "r") as f:
        measurand = json.load(f)
        logging.info(f"Read device under test from {path_or_config}.")
else:
    m = return_measurand_object(path_or_config["type"], path_or_config["args"])
    time = return_timestamps(**path_or_config["time_args"])
    measurand = m.value(time)

measurand_path = os.path.join(working_directory, "measurand.json")
with open(measurand_path, "w") as f:
    json.dump(measurand, f, cls=NumpyEncoder, indent=4)


# sensor readings


# run cocalibration methods


# clean up
logging.info(f"End: {datetime.datetime.now().isoformat()}")
logging.info("=" * 30)
