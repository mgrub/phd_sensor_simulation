import argparse
import copy
import datetime
import json
import logging
import os
import platform
import shlex
import subprocess
import sys

import numpy as np
from pip._internal.operations import freeze

from io_helper import NumpyEncoder, split_sensor_readings
from measurands import return_measurand_object, return_timestamps
from sensor import generate_sensor_description, generate_sensor_descriptions, init_sensor_objects
import cocalibration_methods

# provide CLI that accepts a configuration file
parser = argparse.ArgumentParser("Run an evaluation based on a configuration file")
parser.add_argument(
    "--config",
    type=str,
    help="Path to configuration file",
    default="experiments/scenario_B/config.json",
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


# get system + python information
modules = list(freeze.freeze())
logging.info(f"Machine: {platform.node()}")
logging.info(f"OS: {platform.platform()}")
logging.info(f"Processor: {platform.processor()}")
logging.info(f"Python: {platform.python_version()}")
logging.info(f"Python environment: {modules}")


# get git status information for log
script_directory = os.path.dirname(os.path.abspath(__file__))
cmd = shlex.split('git log -1 --format="%H"')
p = subprocess.Popen(cmd, cwd=script_directory, stdout=subprocess.PIPE)
out, err = p.communicate()
logging.info(f"Repository Version: {out.decode('utf-8')}")


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
    reference_sensors_description = generate_sensor_descriptions(**path_or_config)

reference_sensors = init_sensor_objects(reference_sensors_description)

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
    device_under_test_description = generate_sensor_description(**path_or_config)
device_under_test = init_sensor_objects(device_under_test_description)

device_under_test_path = os.path.join(working_directory, "device_under_test.json")
with open(device_under_test_path, "w") as f:
    json.dump(device_under_test_description, f, cls=NumpyEncoder, indent=4)


# measurand
path_or_config = config["measurand"]
if isinstance(path_or_config, str):
    with open(path_or_config, "r") as f:
        measurand = json.load(f)
        logging.info(f"Read measurand from {path_or_config}.")
else:
    m = return_measurand_object(path_or_config["type"], path_or_config["args"])
    time = return_timestamps(**path_or_config["time_args"])
    measurand = m.value(time)

measurand_path = os.path.join(working_directory, "measurand.json")
with open(measurand_path, "w") as f:
    json.dump(measurand, f, cls=NumpyEncoder, indent=4)


# sensor readings (simulate and make blocks in advance)
path_or_config = config["sensor_readings"]
if isinstance(path_or_config, str):
    with open(path_or_config, "r") as f:
        sensor_readings = json.load(f)
        logging.info(f"Read sensor readings from {path_or_config}.")
else:
    sensor_readings = {}
    for sensor_name, sensor in {
        **device_under_test,
        **reference_sensors,
    }.items():

        indications, indication_uncertainties = sensor.indicated_value(
            measurand["quantity"]
        )
        indications += indication_uncertainties * np.random.randn(len(indications))

        sensor_readings[sensor_name] = {
            "time": measurand["time"],
            "val": indications,
            "val_unc": indication_uncertainties,
        }

sensor_readings_path = os.path.join(working_directory, "sensor_readings.json")
with open(sensor_readings_path, "w") as f:
    json.dump(sensor_readings, f, cls=NumpyEncoder, indent=None)


# load (nested) cocalibration methods
path_or_config = config["cocalibration"]
if isinstance(path_or_config, str):
    with open(path_or_config, "r") as f:
        cocalibration = json.load(f)
        logging.info(f"Read cocalibration configuration from {path_or_config}.")
else:
    cocalibration = path_or_config

use_interpolation = cocalibration["interpolate"]
run_blockwise = cocalibration["blockwise"]
if run_blockwise:
    t = measurand["time"]
    if "split_indices" in cocalibration.keys():
        split_indices = cocalibration["split_indices"]
    else:
        n_splits = np.random.randint(1, len(t) // 3)
        split_indices = np.sort(np.random.permutation(np.arange(3, len(t)-1))[:n_splits])
        cocalibration["split_indices"] = split_indices
    
    sensor_readings_splitted = split_sensor_readings(sensor_readings, split_indices, len(t))

methods_full = {}
for method_name, method_path_or_config in cocalibration["methods"].items():
    if isinstance(method_path_or_config, str):
        with open(method_path_or_config, "r") as f:
            method_description = json.load(f)
            logging.info(f"Read {method_name} from {method_path_or_config}.")
    else:
        method_description = method_path_or_config
    methods_full[method_name] = method_description
cocalibration["methods"] = methods_full

cocalibration_path = os.path.join(working_directory, "cocalibration.json")
with open(cocalibration_path, "w") as f:
    json.dump(cocalibration, f, cls=NumpyEncoder, indent=4)


# run cocalibration methods

results_directory = os.path.join(working_directory, "results")
if not os.path.exists(results_directory):
    os.mkdir(results_directory)

for method_name, method_args in cocalibration["methods"].items():
    logging.info("-" * 10)
    logging.info(f"Starting {method_name} at {datetime.datetime.now().isoformat()}.")

    # init method class
    method_class = getattr(cocalibration_methods, method_args["class_name"])
    method = method_class(**method_args["arguments"])

    device_under_test_copy = copy.deepcopy(device_under_test)

    results = []

    if run_blockwise:
        # TODO (split sensor readings into blocks, consume blocks one after another)
        # maybe implement this already at the level of sensor_readings and measurand?
        # somehow store the selected blocks?

        for current_sensor_readings in sensor_readings_splitted:
            logging.info("new block")
            block_result = method.update_params(current_sensor_readings, device_under_test_copy)
            results.append(block_result)

    else:
        result = method.update_params(sensor_readings, device_under_test_copy)
        results.append(result)
    
    # write results to file
    method_results_path = os.path.join(results_directory, f"{method_name}.json")
    with open(method_results_path, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=4)
        
    logging.info(f"Finished {method_name} at {datetime.datetime.now().isoformat()}.")
logging.info("-" * 10)


# clean up
logging.info(f"End: {datetime.datetime.now().isoformat()}")
logging.info("=" * 30)
