# Simulation environment to test, compare and evaluate multiple co-calibration methods

## Run a given scenario

A scenario can be configured and then multiple methods are applied to the same configuration.

```bash
python evaluation_runner.py experiments/scenario_A/
```

## Visualize results of a scenario

Once run, the results can be visualized:

```bash
python evaluation_runner.py experiments/scenario_A/
```
## Define a scenario

A scenario is defined inside a `config.json`, which stores information about the

- random state
- reference sensors
- device under test
- measurand
- sensor readings
- methods to be used

All options can also be loaded from an existing (i.e. from a previous run, to achieve or manipulate a specific setting).

## Define a method

Available methods are defined in `method_args/<method_name>.json`.

- class name (as used in the actual source `cocalibration_methods.py`)
- arguments to initialize the class

