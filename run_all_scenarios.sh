#!/bin/bash

# setup environment
. /home/gruber04/python_venv/cocalibration/bin/activate
. restrict_cpus.sh 4

# test python
python -c 'print("abc")'; python --version; which python


# run scenarios

python evaluation_runner.py experiments/01a_static_input/
python evaluation_runner.py experiments/01b_sinusoidal_input/
python evaluation_runner.py experiments/01c_jumping_input/
python evaluation_runner.py experiments/02a_static_input_noisy/
python evaluation_runner.py experiments/02b_sinusoidal_input_noisy/
python evaluation_runner.py experiments/02c_jumping_input_noisy/
python evaluation_runner.py experiments/03a_two_references/
python evaluation_runner.py experiments/03b_ten_references/
python evaluation_runner.py experiments/04a_dropouts/
python evaluation_runner.py experiments/04b_outliers/
python evaluation_runner.py experiments/05a_better_references/
python evaluation_runner.py experiments/05b_equal_references/
python evaluation_runner.py experiments/05c_worse_references/

# clean up scenarios
# . clean_up_experiment.sh experiments/01a_static_input/
# . clean_up_experiment.sh experiments/01b_sinusoidal_input/
# . clean_up_experiment.sh experiments/01c_jumping_input/
# . clean_up_experiment.sh experiments/02a_static_input_noisy/
# . clean_up_experiment.sh experiments/02b_sinusoidal_input_noisy/
# . clean_up_experiment.sh experiments/02c_jumping_input_noisy/
# . clean_up_experiment.sh experiments/03a_two_references/
# . clean_up_experiment.sh experiments/03b_ten_references/
# . clean_up_experiment.sh experiments/04a_dropouts/
# . clean_up_experiment.sh experiments/04b_outliers/
# . clean_up_experiment.sh experiments/05a_better_references/
# . clean_up_experiment.sh experiments/05b_equal_references/
# . clean_up_experiment.sh experiments/05c_worse_references/