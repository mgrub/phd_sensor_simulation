#!/bin/bash

# setup environment
. /home/gruber04/python_venv/cocalibration/bin/activate
. restrict_cpus.sh 16

# test python
python -c 'print("")'
python --version
which python

declare -a scenarios=(
"experiments/01a_static_input/"
"experiments/01b_sinusoidal_input/"
"experiments/01c_jumping_input/"
"experiments/02a_static_input_noisy/"
"experiments/02b_sinusoidal_input_noisy/"
"experiments/02c_jumping_input_noisy/"
"experiments/03a_variable_blockwise/"
"experiments/03b_smaller_blocks/"
"experiments/03c_larger_blocks/"
"experiments/04a_dropouts/"
"experiments/04b_outliers/"
"experiments/05a_better_references/"
"experiments/05b_equal_references/"
"experiments/05c_worse_references/"
"experiments/07_thesis_example/"
)


while getopts ":rcv" flag
do
    case $flag in

    r) 
       for s in ${scenarios[*]}; do
         echo "running $s"
         python evaluation_runner.py $s
       done
       ;;

    c) 
       for s in ${scenarios[*]}; do
         echo "cleaning $s"
         . clean_up_experiment.sh $s
       done
       ;;

    v) 
       for s in ${scenarios[*]}; do
         echo "visualizing $s"
         python visualization_runner.py --novis $s
       done
       ;;

    *) 
       echo "invalid argument"
       ;;
    esac
done

