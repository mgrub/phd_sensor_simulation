#!/bin/bash
SCENARIO_PATH=${1:-experiments/scenario_A}  # use arg $1 or default path /experiments/scenario_A

echo "Cleaning up scenario $SCENARIO_PATH."

echo "Found the following files:"
find $SCENARIO_PATH \! -name config.json -type f

read -p "Proceed with deleting? (y/n) " delete_files

if [ $delete_files = "y" ]; then
    find $SCENARIO_PATH \! -name config.json -type f -delete
else
    echo "Aborting."
fi
