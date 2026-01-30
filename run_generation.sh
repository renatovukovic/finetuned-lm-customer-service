#!/bin/bash

config_name=bart_base_german_trained_on_reklam_data_removed_columns


python3 -u -m inference.generate \
    --input_file data/examples/reklam_removed_columns_input_example.json \
    --config_name ${config_name}
    