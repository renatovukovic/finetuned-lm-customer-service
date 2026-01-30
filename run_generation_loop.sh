#!/bin/bash

config_name=bart_base_german_trained_on_reklam_data_removed_columns


python3 -u -m inference.generation_loop \
    --config_name ${config_name}
    