#!/bin/bash

#config_name=bart_base_german_reklam_removed_columns_training
config_name=bart_base_german_reklam_removed_columns_training_updating



python3 -u -m training.seq2seq_training \
    --config_name ${config_name}



