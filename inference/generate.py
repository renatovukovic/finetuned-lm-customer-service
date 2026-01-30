####################################################################################################################
#do inference on inputs
#input is either a text, a file containing the input dict or an interactively typed text
#output is for one input the generated e-mail response
####################################################################################################################

import transformers
from transformers import pipeline
import logging
import sys
import argparse
import json
from pathlib import Path

from .email_generation_model import seq2seq_email_model
from .generation_config_list import *
from data.data_preparation import get_input_string_from_dict
from handle_logging_config import setup_logging



def main():
    logging.basicConfig(filename='logs/generate.log', encoding='utf-8', level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_text', action='store', type=str, help='The text to use as input', default="")
    parser.add_argument('--input_file', action='store', type=str, help='The file to use as input', default="")
    parser.add_argument("--interactive", help="input text interactively", action="store_true", default=False)
    parser.add_argument("--config_name", help="Name of the config to use", type=str, default="bart_base_german_trained_on_reklam_data_removed_columns")
    args = parser.parse_args()

    #setup logging
    logger = setup_logging("generation_" + args.config_name)

    logger.info(f"Running with config: {args.config_name}")

    #load the config
    config = get_config(args.config_name)
    config_param_dict = config.to_dict()
    logger.info(f"Loaded config: {config_param_dict}")


    logger.info(f"Load model {config.model_name}.")
    model = seq2seq_email_model(config) 
    logger.info("Loaded model successfully.")

    logger.info(f"Generate with input {args.input_file}.")

    if args.input_text: 
        output = model.generate_email_from_data(input_text)
        
    elif args.input_file:
        with Path(args.input_file).open("r") as file:
            inputfile = json.load(file)
        #convert the input file to a prompt for the model to generate the corresponding output
        input_text = get_input_string_from_dict(inputfile)
        output = model.generate_email_from_data(input_text)


    elif args.interactive:
        input_text = input("Enter the input text: ")
        output = model.generate_email_from_data(input_text)
    
    print("input: \n", input_text)
    print("output to the given input text by generation model:\n", output)


    if args.input_file:
        generation_output_filepath = "inference/generation_outputs/generated_output_for_" + args.input_file.replace("json", "txt").replace("data/", "")
        logger.info(f"Save generated output to {generation_output_filepath}")
        #save the output text
        with Path(generation_output_filepath).open("w") as output_file:
            print(output, file=output_file)
        
    logger.info("DONE")



if __name__=="__main__":
    main()