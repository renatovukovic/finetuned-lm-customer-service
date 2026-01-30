####################################################################################################################
#Program for inference on inputs, load the model once and get text inputs for inference in infinite loop in a simple graphical user interface 
####################################################################################################################
import logging
import argparse
import tkinter as tk

from .email_generation_model import seq2seq_email_model
from .generation_config_list import *
from data.data_preparation import Prompt
from handle_logging_config import setup_logging



def main():
    logging.basicConfig(filename='logs/generation_loop.log', encoding='utf-8', level=logging.DEBUG)

    parser = argparse.ArgumentParser()
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

    logger.info("Set up in GUI")

    
    root = tk.Tk()
    root.title("Reklamationsantwort Generierung")
    #make the program fullscreen at start
    width=root.winfo_screenwidth() 
    height=root.winfo_screenheight()
    root.geometry("%dx%d" % (width, height))      

    inputfield=tk.Label(root, text="Geben Sie die Reklamationsdaten ein:")
    inputfield.grid(row=1,column=1)
    entryvalue = tk.StringVar()

    entry=tk.Text(root, height=30, width=70)
    entry.grid(row=2, column=2)


    def generate_output():
        outputfield.configure(state='normal')
        outputfield.delete('1.0', tk.END)  # Clear existing text
        outputfield.insert(tk.END, f'Generiere E-Mail aus Daten...')
        input_text = Prompt + entry.get("1.0",'end-1c')
        output = model.generate_email_from_data(input_text)
        outputfield.delete('1.0', tk.END)  # Clear existing text
        outputfield.insert(tk.END, output)
        outputfield.configure(state='disabled')

    button = tk.Button(root, text="Eingabe", command=generate_output)
    button.grid(row=3, column=2)

    outputtext = inputfield=tk.Label(root, text="Generierte E-mail:")
    outputtext.grid(row=1,column=3)

    #Create a text to print the output
    outputfield=tk.Text(root, height=30, width=70)
    # disable text widget to prevent editing
    outputfield.configure(state='disabled')
    outputfield.grid(row=2, column=5)
    root.mainloop()
        
    logger.info("DONE")



if __name__=="__main__":
    main()