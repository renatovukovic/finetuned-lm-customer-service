####################################################################################################################
#server for inference via API requests
#do inference based on Reklamation ID as input via API request on server
#based on the ID get the needed information from the MongoDB
#generate an answer for the ID, which is the output of the API request
####################################################################################################################

import transformers
from transformers import pipeline
import logging
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import socket
from threading import Thread

from .email_generation_model import seq2seq_email_model
from .generation_config_list import *
from data.data_preparation import get_input_string_from_dict, get_seg2seqdata_from_dicts
from handle_logging_config import setup_logging



def main():
    logging.basicConfig(filename='logs/generate.log', encoding='utf-8', level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_name", help="Name of the config to use", type=str, default="bart_base_german_trained_on_reklam_data_removed_columns")
    parser.add_argument('--port', action='store', type=int, help='Port for the server', default=12345)
    args = parser.parse_args()

    #setup logging
    logger = setup_logging("generation_via_api_server" + args.config_name)

    logger.info(f"Running with config: {args.config_name}")

    #load the config
    config = get_config(args.config_name)
    config_param_dict = config.to_dict()
    logger.info(f"Loaded config: {config_param_dict}")


    logger.info(f"Load model {config.model_name}.")
    model = seq2seq_email_model(config) 
    logger.info("Loaded model successfully.")
    logger.info(f"Start server on port {args.port}")


    #based on https://www.digitalocean.com/community/tutorials/python-socket-programming-server-client


    try:
        ###### add threading for multiple clients simultaneously based on https://stackoverflow.com/questions/10810249/python-socket-multiple-clients
        def on_new_client(client_socket,addr):
            while True:
                msg = client_socket.recv(8192).decode('utf-8')
                if not msg:
                    print(f"Closing connection {client_socket=} {addr=}")
                    break
                print(addr, ' >> ', msg)
                #msg = 'Die generierte Antwort zur Reklamations ID lautet >> \n' + msg
                #generiere die KI-Antwort auf Grundlage der Reklamations ID und Infos von MongoDB
                ki_input = get_ki_input(msg, mongo_db=False)
                if not ki_input:
                    ki_antwort = "Die angegebene Reklamations ID ist nicht in der Datenbank."
                else:
                    logger.info(f"Generate response for ID {msg}")
                    ki_antwort = model.generate_email_from_data(ki_input)
                    ki_antwort = "Die generierte Antwort zur Reklamations ID lautet >> \n" + ki_antwort

                client_socket.send(ki_antwort.encode())
            

        host = socket.gethostname() # Get local machine name
        port = args.port  # initiate port no above 1024

        server_socket = socket.socket()  # get instance
        server_socket.bind((host, port))  # bind host address and port together
        # configure how many client the server can listen simultaneously
        server_socket.listen(20)


        
        while True:
            c, addr = server_socket.accept()     # Establish connection with client.
            print('New connection from', addr)
            rekla_thread = Thread(target=on_new_client, args=(c,addr))
            rekla_thread.start()
        c.close()
        rekla_thread.join()
        server_socket.close()
        

    except KeyboardInterrupt:
        print("\nClosing Connection and freeing the port.")
        server_socket.close()
        sys.exit()



def get_ki_input(rekla_id: str = None, mongo_db=True) -> str:
    #input: 
    #rekla_id: ReklamationsID als String
    #mongo_db: whether to use information from mongo_db

    #output:
    #the input string to the ai model

    rekla_input_dict = {}
    print(f"Get input for ID: {rekla_id}")

    if mongo_db:
        ### Read Reklamationsdaten from MongoDB
        ##TODO
        pass

    else:
        reklam_dataframe = pd.read_excel("../data/20240404_114735__DVS-Kundenreklamationen.xlsx")
        
        #Remove sendungsfunde
        reklam_dataframe = reklam_dataframe[reklam_dataframe["Rekla-Grund"] != "Sendungsfund"]
        reklam_dataframe = reklam_dataframe[reklam_dataframe["Rekla-Typ"] != "SF"]
        
        #only keep the columns that are really relevant for the response
        columns_to_keep = ['ID', 'Erstellt am', 'Aktualisiert', 'Ersteller', 'Rekla-Grund', 'Prozess', 'Rekla-Typ', 'Debitor', 'Debitor Nummer', 'Beanstander', 'Firma', 'Anrede', 'Vorname', 'Nachname', 'Straße', 'Hausnr.', 'Zusatz', 'Postleitzahl', 'Ort', 'Sendungsnr.', 'Sendungsdatum', 'Produkt', 'ZUP Nr.', 'ZUP Name', 'Verschulden', 'Original Sachverhalt', 'Sachverhalt', 'Priorität', 'Status', 'Interne Notizen', 'ZUP', 'Debitor.1', 'System-extern']
        columns_to_drop = [item for item in reklam_dataframe.columns if item not in columns_to_keep]

        remove_columns = True
        #remove_columns = False
        if remove_columns:  
            reklam_dataframe.drop(columns=columns_to_drop, inplace=True)
    
        #turn the dataframe to a dict as input into the model
        reklam_dataframe_dict = reklam_dataframe.to_dict("index")
        #prepare the input and output
        output_columns = ["Debitor.1", "System-extern"]
        raw_data = []
        #get a dict with the ids as keys and the rest of the dict as value
        id_input_dict = {}
        for row_id, row_dict in tqdm(reklam_dataframe_dict.items()):
            entry = {}
            input_dict = {}
            #fill the input dict with the input information and put an empty string for nan
            for key, value in row_dict.items():
                if key not in output_columns:
                    if pd.isna(value):
                        input_dict[key] = ""
                    else:
                        input_dict[key] = value
            
            entry["Input"] = input_dict
            #if both are empty just make the output empty, as the problem was not answered yet by the ZUP
            if pd.isna(row_dict["Debitor.1"]) and pd.isna(row_dict["System-extern"]):
                entry["Output"] = ""
            #check if Debitor is empty, then use System-extern
            elif pd.isna(row_dict["Debitor.1"]):
                entry["Output"] = row_dict["System-extern"]
            else:
                entry["Output"] = row_dict["Debitor.1"]

            raw_data.append(entry)

            current_id = entry["Input"]["ID"]
            id_input_dict[current_id] = {"input": entry["Input"], "output": entry["Output"]}  #also save the output for potential comparison

            #check whether the given id is in the dict
            if rekla_id in id_input_dict:
                rekla_input_dict = id_input_dict[rekla_id]["input"]



        if not rekla_input_dict:
            return ""
        
        input_text = get_input_string_from_dict(rekla_input_dict)

        return input_text
        



if __name__=="__main__":
    main()




