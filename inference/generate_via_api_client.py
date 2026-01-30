####################################################################################################################
#client for inference via API requests
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

from .email_generation_model import seq2seq_email_model
from .generation_config_list import *
from data.data_preparation import get_input_string_from_dict, get_seg2seqdata_from_dicts
from handle_logging_config import setup_logging



def main():
    logging.basicConfig(filename='logs/generate.log', encoding='utf-8', level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument('--ReklaID', action='store', type=str, help='The ID of the complaint', default='1234')
    parser.add_argument('--port', action='store', type=int, help='Port for the server', default=12345)
    args = parser.parse_args()

    #setup logging
    logger = setup_logging("generation_via_api_client_ID_" + args.ReklaID + "_port" + str(args.port))



    #based on https://www.digitalocean.com/community/tutorials/python-socket-programming-server-client

    try:
        host = socket.gethostname()  # as both code is running on same pc
        port = args.port  # socket server port number

        client_socket = socket.socket()  # instantiate
        client_socket.connect((host, port))  # connect to the server

        message = input("Gib eine Reklamations ID an: ")  # take input

        while True:
            client_socket.send(message.encode())  # send message
            data = client_socket.recv(8192).decode()  # receive response

            print('Received from server: ' + data)  # show in terminal

            message = input("Geben Sie eine Reklamations ID an: ")  # again take input

        

    except KeyboardInterrupt:
        print("Closing Connection and freeing the port.")
        client_socket.close()
        sys.exit()


    
        
    logger.info("DONE")



if __name__=="__main__":
    main()


