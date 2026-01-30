####################################################################################################################
#example for getting answers from the server via API
####################################################################################################################

import socket
from inference.generation_api import generation_client

def main():
    hostname = socket.gethostname()
    port = 12345

    #instantiate the client
    example_client = generation_client(hostname=hostname, port=port)

    

    

    example_input = """Generiere eine Reklamationsantwort E-Mail aus den folgenden Daten:"""
    
    #generate an answer based on the given input
    print("generate answer from input")
    example_answer_from_input = example_client.generate_answer_from_input(example_input)
    print(f"Given input: {example_input}\nAnswer generated for the input:\n{example_answer_from_input}")


    #generate an answer based on the given ID
    # example_id = "RE39201"
    # print("generate answer from ID")
    # example_answer_from_id = example_client.generate_answer_from_id(example_id)
    # print(f"Answer generated for ID {example_id}:\n{example_answer_from_id}\n")



if __name__=="__main__":
    main()