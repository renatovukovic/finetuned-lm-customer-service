####################################################################################################################
#prepare data for seq2seq and autoregressive training
####################################################################################################################


Prompt = "Generiere eine Reklamationsantwort E-Mail aus den folgenden Daten:\n"

def get_seg2seqdata_from_dicts(data: list[str, str]) -> dict:
    #input: data: data input list consisting of (input, output) pair tuples
    #output: a data set dict consisting of a list of inputs and a list of output that can then be turned into a huggingface dataset
    inputs = []
    labels = []
    for input, output in data:
        inputstring = Prompt + str(input)
        inputs.append(inputstring)
        labels.append(output)

    dataset_dict = {"inputs": inputs, "labels": labels}
    return dataset_dict

def get_input_string_from_dict(inputdict: dict) -> str:
    #input: the inputdict with the information for e-mail generation
    #output: turn the inputdict into a string and prepend the prompt
    inputstring = Prompt + str(inputdict)
    return inputstring

