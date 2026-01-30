####################################################################################################################
#config for e-mail generation from structured data string
####################################################################################################################

class generation_config():
    model_name: str #name/path for loading the model

    def __init__(self, model_name):
        self.model_name = model_name

    #method for returning the config params as a dict for saving as json
    def to_dict(self):
        return vars(self)



