####################################################################################################################
#config for e-mail generation from structured data string
####################################################################################################################

class generation_config():
    model_name: str #name/path for loading the model
    anredencheck: bool #whether to check the presence of greeting in the output mail
    sample: bool #whether to sample during generation

    def __init__(self, model_name, anredencheck=True, sample=False):
        self.model_name = model_name
        self.anredencheck = anredencheck
        self.sample = sample

    #method for returning the config params as a dict for saving as json
    def to_dict(self):
        return vars(self)



