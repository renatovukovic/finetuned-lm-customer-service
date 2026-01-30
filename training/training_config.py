####################################################################################################################
#class for the training configs
####################################################################################################################

class training_config():
    model_name: str
    dataset_name: str
    autoregressive: bool #whether to use autoregressive or 
    num_train_epochs: int
    batch_size: int
    fp16: bool #floating point 16 precision for less memory consumption during training/inference
    eval_steps: int #when to eval/save during training
    

    def __init__(self, model_name, dataset_name, autoregressive=False, num_train_epochs=10, batch_size=8, fp16=True, eval_steps=5000):
        self.model_name = model_name
        self.dataset_name=dataset_name
        self.autoregressive=autoregressive
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.fp16=fp16
        self.eval_steps=eval_steps

    #method for returning the config params as a dict for printing/saving as json
    def to_dict(self):
        return vars(self)