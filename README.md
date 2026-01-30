# Customer Service E-mail Response Generation by fine-tuning a language model on proprietary data compliant with GDPR Data protection

Code was adapted to not violate any proprietary information.


## 1. Requirements
Install python with tkinter (automatically installed together with python on Windows):
Linux:
```bash
apt-get install python3-tk
```
MacOS:
```bash
brew install python3-tk
```

Install requirements using:
```bash
python3 -m pip install -r requirements.txt
```

## 2. Training
Train a model by running the run_training script where the model to train is determined by the config_name in the bash script:
```bash
./run_training.sh
```
The config_name has to be added to the training/training_config_list.py file.

## 3. Generation
Run inference with a model with the run_generation or run_generation_loop scripts where the model to use is determined by the config_name in the script:
```bash
./run_generation.sh
./run_generation_loop.sh
```
Here run_generation works with textual or file as input, while the loop opens a GUI where textual input can be inserted.

Alternatively for the E-Mail generation GUI run:
```bash
python email_generation_gui.py
```

For a simple server deployment set-up see the Server_based_generation folder with the respective README.

Note that all the data and models were removed, as they are proprietary.






