####################################################################################################################
#list of configs for fine-tuning models on the e-mail generation task
####################################################################################################################

from training.training_config import training_config

def get_config(config_name: str):
	#input: name of the config
	#output: the respective config instance for the names

	#return config based on string name
	if config_name not in globals():
		raise ValueError(f"Config name {config_name} not found")
	return globals()[config_name]


###german bart config
bart_base_german_reklam_removed_columns_training = training_config(
        model_name="Shahm/bart-german",
        dataset_name="reklamation_dataset_removed_columns",
        num_train_epochs=15,
        batch_size=4,
)

#update the already trained model
bart_base_german_reklam_removed_columns_training_updating = training_config(
        model_name="models/Shahm-bart-german_trained_on_reklamation_dataset_removed_columns_OLD_april",
        dataset_name="reklamation_dataset_removed_columns",
        num_train_epochs=15,
        batch_size=4,
)

