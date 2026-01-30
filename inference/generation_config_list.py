####################################################################################################################
#list of prompt instances for e-mail generation
####################################################################################################################

from inference.generation_config import generation_config

def get_config(config_name: str):
	#input: config_name
	#output: config instance if the name exists

	#return config based on string name
	if config_name not in globals():
		raise ValueError(f"Config name {config_name} not found")
	return globals()[config_name]



###german bart fine-tuned on reklamation data without the columns specified in README
bart_base_german_trained_on_reklam_data_removed_columns = generation_config(
	model_name="models/Shahm-bart-german_trained_on_reklamation_dataset_removed_columns"
)
