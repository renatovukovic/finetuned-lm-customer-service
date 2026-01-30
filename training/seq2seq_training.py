####################################################################################################################
#Train sequence to sequence model on the data input information as input and the e-mail as output
####################################################################################################################

from transformers import  AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import logging
from datasets import load_from_disk
import evaluate
import argparse
import tensorboard
import nltk
import torch
import numpy as np

from training.training_config_list import *
from handle_logging_config import *



def train():
    logging.basicConfig(filename='generate.log', encoding='utf-8', level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_name", help="Name of the config to use", type=str, default="bart_base_german_reklam_removed_columns_training")

    args = parser.parse_args()

    #load the training data
    #setup logging
    logger = setup_logging("training_" + args.config_name)

    logger.info(f"Running with config: {args.config_name}")

    #load the config
    config = get_config(args.config_name)
    config_param_dict = config.to_dict()
    logger.info(f"Loaded config: {config_param_dict}")

    #load the model
    logger.info(f"Load the model {config.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    #put model in train mode
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running with device {device}")

    #fp16 only works on cuda devices
    if torch.device != "cuda":
        config.fp16 = False

    model.to(device)

    logger.info("Loading the model done.")

    logger.info(f"Load the dataset {config.dataset_name}.")
    seq2seq_dataset = load_from_disk(f"data/{config.dataset_name}")
    logger.info("Successfully loaded the dataset.")


    #setup training
    batch_size = config.batch_size
    model_name = config.model_name.replace("/", "-") + "_trained_on_" + config.dataset_name  
    model_dir = f"models/{model_name}"
    args = Seq2SeqTrainingArguments(
        model_dir,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        logging_strategy="steps",
        logging_steps=config.eval_steps / 10,
        save_strategy="steps",
        save_steps=config.eval_steps * 10,
        learning_rate=4e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=config.num_train_epochs,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        report_to="tensorboard",
        generation_max_length=tokenizer.model_max_length,
        warmup_ratio=0.1,
        fp16=config.fp16,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    def preprocess_function(examples):
        inputs = examples["inputs"]
        targets = examples["labels"]
        model_inputs = tokenizer(
            inputs, text_target=targets, truncation=True)
        return model_inputs

    tokenized_dataset = seq2seq_dataset.map(
        preprocess_function,
        batched=True,
    )
    
    #metrics for evaluation
    def compute_metrics(eval_preds):
        metric = evaluate.combine(["rouge", "bleu", "meteor", "sacrebleu"])
        predictions, labels = eval_preds
        #remove undecodable token ids
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        return metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    

    logger.info("Start training the model.")
    trainer.train()
    logger.info("Training done.")

    logger.info(f"Training done. Save model to {model_dir}")
    trainer.save_model()
    logger.info("Saving successful.")

    logger.info("Make predictions on the validation dataset.")
    val_evaluation = trainer.evaluate(max_length=tokenizer.model_max_length)
    logger.info(f"Results on validation set after training: \n {val_evaluation}")


    logger.info("Make predictions on the test dataset.")
    test_evaluation = trainer.predict(tokenized_dataset["test"], max_length=tokenizer.model_max_length)
    logger.info(f"Results on test set after training: \n {test_evaluation}")
    logger.info("DONE")

if __name__=="__main__":
    train()