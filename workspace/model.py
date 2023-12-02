"""
Program Name: model.py
Description: The program creates the TaggedGameQuestGenerator model
Author: Andre Feijó, Pedro Fiorio, Rafael Zardo
Date: 11/02/2023
"""

from transformers import Trainer, TrainingArguments, GPT2LMHeadModel
from transformers import pipeline

class TaggedGameQuestGenerator:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.training_args = None  
        self.path_train_dataset = None 
        self.path_test_dataset = None
        self.path_data_collator = None

    def set_training_args(self, path_outdir_trained_model, num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size, eval_steps, save_steps, warmup_steps, path_train_dataset, path_test_dataset, path_data_collator):
        """
        Sets up training arguments for the model.

        Args:
            path_outdir_trained_model (str): Output directory for the trained model.
            num_train_epochs (int): Number of training epochs.
            per_device_train_batch_size (int): Batch size for training.
            per_device_eval_batch_size (int): Batch size for evaluation.
            eval_steps (int): Evaluation steps.
            save_steps (int): Save steps during training.
            warmup_steps (int): Warmup steps.
            path_train_dataset (str): Path to the training dataset.
            path_test_dataset (str): Path to the testing dataset.
            path_data_collator (str): Path to the data collator.
        """
        self.training_args = TrainingArguments(
            output_dir=path_outdir_trained_model + "/tagged-quest-generator",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_steps=eval_steps,
            save_steps=save_steps,
            warmup_steps=warmup_steps
        )
        self.path_train_dataset = path_train_dataset
        self.path_test_dataset = path_test_dataset
        self.path_data_collator = path_data_collator

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.path_data_collator,
            train_dataset=self.path_train_dataset,
            eval_dataset=self.path_data_collator,
        )

    def train(self):
        """
        Trains the model using the specified training arguments.
        """
        if self.training_args is None:  # Checks if training arguments are configured
            raise ValueError("Please configure the training arguments using set_training_args() before calling the train method.")
        else:
            self.trainer.train()

    def get_pipeline(self, path_trained_model):
        """
        Gets a pipeline for text generation using the trained model.

        Args:
            path_trained_model (str): Path to the trained model.

        Returns:
            Pipeline for text generation.
        """
        return pipeline('text-generation', model = path_trained_model, tokenizer='gpt2', max_length=10000, temperature=0.5)