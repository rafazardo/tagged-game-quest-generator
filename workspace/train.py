"""
Program Name: model.py
Description: The program creates the TaggedGameQuestGenerator model
Author: Andre Feijó, Pedro Fiorio, Rafael Zardo
Date: 11/02/2023
"""

from transformers import pipeline
from transformers import GPT2Tokenizer
import argparse
import os


from model import TaggedGameQuestGenerator
from tvgqs import TVGQS

def run_train(path_outdir_trained_model, num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size, eval_steps, save_steps, warmup_steps, path_indir_dataset, path_outdir_train_dataset, path_outdir_test_dataset, test_size):
    tagged_game_quest_generator = TaggedGameQuestGenerator()
    tvgqs = TVGQS(path_indir_dataset, test_size)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    tokenizer.add_special_tokens({'bos_token': '<SOS>'})
    tokenizer.add_special_tokens({'eos_token': '<EOS>'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['<SOA>', '<EOA>']})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[CHARACTER]', '[ITEM]', '[PLACE]', '[ENEMY]', '[NUMBER]']})

    path_indir_train_dataset = tvgqs.build_train_txt(path_outdir_train_dataset)
    path_indir_test_dataset = tvgqs.build_test_txt(path_outdir_test_dataset)
    train_tvgqs_dataset, test_tvgqs_dataset, data_collator = tvgqs.load_dataset(path_indir_train_dataset, path_indir_test_dataset, tokenizer)

    tagged_game_quest_generator.set_training_args(path_outdir_trained_model, num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size, eval_steps, save_steps, warmup_steps, train_tvgqs_dataset, test_tvgqs_dataset, data_collator, tokenizer)
    tagged_game_quest_generator.train()


def is_valid_directory(path):
    """
    Checks if the provided path is a valid directory.

    Arguments:
    path (str): Path to be checked.

    Outputs:
    bool: Returns True if the path is a valid directory, False otherwise.
    """
    if os.path.isdir(path):
        return True
    return False

if __name__ == "__main__":
    # Setting up the parser to accept command line arguments
    parser = argparse.ArgumentParser(description='Download dataset to a specific directory.')
    parser.add_argument('--path_outdir_trained_model', type=str, required=True, help='Destination path to save the trained model')
    parser.add_argument('--num_train_epochs', type=int, required=True, help='Number of train epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=32, help='Batch size per device during training')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=64, help='Batch size per device during evaluation')
    parser.add_argument('--eval_steps', type=int, default=400, help='Number of evaluation steps')
    parser.add_argument('--save_steps', type=int, default=100, help='Number of steps to save the model')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of steps to stabilize the learning rate')
    parser.add_argument('--path_indir_dataset', type=str, required=True, help='Destination path to load the dataset')
    parser.add_argument('--path_outdir_train_dataset', type=str, required=True, help='Destination path to save the train dataset')
    parser.add_argument('--path_outdir_test_dataset', type=str, required=True, help='Destination path to save the teste dataset')
    parser.add_argument('--test_size', type=float, default=0.15, help='Percentage of the data destined to test')

    # Parsing command line arguments
    args = parser.parse_args()
    path_outdir_trained_model = args.path_outdir_trained_model
    num_train_epochs = args.num_train_epochs
    per_device_train_batch_size = args.per_device_train_batch_size
    per_device_eval_batch_size = args.per_device_eval_batch_size
    eval_steps = args.eval_steps
    save_steps = args.save_steps
    warmup_steps = args.warmup_steps
    path_indir_dataset = args.path_indir_dataset
    path_train_dataset = args.path_outdir_train_dataset
    path_test_dataset = args.path_outdir_test_dataset
    test_size = args.test_size

    # Checking if the provided path is a valid directory
    if not is_valid_directory(path_outdir_trained_model):
        print("The provided path is not a valid directory.")
    else:
        # Constructing the output file path
        output_file = path_outdir_trained_model

    run_train(path_outdir_trained_model, num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size, eval_steps, save_steps, warmup_steps, path_indir_dataset, path_train_dataset, path_test_dataset, test_size)