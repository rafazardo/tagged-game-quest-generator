from transformers import pipeline
import argparse
import os
from model import TaggedGameQuestGenerator

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
    parser.add_argument('--path_outdir_sentences', type=str, required=True, help='Destination path to save the generated sentences')
    parser.add_argument('--path_indir_trained_model', type=str, required=True, help='Destination get the trained model')
    parser.add_argument('--num_sentences', type=int, required=True, help='Number of sentences to be generated')

    # Parsing command line arguments
    args = parser.parse_args()
    output_directory = args.path_outdir_sentences
    trained_model_path = args.path_indir_trained_model
    num_sentences = args.num_sentences

    # Checking if the provided path is a valid directory
    if not is_valid_directory(output_directory):
        print("The provided path is not a valid directory.")
    else:
        # Constructing the output file path
        output_file = os.path.join(output_directory, "tvgqs.txt")

    model = TaggedGameQuestGenerator()

    quest_generator = model.get_pipeline(trained_model_path)

    sentences = []

    for i in range(num_sentences):
        speech = quest_generator("<SOS>", max_length=100, min_length=10, do_sample=True, temperature=1.2)[0]['generated_text']
        sentences.append(speech)
        if '<EOS>' in speech:
            break

    with open(output_directory, 'w', encoding='utf-8') as out_file:
        for sentence in sentences:
            if sentence[:-1] != '\n':
                sentence += '\n'
            out_file.write(sentence)