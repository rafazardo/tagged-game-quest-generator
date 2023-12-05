import os
from transformers import pipeline
from model import TaggedGameQuestGenerator
import argparse

def is_valid_directory(path):
    """
    Checks if the provided path is a valid directory.

    Arguments:
    path (str): Path to be checked.

    Outputs:
    bool: Returns True if the path is a valid directory, False otherwise.
    """
    return os.path.isdir(path)

def run_inference(trained_model_path, output_directory, num_sentences, temperature):
    """
    Runs the inference using the provided model and generates sentences.

    Arguments:
    trained_model_path (str): Path to the trained model.
    output_directory (str): Destination path to save the generated sentences.
    num_sentences (int): Number of sentences to be generated.
    temperature (float): Hiperparameter to control randomness of output 

    This function initializes the model, generates sentences using the provided model path,
    and saves the generated sentences to the specified output directory.
    """
    model = TaggedGameQuestGenerator()  # Initialize the model
    quest_generator = model.get_pipeline(trained_model_path)  # Get the model pipeline
    sentences = []  # List to store generated sentences

    # Generate sentences until <EOS> token is encountered or the specified number of sentences is reached
    for i in range(num_sentences):
        speech = quest_generator("<SOS>", max_length=200, min_length=10, do_sample=True, temperature=temperature)[0]['generated_text']
    
        # Verifica se <EOS> está presente na string
        if '<EOS>' in speech:
            # Encontra a posição de <EOS> na string
            eos_index = speech.index('<EOS>')
            
            # Corta a string do início até <EOS>
            speech = speech[:eos_index + len('<EOS>')]

    # Write the generated sentences to the output file
    with open(output_directory, 'w', encoding='utf-8') as out_file:
        for sentence in sentences:
            if sentence[:-1] != '\n':
                sentence += '\n'
            out_file.write(sentence)

if __name__ == "__main__":
    # Setting up the parser to accept command line arguments
    parser = argparse.ArgumentParser(description='Download dataset to a specific directory.')
    parser.add_argument('--path_outdir_sentences', type=str, required=True, help='Destination path to save the generated sentences')
    parser.add_argument('--path_indir_trained_model', type=str, required=True, help='Destination get the trained model')
    parser.add_argument('--num_sentences', type=int, required=True, help='Number of sentences to be generated')
    parser.add_argument('--temperature', type=float, default=1.2, help='Hiperparameter to control randomness of output')

    # Parsing command line arguments
    args = parser.parse_args()
    output_directory = args.path_outdir_sentences
    trained_model_path = args.path_indir_trained_model
    num_sentences = args.num_sentences
    temperature = args.temperature

    # Checking if the provided path is a valid directory
    if not is_valid_directory(output_directory):
        print("The provided path is not a valid directory.")
    else:
        # Constructing the output file path
        output_file = os.path.join(output_directory, "tvgqs.txt")

        # Running the inference
        run_inference(trained_model_path, output_directory, num_sentences, temperature)
