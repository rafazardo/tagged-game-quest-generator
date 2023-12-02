"""
Program Name: download_dataset.py
Description: The program downloads the tvgqs dataset
Author: Andre Feijó, Pedro Fiorio, Rafael Zardo
Date: 11/02/2023
"""

import os
import requests
import argparse

def download_file(output_file):
    """
    Downloads the file from the URL and saves it in the specified directory.

    Arguments:
    output_file (str): Path of the output file.

    Outputs:
    None
    """
    url = "https://raw.githubusercontent.com/rafazardo/tvgqs/main/dataset/quests.txt"
    response = requests.get(url)

    if response.status_code == 200:
        with open(output_file, 'wb') as file:
            file.write(response.content)
        print(f"The tvgqs.txt file was downloaded successfully and saved in {output_file}")
    else:
        print("Failed to download the file")

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
    parser.add_argument('--path_to_save_dataset', type=str, required=True, help='Destination path to save the dataset')

    # Parsing command line arguments
    args = parser.parse_args()
    output_directory = args.path_to_save_dataset

    # Checking if the provided path is a valid directory
    if not is_valid_directory(output_directory):
        print("The provided path is not a valid directory.")
    else:
        # Constructing the output file path
        output_file = os.path.join(output_directory, "tvgqs.txt")

        # Calling the function to download the file
        download_file(output_file)
