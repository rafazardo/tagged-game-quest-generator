"""
Program Name: tvgqs.py
Description: The program creates train and test files
Author: Andre Feijó, Pedro Fiorio, Rafael Zardo
Date: 11/02/2023
"""

from sklearn.model_selection import train_test_split
import csv
import re

class TVGQS:
    def __init__(self, path_indir_dataset, test_size):
        """
            This method reads the dataset and splits it into train and test slices

            Args:
                path_indir_dataset (str): Path to the dataset csv file.
                test_size (float): Percentage of the data destined to test.

            Returns:
                tipo_de_retorno: Descrição do que o método retorna.
        """

        self.path_indir_dataset = path_indir_dataset

        with open(self.path_indir_dataset, 'r') as arquivo:
            leitor_csv = csv.reader(arquivo)

        speeches_txt = []

        for linha in leitor_csv:
            speeches_txt.append(linha[0])

        speeches_txt = ["<SOS>" + speech + "<EOS>" for speech in speeches_txt]

        self.train, self.test = train_test_split(speeches_txt, test_size=test_size)

    def build_train_txt(self, path_outdir_train_txt):
        """
        This method creates a txt file containing the train data

        Args:
            path_outdir_train_txt (str): Path to create the train file.
        """
        self.build_text_file(self.train, path_outdir_train_txt)

    def build_test_txt(self, path_outdir_test_txt):
        """
        This method creates a txt file containing the test data

        Args:
            path_outdir_test_txt (str): Path to create the test file.
        """
        self.build_text_file(self.test, path_outdir_test_txt)

    def _build_text_file(self, txts, dest_path):
        with open(dest_path, 'w') as f:
            for txt in txts:
                summary = txt.strip()
                summary = re.sub(r"\s", " ", summary)
                f.write(summary + "\n")