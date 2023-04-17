import csv
import os
import sys

import numpy as np
import pandas as pd

from config import DICTIONARY_DIR


def load_dictionary(dic):
    """
    Loads dictionary

    :return: dictionary
    """
    print('Loading dictionary...')
    df = pd.read_csv(os.path.join(DICTIONARY_DIR, dic), encoding='UTF-8', names=['source', 'target'],
                     sep='\t', quoting=csv.QUOTE_NONE)
    # try:
    #     df = pd.read_csv(os.path.join(DICTIONARY_DIR, dic), encoding='UTF-8', names=['source', 'target'],
    #                      sep='\t', quoting=csv.QUOTE_NONE)
    # except FileNotFoundError:
    #     print('Could not load dictionary data! Ending the program')
    #     raise Exception("Could not load dictionary data! Ending the program")

    source = df['source'].apply(lambda x: np.str_(x))
    target = df['target'].apply(lambda x: np.str_(x))
    dictionary = dict(zip(source.values, target.values))

    print('Dictionary loaded.')

    return dictionary
