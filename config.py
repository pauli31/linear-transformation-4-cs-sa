import os
from pathlib import Path

# Important to reproduce results
RANDOM_SEED = 666

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_PATH, 'data')
LOG_DIR = os.path.join(BASE_PATH, 'logs')
RESULTS_DIR = os.path.join(BASE_PATH, "results")
POLARITY_DIR = os.path.join(DATA_DIR, 'polarity')
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
DICTIONARY_DIR = os.path.join(DATA_DIR, 'dictionary')

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)

LSTM_CLASSIFIER = 'lstm'  # LSTM (neuronova sit)
CNN_CLASSIFIER = 'cnn'  # CNN (neuronova sit)

DATASET_NAMES = ['fb', 'csfd', 'mallcz', 'imdb', 'sst', 'allocine', 'csfd-imdb', 'imdb-csfd', 'csfd-sst', 'sst-csfd',
                 'allocine-csfd', 'allocine-imdb', 'allocine-sst', 'csfd-allocine', 'imdb-allocine', 'sst-allocine']
MODEL_NAMES = [LSTM_CLASSIFIER, CNN_CLASSIFIER]
SCHEDULERS = ["linear_wrp", "constant", "cosine_wrp", "polynomial_wrp"]
OPTIMIZERS_CHOICES = ['Adam', 'AdamW', 'RMSprop', 'Adagrad', 'Adadelta', 'SGD']
TRANSFORMATIONS = ['none', 'orthogonal', 'lst', 'cca', 'ranking', 'orto-ranking']

MODES = ['crosslingual', 'monolingual']
CROSS_LINGUAL_MODE = 'crosslingual'
MONOLINGUAL_MODE = 'monolingual'

SOURCE_TO_TARGET = 'source_to_target'
TARGET_TO_SOURCE = 'target_to_source'
TRANSFORMATION_TYPES = [SOURCE_TO_TARGET, TARGET_TO_SOURCE]

LOGGING_FORMAT = '%(asctime)s: %(levelname)s: %(name)s %(message)s'
LOGGING_DATE_FORMAT = '%m/%d/%Y %H:%M:%S'

# Datasets
# fb dataset dirs
FACEBOOK_DATASET_DIR = os.path.join(POLARITY_DIR, 'fb', 'split')
FACEBOOK_DATASET_TRAIN = os.path.join(FACEBOOK_DATASET_DIR, 'train', 'train.csv')
FACEBOOK_DATASET_TEST = os.path.join(FACEBOOK_DATASET_DIR, 'test', 'test.csv')
FACEBOOK_DATASET_DEV = os.path.join(FACEBOOK_DATASET_DIR, 'dev', 'dev.csv')
FACEBOOK_DATASET = os.path.join(FACEBOOK_DATASET_DIR, 'dataset.csv')

# csfd dataset dirs
CSFD_DATASET_DIR = os.path.join(POLARITY_DIR, 'csfd', 'split')
CSFD_DATASET_TRAIN = os.path.join(CSFD_DATASET_DIR, 'train', 'train.csv')
CSFD_DATASET_TEST = os.path.join(CSFD_DATASET_DIR, 'test', 'test.csv')
CSFD_DATASET_DEV = os.path.join(CSFD_DATASET_DIR, 'dev', 'dev.csv')
CSFD_DATASET = os.path.join(CSFD_DATASET_DIR, 'dataset.csv')

# mallcz cz dataset dirs
MALL_DATASET_DIR = os.path.join(POLARITY_DIR, 'mallcz', 'split')
MALL_DATASET_TRAIN = os.path.join(MALL_DATASET_DIR, 'train', 'train.csv')
MALL_DATASET_TEST = os.path.join(MALL_DATASET_DIR, 'test', 'test.csv')
MALL_DATASET_DEV = os.path.join(MALL_DATASET_DIR, 'dev', 'dev.csv')
MALL_DATASET = os.path.join(MALL_DATASET_DIR, 'dataset.csv')

IMDB_DATASET_CL_DIR = os.path.join(POLARITY_DIR, 'imdb', 'split-cl')
IMDB_DATASET_CL_TRAIN = os.path.join(IMDB_DATASET_CL_DIR, 'train', 'train.csv')
IMDB_DATASET_CL_TEST = os.path.join(IMDB_DATASET_CL_DIR, 'test', 'test.csv')
IMDB_DATASET_CL_DEV = os.path.join(IMDB_DATASET_CL_DIR, 'dev', 'dev.csv')
IMDB_DATASET_CL = os.path.join(IMDB_DATASET_CL_DIR, 'dataset.csv')

IMDB_DATASET_DIR = os.path.join(POLARITY_DIR, 'imdb', 'split')
IMDB_DATASET_TRAIN = os.path.join(IMDB_DATASET_DIR, 'train', 'train.csv')
IMDB_DATASET_TEST = os.path.join(IMDB_DATASET_DIR, 'test', 'test.csv')
IMDB_DATASET_DEV = os.path.join(IMDB_DATASET_DIR, 'dev', 'dev.csv')
IMDB_DATASET = os.path.join(IMDB_DATASET_DIR, 'dataset.csv')

SST_DATASET_DIR = os.path.join(POLARITY_DIR, 'sst', 'split')
SST_DATASET_TRAIN = os.path.join(SST_DATASET_DIR, 'train', 'train.csv')
SST_DATASET_TEST = os.path.join(SST_DATASET_DIR, 'test', 'test.csv')
SST_DATASET_DEV = os.path.join(SST_DATASET_DIR, 'dev', 'dev.csv')
SST_DATASET = os.path.join(SST_DATASET_DIR, 'dataset.csv')

ALLOCINE_DATASET_DIR = os.path.join(POLARITY_DIR, 'allocine', 'split')
ALLOCINE_DATASET_TRAIN = os.path.join(ALLOCINE_DATASET_DIR, 'train', 'train.csv')
ALLOCINE_DATASET_TEST = os.path.join(ALLOCINE_DATASET_DIR, 'test', 'test.csv')
ALLOCINE_DATASET_DEV = os.path.join(ALLOCINE_DATASET_DIR, 'dev', 'dev.csv')
ALLOCINE_DATASET = os.path.join(ALLOCINE_DATASET_DIR, 'dataset.csv')

EN_TO_CS_DIC = 'en_to_cs_dic.csv'
CS_TO_EN_DIC = 'cs_to_en_dic.csv'
EN_TO_FR_DIC = 'en_to_fr_dic.csv'
FR_TO_EN_DIC = 'fr_to_en_dic.csv'
CS_TO_FR_DIC = 'cs_to_fr_dic.csv'
FR_TO_CS_DIC = 'fr_to_cs_dic.csv'

ENGLISH = 'english'  # english
CZECH = 'czech'  # czech
FRENCH = 'french'
TOKENIZER_CHOICES = ['white_space', 'toktok', 'word_tokenizer', 'corpy']

PAD_TOKEN = '<pad>'  # PAD token
TAB_SEPARATOR = '\t'  # tabulator

TEXT = 'text'  # text
LABEL = 'label'  # label
DATASET_EXAMPLE_INDEX = 'dataset_indices'  # label text
ORIGINAL_TEXT = 'original_text'
MIN_FREQ_5 = 5  # minimal word frequency 5
MIN_FREQ_1 = 1  # minimal word frequency 1
RANDOM_VAR = 0.25  # treshold of random initialization for words missing in word2vec vocabulary
