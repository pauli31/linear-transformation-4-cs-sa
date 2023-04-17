import logging

import os
import sys
from pathlib import Path


from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT, DATASET_NAMES, MODEL_NAMES, SCHEDULERS, OPTIMIZERS_CHOICES, \
    RESULTS_DIR, EMBEDDINGS_DIR
from src.polarity_detection import train_model
from src.utils.args_utils import build_parser, init_logging
from src.utils.general_utils import generate_file_name

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)

# LOCAL_MODEL_NAME = 'lstm'
LOCAL_MODEL_NAME = 'cnn'


# Only for debugging, it will set the parameters
RUN_LOCAL = False


def main(result_file, args):
    """

    :param result_file: if ful mode then it is dictionary with key for each epoch
    :return:
    """
    print('Hello, Cross-lingual transform method for Polarity detection')

    parser = build_parser()
    args = parser.parse_args()

    args = init_logging(args, parser, result_file)
    logger.info(f"Running fine-tuning with the following parameters: {args}")
    logger.info("-------------------------")

    train_model(args, result_file)


def set_LSTM():
    sys.argv.extend(['--model_name', 'lstm'])
    sys.argv.extend(['--dataset_name', 'csfd'])
    sys.argv.extend(['--embeddings', 'word2vec_csfd.bin'])
    sys.argv.extend(['--embeddings_type', 'w2v'])
    # sys.argv.extend(['--use_cpu'])

    # sys.argv.extend(['--draw_dataset_stats'])
    pass


def set_CNN():
    # TODO
    sys.argv.extend(['--model_name', 'cnn'])
    # sys.argv.extend(['--dataset_name', 'sst-csfd'])
    sys.argv.extend(['--dataset_name', 'imdb-csfd'])
    # sys.argv.extend(['--dataset_name', 'imdb'])
    sys.argv.extend(['--binary'])
    sys.argv.extend(['--embeddings', 'fasttext_en.bin'])
    # sys.argv.extend(['--embeddings', 'fasttext_csfd.bin'])
    sys.argv.extend(['--embeddings_type', 'fasttext'])
    sys.argv.extend(['--target_embeddings', 'fasttext_csfd.bin'])
    # sys.argv.extend(['--target_embeddings', 'fasttext_en.bin'])
    sys.argv.extend(['--mode', 'crosslingual'])
    # sys.argv.extend(['--transformation', 'lst'])
    sys.argv.extend(['--transformation', 'lst'])
    # sys.argv.extend(['--transformation_type', 'target_to_source'])
    sys.argv.extend(['--lowercase'])
    sys.argv.extend(['--normalize_before'])
    sys.argv.extend(['--batch_size', '32'])
    sys.argv.extend(['--epoch_num', '10'])
    sys.argv.extend(['--print_error'])
    # sys.argv.extend(['--use_cpu'])
    sys.argv.extend(['--normalize_after'])
    sys.argv.extend(['--user', 'common'])
    # sys.argv.extend(['--dictionary_size', '5000'])
    sys.argv.extend(['--scheduler', 'constant'])
    sys.argv.extend(['--num_repeat', '2'])
    pass


def set_test():
    sys.argv.extend(['--user', 'Pauli'])
    sys.argv.extend(['--model_name', 'lstm'])
    sys.argv.extend(['--dataset_name', 'csfd-imdb'])
    # sys.argv.extend(['--embeddings', 'fasttext_en.bin'])
    sys.argv.extend(['--embeddings', 'fasttext_csfd.bin'])
    sys.argv.extend(['--embeddings_type', 'fasttext'])
    # sys.argv.extend(['--target_embeddings', 'fasttext_csfd.bin'])
    sys.argv.extend(['--target_embeddings', 'fasttext_en.bin'])
    sys.argv.extend(['--mode', 'crosslingual'])
    sys.argv.extend(['--transformation', 'cca'])
    sys.argv.extend(['--binary'])
    sys.argv.extend(['--print_error'])
    sys.argv.extend(['--lowercase'])
    sys.argv.extend(['--batch_size', '32'])
    sys.argv.extend(['--epoch_num', '5'])
    sys.argv.extend(['--max_seq_len', '0'])
    sys.argv.extend(['--use_random_seed'])
    sys.argv.extend(['--scheduler', 'constant'])
    sys.argv.extend(['--lr', '1e-5'])



if __name__ == '__main__':
    if RUN_LOCAL is True:
        if LOCAL_MODEL_NAME == 'lstm':
            set_LSTM()
        elif LOCAL_MODEL_NAME == 'cnn':
            set_CNN()
        else:
            raise Exception("Unknown local model name: " + LOCAL_MODEL_NAME)


    parser = build_parser()
    args = parser.parse_args()
    result_file = generate_file_name(args)

    result_dir = os.path.join(RESULTS_DIR, str(args.transformation) + '-' + str(args.dictionary_size) + '-' + str(args.transformation_type) + '-' + str(args.embeddings))
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Created a result dir:" + str(result_dir))

    if args.full_mode is True:
        for key, val in result_file.items():
            tmp_val = val + ".results"
            tmp_val = os.path.join(result_dir, tmp_val)
            result_file[key] = tmp_val
    else:
        result_file = result_file + ".results"
        result_file = os.path.join(result_dir, result_file)

    # +1 bcs of the range
    num_repeat = args.num_repeat + 1
    for repeat in range(1, num_repeat):
        logger.info("Running repeat:" + str(repeat))
        main(result_file, args)
        logger.info("Run completed")
        logger.info("----------------------------------------------------")

