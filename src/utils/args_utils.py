import argparse

from config import DATASET_NAMES, MODEL_NAMES, SCHEDULERS, OPTIMIZERS_CHOICES, TRANSFORMATIONS, MODES, LOG_DIR, \
    LOGGING_FORMAT, LOGGING_DATE_FORMAT, MONOLINGUAL_MODE, TOKENIZER_CHOICES, MIN_FREQ_1, MIN_FREQ_5, SOURCE_TO_TARGET, \
    TRANSFORMATION_TYPES
import os
import logging

from src.utils.general_utils import generate_file_name


def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False, description='Cross-lingual sentiment polarity detection')

    # Required parameters
    parser.add_argument('--dataset_name',
                        required=True,
                        choices=DATASET_NAMES,
                        help="The dataset that will be used, they correspond to names of folders")

    parser.add_argument('--model_name',
                        required=True,
                        choices=MODEL_NAMES,
                        help='Name of the model')

    parser.add_argument('--embeddings',
                        required=True,
                        help='Path to embeddings')

    parser.add_argument('--embeddings_type',
                        required=True,
                        choices=['fasttext', 'w2v'],
                        help='Type of word embeddings')

    # Optional
    parser.add_argument('--target_embeddings',
                        action='store',
                        help='Path to target language embeddings when cross-lingual mode is used')

    parser.add_argument("--draw_dataset_stats",
                        action="store_true",
                        help="If specified the statistics about the given datasets are printed and saved to dataset folder"
                             "don't forget to specify the correct tokenizer, it can be slower because it loads the entire dataset"
                             " and it tokenizes it, The fine-tuning is not run with this parameter")

    # TODO
    parser.add_argument("--transformation",
                        default='none',
                        choices=TRANSFORMATIONS,
                        help='Transformation that will be used, only used when --mode is set to crosslingual')

    parser.add_argument("--mode",
                        default=MONOLINGUAL_MODE,
                        choices=MODES,
                        help='Mode that will be used for training, if monolingual only monolingual model is trained and evaluated')

    parser.add_argument("--transformation_type",
                        default=SOURCE_TO_TARGET,
                        choices=TRANSFORMATION_TYPES,
                        help='Direction of transformation - from source language to target or from target language to source')

    parser.add_argument("--data_parallel",
                        default=False,
                        action='store_true',
                        help='If set, the program will run on all available GPUs')

    parser.add_argument("--use_cpu",
                        default=False,
                        action='store_true',
                        help="If set, the program will always run on CPU")

    parser.add_argument("--silent",
                        default=False,
                        action='store_true',
                        help="If used, logging is set to ERROR level, otherwise INFO is used")

    parser.add_argument("--num_repeat",
                        default=1,
                        type=int,
                        help="Specify the number, of how many times will be the experiment repeated")

    parser.add_argument("--use_random_seed",
                        default=False,
                        action='store_true',
                        help="If set, the program will NOT set a seed value for all random sources, "
                             "if set the results should NOT be same across runs with the same configuration.")

    # TODO Not implemented
    parser.add_argument("--full_mode",
                        default=False,
                        action='store_true',
                        help="If set, the program will evaluate the model after each epoch and write results into the result files")

    parser.add_argument("--binary",
                        default=False,
                        action='store_true',
                        help="If used the polarity task is treated as binary classification, i.e., positive/negative"
                             " The neutral examples are dropped")

    parser.add_argument("--epoch_num",
                        default=5,
                        type=int,
                        help="Number of epochs for fine tuning")

    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Train batch size")

    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Evaluation batch size, can be larger since there is no need to remember gradients -> more memory")

    parser.add_argument("--lr",
                        default=1e-3,
                        type=float,
                        help="Learning rate")

    parser.add_argument("--max_seq_len",
                        default=0,
                        type=int,
                        help="Maximum sequence length of tokens  used as an input for the model")

    parser.add_argument("--lowercase",
                        default=False,
                        action='store_true',
                        help="Use lowercase")

    parser.add_argument("--min_freq",
                        default=MIN_FREQ_1,
                        choices=[MIN_FREQ_1, MIN_FREQ_5],
                        help="Minimal frequency of words used for classification")

    parser.add_argument("--scheduler",
                        default='linear_wrp',
                        choices=SCHEDULERS,
                        type=str,
                        help="Scheduler used for scheduling learning rate,"
                             " see https://huggingface.co/transformers/main_classes/optimizer_schedules.html#learning-rate-schedules-pytorch")

    parser.add_argument("--print_stat_frequency",
                        default=64,
                        type=int,
                        help="Specify the frequency of printing train info, i.e. after how many batches will be the "
                             "info printed")

    parser.add_argument("--normalize_before",
                        default=False,
                        action="store_true",
                        help="If set, the vectors are normalized before the transformation ")


    parser.add_argument("--normalize_after",
                        default=False,
                        action='store_true',
                        help="If set, the vectors are normalized after the transformation")

    parser.add_argument("--disable_clip_grad_norm",
                        default=True,
                        action='store_false',
                        help="If set, clip gradient norm is NOT used, default True")

    parser.add_argument("--warm_up_steps",
                        default=0,
                        type=float,
                        help="Number of warmup steps, if less than 1 than it is used as percents/fraction of the total"
                             " number of steps, cannot be set to one")

    parser.add_argument("--optimizer",
                        default='AdamW',
                        choices=OPTIMIZERS_CHOICES,
                        help="Optimizer one of: " + str(OPTIMIZERS_CHOICES))

    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay parameter, is only applied when AdamW optimizer is used")

    parser.add_argument("--adadelta_rho",
                        default=0.6,
                        type=float,
                        help="Rho value for the adadelta optimizer, is only applied when Adadelta is used")

    parser.add_argument("--max_train_data",
                        default=-1,
                        type=float,
                        help="Amount of data that will be used for training, "
                             "if (-inf, 0> than it is ignored, "
                             "if (0,1> than percentage of the value is used as training data, "
                             "if (1, inf) absolute number of training examples is used as training data")
    # TODO
    parser.add_argument("--eval",
                        default=False,
                        action='store_true',
                        help="If used, the evaluation on a given dataset and model is performed,"
                             "if used all other param \"--model_name\" must be a path to the model folder")

    # Watch the cross-lingual datasets
    parser.add_argument("--use_only_train_data",
                        default=False,
                        action='store_true',
                        help="If set, the program will use training and development data for training, i.e. it will use"
                             "train + dev for training, no validation is done during training")

    parser.add_argument("--hidden_size",
                        default=512,
                        type=int,
                        help="Size of hidden layer inside LSTM")

    parser.add_argument('--num_layers',
                        default=1,
                        type=int,
                        help='Number of layers in LSTM')

    parser.add_argument('--bidirectional',
                        action='store_true',
                        default=False,
                        help='use bidirectional LSTM')

    parser.add_argument("--dropout",
                        default=0.5,
                        type=float,
                        help="Dropout for neural network")

    parser.add_argument("--embeddings_size",
                        default=300,
                        type=int,
                        help="Size of word embeddings")

    parser.add_argument("--num_filters",
                        default=256,
                        type=int,
                        help="Number of filters for CNN")

    parser.add_argument("--filter_sizes",
                        nargs='+',
                        type=int,
                        default=[2, 3, 4],
                        help="Filter sizes for CNN")

    # Preprocessing options
    parser.add_argument("--tokenizer",
                        default='corpy',
                        choices=TOKENIZER_CHOICES,
                        help="Possible tokenizers that are used for tokenization")

    parser.add_argument('--dictionary_size',
                        default=20000,
                        type=int,
                        help='Size of dictionary')

    parser.add_argument('--print_error',
                        action='store_true',
                        default=False,
                        help='Print error')

    return parser


def init_logging(args, parser, result_file, generating_fce=generate_file_name, set_format=True):
    config_name = generating_fce(args)
    if args.full_mode is True:
        # select the first epoch config name
        file_name = os.path.join(LOG_DIR, config_name[1] + '-full_mode.log')
    else:
        file_name = os.path.join(LOG_DIR, config_name + '.log')
    parser.add_argument("--config_name",
                        default=config_name)
    parser.add_argument("--result_file",
                        default=result_file)
    args = parser.parse_args()

    if set_format:
        # just to reset logging settings
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(format=LOGGING_FORMAT,
                            datefmt=LOGGING_DATE_FORMAT,
                            filename=file_name)

        formatter = logging.Formatter(fmt=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logging.root.setLevel(level=logging.INFO)
        if args.silent is True:
            # logging.root.setLevel(level=logging.ERROR)
            console_handler.setLevel(level=logging.ERROR)
        else:
            # logging.root.setLevel(level=logging.INFO)
            console_handler.setLevel(level=logging.INFO)

        logging.getLogger().addHandler(console_handler)

    return args
