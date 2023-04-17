import logging

from src.nn_train import TrainerNN

logger = logging.getLogger(__name__)


def train_model(args, result_file):
    logger.info(f"Running training with the following parameters: {args}")
    logger.info("-------------------------")

    if args.model_name == 'cnn' or args.model_name == 'lstm':
        nn_trainer = TrainerNN(args)
        if args.draw_dataset_stats is True:
            logger.info("Dataset stats saved")
        elif args.eval is True:
            nn_trainer.evaluate_fine_tuned_model(args)
        else:
            nn_trainer.train_nn_model(args)
    else:
        raise Exception("Unknown model name: " + str(args.model_name))

    # Init data loading
