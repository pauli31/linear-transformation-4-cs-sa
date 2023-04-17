import math
import os
import numpy as np

import torch
import logging
import fasttext
import fasttext.util
from fasttext.FastText import _FastText
from gensim.models import KeyedVectors, utils_any2vec
from gensim.models.word2vec import Word2Vec

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import EMBEDDINGS_DIR
from src.utils.transformers_optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_constant_schedule, get_transformer_polynomial_decay_schedule_with_warmup

logger = logging.getLogger(__name__)


def normalize(X, mean_centering=True, unit_vectors=True, unit_vec_features=False):
    """
    Normalize given ndarray

    :param X: ndarray representing semantic space,
                axis 0 (rows)       - vectors for words
                axis 1 (columns)    - elements of word vectors
    :param mean_centering: if true values are centered around zero
    :param unit_vectors: is true vectors are converted to unit vectors

    :return: normalized ndarray
    """
    mean = None
    norms = None

    if mean_centering is True:
        # mean vector and normalization
        mean = X.sum(0) / X.shape[0]
        X = X - mean

    if unit_vectors is True:
        # compute norm
        # norms = np.sqrt((X ** 2).sum(-1))[..., np.newaxis]
        if unit_vec_features is False:
            # normalization over vectors of words
            norms = np.linalg.norm(X, axis=1).reshape((X.shape[0], 1))
        else:
            # normalization over features (columns)
            norms = np.linalg.norm(X, axis=0).reshape((1, X.shape[1]))
        X = X / norms

    return X, mean, norms


class NormalizedFastText(object):

    def __init__(self, fasttext, args, mean_centering=True, unit_vectors=True, unit_vec_features=False):
        self.fasttext = fasttext
        self.args = args
        self.mean_centering = mean_centering
        self.unit_vectors = unit_vectors
        self.unit_vec_features = unit_vec_features
        self.words_set = set(self.fasttext.get_words())
        # self.normalized_X, self.normalized_mean, self.normalized_norms = self.compute_normalization()
        _, self.normalized_mean, self.normalized_norms = self.compute_normalization()


    def compute_normalization(self):
        words = self.fasttext.get_words()
        words_size = len(words)

        X = np.zeros(shape=(words_size, self.args.embeddings_size))

        for i, word in enumerate(words) :
            index = self.fasttext.get_word_id(word)
            if index != i:
                raise Exception("Failed check index:" + str(index) + " i:" + str(i) + " word:" + str(word))
            X[i] = self.fasttext[word]

        X_norm, mean, norms = normalize(X, mean_centering=self.mean_centering,
                                        unit_vectors=self.unit_vectors,
                                        unit_vec_features=self.unit_vec_features)
        return X_norm, mean, norms

    def get_dimension(self):
        return self.fasttext.get_dimension()

    def get_words(self, include_freq=False, on_unicode_error='strict'):
        return self.fasttext.get_words(include_freq=include_freq, on_unicode_error=on_unicode_error)

    # TODO check slovao <unk>
    def get_word_vector(self, word):
        # if word in self.words_set:
        #     word_id = self.fasttext.get_word_id(word)
        #     vector = self.normalized_X[word_id]
        # else:
        #     vector = self.fasttext.get_word_vector(word)
        #
        #     if self.mean_centering:
        #         vector = vector - self.normalized_mean
        #
        #     if self.unit_vectors is True:
        #         if self.unit_vec_features is False:
        #             norm = np.linalg.norm(vector)
        #             vector = vector / norm
        #         else:
        #             vector = vector / self.normalized_norms

        vector = self.fasttext.get_word_vector(word)

        if self.mean_centering:
            vector = vector - self.normalized_mean

        if self.unit_vectors is True:
            if self.unit_vec_features is False:
                norm = np.linalg.norm(vector)
                vector = vector / norm
            else:
                vector = vector / self.normalized_norms

        return vector

    @property
    def words(self):
        if self._words is None:
            self._words = self.get_words()
        return self._words

    def __getitem__(self, word):
        return self.get_word_vector(word)

    def __contains__(self, word):
        return word in self.words



    # //pak tady musim prepsat kazdou metodu toho Fasttext objektu


def load_embeddings(model_name, type='fasttext', normalize_vec=False, args=None):
    """

    :param model_name: embeddings name in embeddings directory
    :param type: fasttext or w2v
    :return: loaded embeddings
    """
    emb_target_path = os.path.join(EMBEDDINGS_DIR, model_name)
    if type == 'fasttext':
        embeddings = fasttext.load_model(emb_target_path)
        if normalize_vec is True:
            normalized_emb = NormalizedFastText(embeddings, args)
            print("Built normalized embeddings")
            embeddings = normalized_emb

    elif type == 'w2v':
        logger = utils_any2vec.logger
        logger.setLevel(logging.WARNING)
        embeddings = KeyedVectors.load_word2vec_format(emb_target_path, binary=True)
        if normalize_vec is True:
            embeddings.vectors, _, _ = normalize(embeddings.vectors)
    else:
        raise Exception("Unknown embedding type: " + str(type))

    return embeddings


def get_table_result_string(config_string, mean_f1_macro, mean_acc, mean_prec, mean_recall, train_test_time):
    results = f'{config_string}\t{mean_f1_macro}\t{mean_acc}\t{mean_prec}\t{mean_recall}\t{int(train_test_time)} s'
    results_head = '\tF1 Macro\tAccuracy\tPrecision\tRecall\ttime\n' + results

    return results_head, results


def evaluate_predictions(y_pred, y_test, average='macro'):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=average)
    precision = precision_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)

    return f1, accuracy, precision, recall


def get_optimizer(model_params, args):
    """
    # Learning rate decay, can be implemented with tf.keras.optimizers.schedules.LearningRateSchedule


    :param optimizer_name:
    :param lr:  floating point value, or a schedule that is a
                `tf.keras.optimizers.schedules.LearningRateSchedule`,
    :return:
    """
    lr = args.lr
    optimizer_name = args.optimizer

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model_params, lr=lr)

    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model_params, lr=lr, weight_decay=args.weight_decay)

    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model_params, lr=lr)

    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(model_params, lr=lr)

    elif optimizer_name == 'Adadelta':
        optimizer = torch.optim.Adadelta(model_params, lr=lr, rho=args.adadelta_rho)

    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model_params, lr=lr)
    else:
        raise Exception('Not valid optimizer: ' + optimizer_name)

    return optimizer


def get_lr_scheduler(args, optimizer, total_steps):
    warm_up_steps = args.warm_up_steps
    scheduler_name = args.scheduler

    if warm_up_steps > 0:
        if warm_up_steps == 1:
            raise Exception("Warmup steps cannot be 1")
        if warm_up_steps < 1:
            warm_up_steps = warm_up_steps * total_steps
            warm_up_steps = math.ceil(warm_up_steps)

    logger.info("Number of warm up steps: " + str(warm_up_steps) + " out of: " + str(
        total_steps) + " original warmup steps: " + str(warm_up_steps))

    # https://huggingface.co/transformers/main_classes/optimizer_schedules.html#learning-rate-schedules-pytorch
    if scheduler_name == 'linear_wrp':
        # linearly decreasing
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=total_steps
        )
    elif scheduler_name == 'cosine_wrp':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=total_steps
        )
    elif scheduler_name == 'polynomial_wrp':
        scheduler = get_transformer_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=total_steps,
            power=2
        )
    elif scheduler_name == 'constant':
        scheduler = get_constant_schedule(optimizer)
    else:
        raise Exception(f"Unknown scheduler: {args.scheduler}")

    return scheduler
