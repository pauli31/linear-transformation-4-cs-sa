import csv
import logging
import random
import numpy as np
import os
import torch
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from torchtext.data import Field, Example, Dataset, BucketIterator
from tqdm import tqdm
import sys
import time
from sklearn.metrics import classification_report
from transformations.CanonicalCorrelationAnalysis import CanonicalCorrelationAnalysis
from transformations.LeastSquareTransformation import LeastSquareTransformation
from transformations.OrthogonalRankingTransformation import OrthogonalRankingTransformation
from transformations.OrthogonalTransformation import OrthogonalTransformation
from transformations.RankingTransformation import RankingTransformation

from config import RANDOM_SEED, CNN_CLASSIFIER, CROSS_LINGUAL_MODE, TEXT, LABEL, DATASET_EXAMPLE_INDEX, \
    LSTM_CLASSIFIER, ORIGINAL_TEXT, RANDOM_VAR, SOURCE_TO_TARGET, EN_TO_CS_DIC, CS_TO_EN_DIC, EMBEDDINGS_DIR, \
    FR_TO_CS_DIC, CS_TO_FR_DIC, EN_TO_FR_DIC, FR_TO_EN_DIC
from src.data.dictionary_loader import load_dictionary
from src.data.data_utils import get_pad_to_min_max_len_fn, get_tokenizer, tokenize_nn, dataset_get_x_y, get_sum_info
from src.data.loader import CzechFBDatasetLoader, CzechCSFDDatasetLoader, CzechMALLCZDatasetLoader, DatasetLoader, \
    CrossLingualCSFDIMDBDataset, CrossLingualIMDBCSFDDataset, CrossLingualSSTCSFDDataset, CrossLingualCSFDSSTDataset, \
    EnglishSSTDatasetLoader, EnglishIMDBDatasetLoader, FrenchAllocineDatasetLoader, CrossLingualAllocineSSTDataset, \
    CrossLingualAllocineIMDBDataset, CrossLingualAllocineCSFDDataset, CrossLingualCSFDAllocineDataset, \
    CrossLingualIMDBAllocineDataset, CrossLingualSSTAllocineDataset
from src.model.cnn import CNN
from src.model.lstm import LSTM
from src.pytorch.train import run_training, get_text_list_by_idx, get_predictions
from src.utils.general_utils import print_gpu_info, format_time, print_time_info
from src.utils.nn_utils import get_optimizer, get_lr_scheduler, evaluate_predictions, get_table_result_string, \
    load_embeddings, normalize

logger = logging.getLogger(__name__)


def print_error(args, texts, y_pred, y_test):
    error_path = args.result_file + "-errors.csv"
    # error_path = os.path.join('results', 'error.csv')
    logger.info('Printing error results to: ' + error_path)
    y_match = np.equal(y_test, y_pred)
    with open(error_path, 'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['text', 'predicted_label', 'true_label', 'matched_label'])
        for i, text in enumerate(texts):
            writer.writerow([text, y_pred[i].item(), y_test[i].item(), y_match[i].item()])
    logger.info('Error file written')


class TrainerNN(object):
    def __init__(self, args):
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch_num
        self.dataset_loader = DATASET_LOADERS[args.dataset_name](args.max_train_data, args.binary)
        self.use_only_train_data = args.use_only_train_data
        self.data_parallel = args.data_parallel

        # set seed if
        if args.use_random_seed is False:
            # init RANDOM_SEED
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)
            torch.cuda.manual_seed_all(RANDOM_SEED)

        if args.use_cpu is True:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Running fine-tuning on: {self.device}")
        print_gpu_info()

        try:
            GPUs_count = torch.cuda.device_count()
            logger.info("We have available more GPUs: " + str(GPUs_count))

            logger.info("We try to run it on multiple GPUs: " + str(self.data_parallel))

        except Exception as e:
            logger.info("Trying to init data parallelism")

        # Warning the Field is now deprecated but for our purposes it is enough since here we use only the LSTMs and CNNs
        self.label_field = Field(sequential=False, use_vocab=False)

        # We need index to the original dataset
        self.index_field = Field(sequential=False, use_vocab=False)

        # War
        # Init tokenizers
        self.source_tokenizer = get_tokenizer(args.tokenizer, self.dataset_loader.get_source_lang())
        self.source_text_field = self.prepare_text_field(args, self.source_tokenizer)
        self.source_fields = [(TEXT, self.source_text_field), (LABEL, self.label_field),
                              (DATASET_EXAMPLE_INDEX, self.index_field)]

        if self.dataset_loader.get_lang_mode() == CROSS_LINGUAL_MODE:
            self.target_tokenizer = get_tokenizer(args.tokenizer, self.dataset_loader.get_target_lang())
            self.target_text_field = self.prepare_text_field(args, self.target_tokenizer)
            self.target_fields = [(TEXT, self.target_text_field), (LABEL, self.label_field),
                                  (DATASET_EXAMPLE_INDEX, self.index_field)]
        else:
            self.target_tokenizer = None
            self.target_text_field = None
            self.target_fields = None

        self.num_labels = self.dataset_loader.get_class_num()

        # prepare vocab for source text_field

        # prepare vocab for target text_field
        if self.dataset_loader.get_lang_mode() == CROSS_LINGUAL_MODE:
            data_all_source = self.dataset_loader.load_entire_source_lang_dataset()
            self.prepare_vocab(data_all_source, use_source=True)
            data_all_target = self.dataset_loader.load_entire_target_lang_dataset()
            self.prepare_vocab(data_all_target, use_source=False)
        else:
            data_all_source = self.dataset_loader.load_entire_dataset()
            self.prepare_vocab(data_all_source, use_source=True)

        # Load dataset
        logger.info("Loading datasets")
        if self.use_only_train_data:
            # Train data
            data_train = self.dataset_loader.get_train_dev_data()
            data_dev = None
            self.train_size = len(data_train)
            self.dev_size = 0
        else:
            data_train = self.dataset_loader.get_train_data()
            data_dev = self.dataset_loader.get_dev_data()
            self.train_size = len(data_train)
            self.dev_size = len(data_dev)

        data_test = self.dataset_loader.get_test_data()
        self.test_size = len(data_test)

        # Train and dev data should be always from source language
        self.iterator_train = self.prepare_iterator(data_train, args, use_source=True)

        # Protoze v lstm obsahuje pack_padded_sequence metodu a tam to musi byt serazene...
        train_dev_test = False
        if args.model_name == 'lstm':
            train_dev_test = True

        self.iterator_dev = self.prepare_iterator(data_dev, args, use_source=True, train=train_dev_test)

        logger.info("Train size: " + str(self.train_size))
        logger.info("Dev size: " + str(self.dev_size))
        logger.info("Test size: " + str(self.test_size))

        # but test data can be from target language if there is the cross lingual mode/dataset
        if self.dataset_loader.get_lang_mode() == CROSS_LINGUAL_MODE:
            self.iterator_test = self.prepare_iterator(data_test, args, use_source=False, train=train_dev_test)
            data_dev_target_lang = self.dataset_loader.get_target_lang_dev_data()
            self.data_dev_target_len = len(data_dev_target_lang)
            self.iterator_dev_target_lang = self.prepare_iterator(data_dev_target_lang, args, use_source=False, train=train_dev_test)
            logger.info("Size of dev data for target language: " + str(self.dataset_loader.get_target_lang()) + " is:" +
                        str(self.data_dev_target_len))
            logger.info("Source language: " + str(self.dataset_loader.get_source_lang()))
            logger.info("Target language: " + str(self.dataset_loader.get_target_lang()))
        else:
            self.iterator_test = self.prepare_iterator(data_test, args, use_source=True, train=train_dev_test)
            self.iterator_dev_target_lang = None
            self.data_dev_target_len = 0

        X_train, y_train = dataset_get_x_y(data_train, 'text', 'label')
        print(
            "Train data: " + str(
                get_sum_info(X_train, data_train['label_text'], self.dataset_loader.get_class_names())))

        self.embeddings = load_embeddings(args.embeddings, args.embeddings_type, args.normalize_before, args)

        if self.dataset_loader.get_lang_mode() == CROSS_LINGUAL_MODE:
            if args.target_embeddings is None:
                raise Exception("Target embeddings path not set")

            self.target_embeddings = load_embeddings(args.target_embeddings, args.embeddings_type, args.normalize_before, args)
            if args.transformation_type == SOURCE_TO_TARGET:
                X, Y = self.get_learning_matrices(self.embeddings, self.target_embeddings, args.dictionary_size)
            else:
                X, Y = self.get_learning_matrices(self.target_embeddings, self.embeddings, args.dictionary_size)
            tr_method = None
            if self.args.transformation == 'cca':
                tr_method = CanonicalCorrelationAnalysis()
            elif self.args.transformation == 'lst':
                tr_method = LeastSquareTransformation()
            elif self.args.transformation == 'orthogonal':
                tr_method = OrthogonalTransformation()
            elif self.args.transformation == 'ranking':
                tr_method = RankingTransformation()
            elif self.args.transformation == 'orto-ranking':
                tr_method = OrthogonalRankingTransformation()
            else:
                raise Exception("Unknown transformation method:" + str(self.args.transformation))

            W = tr_method.compute_T_mat(X, Y)
            W = np.transpose(W)

            if self.args.transformation_type == SOURCE_TO_TARGET:
                self.set_embeddings(self.embeddings, args, self.source_text_field, W)
                self.set_embeddings(self.target_embeddings, args, self.target_text_field)
            else:
                self.set_embeddings(self.embeddings, args, self.source_text_field)
                self.set_embeddings(self.target_embeddings, args, self.target_text_field, W)

            # Editing the matrix indexes and adding embeddings from both languages to the matrix
            for token in self.target_text_field.vocab.itos:
                self.target_text_field.vocab.stoi[token] += (len(self.source_text_field.vocab.itos))
            self.vectors_to_net = torch.cat(
                [self.source_text_field.vocab.vectors, self.target_text_field.vocab.vectors])
        else:
            self.set_embeddings(self.embeddings, args, self.source_text_field)
            self.vectors_to_net = self.source_text_field.vocab.vectors

    def get_dictionary_path(self):
        """
        Returns path to required dictionary

        :return: path to dictionary
        """
        dictionary_path = ''
        if self.args.dataset_name == 'imdb-csfd' or self.args.dataset_name == 'sst-csfd':
            if self.args.transformation_type == SOURCE_TO_TARGET:
                dictionary_path = EN_TO_CS_DIC
            else:
                dictionary_path = CS_TO_EN_DIC
        elif self.args.dataset_name == 'csfd-imdb' or self.args.dataset_name == 'csfd-sst':
            if self.args.transformation_type == SOURCE_TO_TARGET:
                dictionary_path = CS_TO_EN_DIC
            else:
                dictionary_path = EN_TO_CS_DIC
        elif self.args.dataset_name == 'allocine-csfd':
            if self.args.transformation_type == SOURCE_TO_TARGET:
                dictionary_path = FR_TO_CS_DIC
            else:
                dictionary_path = CS_TO_FR_DIC
        elif self.args.dataset_name == 'csfd-allocine':
            if self.args.transformation_type == SOURCE_TO_TARGET:
                dictionary_path = CS_TO_FR_DIC
            else:
                dictionary_path = FR_TO_CS_DIC
        elif self.args.dataset_name == 'imdb-allocine' or self.args.dataset_name == 'sst-allocine':
            if self.args.transformation_type == SOURCE_TO_TARGET:
                dictionary_path = EN_TO_FR_DIC
            else:
                dictionary_path = FR_TO_EN_DIC
        elif self.args.dataset_name == 'allocine-imdb' or self.args.dataset_name == 'allocine-sst':
            if self.args.transformation_type == SOURCE_TO_TARGET:
                dictionary_path = FR_TO_EN_DIC
            else:
                dictionary_path = EN_TO_FR_DIC

        return dictionary_path

    def get_learning_matrices(self, source_embeddings, target_embeddings, dic_size):
        """
        Returns matrices that are than used to create transformation matrix

        :param source_embeddings: embeddings of source language
        :param target_embeddings: embeddings of target language
        :param dic_size: number of words from dictionary used to construct transformation matrix
        :return: matrices of the vectors of words of the source language and the vectors of their translation in the target language
        """
        logger.info('\nGetting matrices for transformation...')

        dictionary_path = self.get_dictionary_path()

        dic = load_dictionary(dictionary_path)
        if self.args.embeddings_type == 'fasttext':
            source_words = source_embeddings.get_words()
            target_words_set = set(target_embeddings.get_words())
        else:
            source_words = source_embeddings.vocab
            target_words_set = set(target_embeddings.vocab)

        X = np.zeros(shape=(dic_size, self.args.embeddings_size))
        Y = np.zeros(shape=(dic_size, self.args.embeddings_size))
        done = 0
        for word in source_words:
            if word not in dic:
                continue
            if dic[word] not in target_words_set:
                continue
            X[done] = source_embeddings[word]
            Y[done] = target_embeddings[dic[word]]
            done += 1
            if done == dic_size:
                break
        if done < dic_size:
            X = X[:done]
            Y = Y[:done]
        logger.info('Matrices initialized.')
        return X, Y

    def set_embeddings(self, embeddings, args, text_field, W=None):

        vectors = []
        if args.embeddings_type == 'w2v':
            for token in tqdm(text_field.vocab.itos, file=sys.stdout):
                if token in embeddings.vocab:
                    if W is None:
                        vectors.append(torch.FloatTensor(np.array(embeddings[token])))
                    else:
                        vectors.append(torch.FloatTensor(np.array(W @ embeddings[token])))
                else:
                    vectors.append(
                        torch.FloatTensor(np.random.uniform(-RANDOM_VAR, RANDOM_VAR, args.embeddings_size)))
        elif args.embeddings_type == 'fasttext':
            for token in tqdm(text_field.vocab.itos, file=sys.stdout):
                if W is None:
                    vectors.append(torch.FloatTensor(np.array(embeddings[token])))
                else:
                    vectors.append(torch.FloatTensor(np.array(W @ embeddings[token])))
        else:
            raise Exception("Unknown embeddings type: " + args.embeddings_type)

        # np.linalg.norm(vectors_X, axis=1).reshape((vectors_X.shape[0], 1))
        # np.linalg.norm(vectors_X_norm, axis=1).reshape((vectors_X_norm.shape[0], 1))
        if args.normalize_after is True:
            logger.info("We normalize the embeddings after the transformation")
            # create np array of vectors the matrix
            vectors_X = torch.stack(vectors).numpy()
            vectors_X_norm, _, _ = normalize(vectors_X)

            # now we convert it back to list of tensors
            vectors_new = []
            for i in tqdm(range(len(vectors_X_norm)), file=sys.stdout):
                vec = vectors_X_norm[i]
                vectors_new.append(torch.FloatTensor(vec))

            if len(vectors_new) != len(vectors):
                raise Exception("The len of normalized vectors should be the same")

            vectors = vectors_new

        # we want to keep unk and pad token as zeros
        vectors[text_field.vocab[text_field.pad_token]] = torch.FloatTensor(np.zeros(args.embeddings_size))
        vectors[text_field.vocab[text_field.unk_token]] = torch.FloatTensor(np.zeros(args.embeddings_size))

        text_field.vocab.set_vectors(text_field.vocab.stoi, vectors, args.embeddings_size)
        logger.info('Vocabulary built, word vectors assigned.')

    def get_fields(self, use_source=True):
        fields = None
        text_field = None
        if use_source is True:
            # use source
            fields = self.source_fields
            text_field = self.source_text_field
        else:
            # use target
            fields = self.target_fields
            text_field = self.target_text_field

        return fields, text_field

    def prepare_vocab(self, data_df, use_source=True):
        """

        :param data_df: it should be the entire dataset, because these data are used for building the
                        vocabulary
        :param args:
        :param use_source: whether the text is source or not, then we can decide
                            which tokenizer we use or which field we use
        :return:
        """
        X, y = dataset_get_x_y(data_df, 'text', 'label')
        fields, text_field = self.get_fields(use_source)

        examples = []

        for index, (text, label) in enumerate(zip(X, y)):
            example = Example.fromlist([text, label, index], fields=fields)
            examples.append(example)

        dataset = Dataset(examples, fields=fields)
        text_field.build_vocab(dataset, min_freq=self.args.min_freq)

    def prepare_iterator(self, data_df, args, use_source=True,train=True):
        """
        :param data_df:
        :param args:
        :param use_source: whether the text is source or not, then we can decide
                            which tokenizer we use or which field we use
        :return:
        """
        if data_df is None:
            logger.info("The data_df are None, return None")
            return None

        X, y = dataset_get_x_y(data_df, 'text', 'label')
        fields, text_field = self.get_fields(use_source)

        examples = []

        for index, (text, label) in enumerate(zip(X, y)):
            example = Example.fromlist([text, label, index], fields=fields)
            examples.append(example)

        # BucketIterator explained https://gmihaila.medium.com/better-batches-with-pytorchtext-bucketiterator-12804a545e2a
        dataset = Dataset(examples, fields=fields)

        iterator = BucketIterator(dataset, batch_size=args.batch_size, sort_key=lambda x: len(x.text),
                                  sort_within_batch=train, sort=False, train=train, shuffle=train)

        # for batch in tqdm(iterator_train, file=sys.stdout):
        #     input_ids = batch.text[0]
        #     labels = batch.label

        return iterator

    def prepare_text_field(self, args, tokenizer):
        logger.info("Preparing text field")
        fix_len = None
        if args.model_name == CNN_CLASSIFIER:
            min_len_padding = get_pad_to_min_max_len_fn(min_length=max(args.filter_sizes), max_length=args.max_seq_len)
        else:
            min_len_padding = None
            if args.max_seq_len > 0:
                fix_len = args.max_seq_len

        tokenize = tokenize_nn(tokenizer)
        text_field = Field(batch_first=True, include_lengths=True, postprocessing=min_len_padding, tokenize=tokenize,
                           lower=args.lowercase, fix_length=fix_len)

        return text_field

    def train_nn_model(self, args):
        model = self.initialize_model(args)

        if args.data_parallel is True:
            if torch.cuda.device_count() > 1:
                logger.info("Trying to apply data parallelism, number of used GPUs: " + str(torch.cuda.device_count()))
                # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
                model = torch.nn.DataParallel(model)
            else:
                logger.info("Data parallelism is enabled but there is only GPUs: " + str(torch.cuda.device_count()))

        # move it to device
        model = model.to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

        # Init optimizer
        optimizer = get_optimizer(model.parameters(), args)
        total_steps = len(self.iterator_train) * self.epoch_num
        scheduler = get_lr_scheduler(args, optimizer, total_steps)

        logger.info('Training model on train data...')
        t0 = time.time()

        history = run_training(args, model, self.iterator_train, self.iterator_dev, loss_fn,
                               optimizer, scheduler, self.device, self.iterator_dev_target_lang)

        train_time = time.time() - t0
        logger.info(f'Total time for training: {format_time(train_time)}')

        self.perform_train_eval(model, args, train_time, args.epoch_num)

        return model, history

    def evaluate_fine_tuned_model(self, args):
        # TODO
        raise NotImplementedError("Not implemented yet, so do it now")

    def initialize_model(self, args):
        if args.model_name == LSTM_CLASSIFIER:
            net = LSTM(pretrained_embeddings=self.vectors_to_net, bidirectional=args.bidirectional,
                       num_layers=args.num_layers, outputs=self.num_labels, hidden_size=args.hidden_size,
                       dropout_value=args.dropout, embeddings_size=args.embeddings_size)
        elif args.model_name == CNN_CLASSIFIER:
            net = CNN(pretrained_embeddings=self.vectors_to_net, outputs=self.num_labels, dropout_value=args.dropout,
                      embeddings_size=args.embeddings_size, num_filters=args.num_filters,
                      filter_sizes=args.filter_sizes)
        else:
            raise Exception("Unknown model name: " + str(args.model_name))

        try:
            # pp = 0
            # for p in list(model.parameters()):
            #     nn = 1
            #     for s in list(p.size()):
            #         nn = nn * s
            #     pp += nn
            model_parameters = filter(lambda p: p.requires_grad, net.parameters())
            pp = sum([np.prod(p.size()) for p in model_parameters])
            logger.info("Number of parameters for model:" + str(args.model_name) + " is:" + str(pp))
        except Exception as e:
            logger.error("Error during count number:" + str(e))

        logger.info("Model initialized")
        return net

    def print_dataset_info(self, args):
        dataset_df = self.dataset_loader.load_entire_dataset()

        # just print some example
        sentence = dataset_df['text'][150]
        # ids = self.tokenizer.encode(sentence, max_length=args.max_seq_len, pad_to_max_length=True)

        logger.info(f' Sentence: {sentence}')
        logger.info(f'   Tokens: {sentence.split(" ")}')
        # TODO
        # logger.info(f'   Tokens: {self.tokenizer.convert_ids_to_tokens(ids)}')
        # logger.info(f'Token IDs: {ids}')

        if self.args.draw_dataset_stats is True:
            # logger.info(f"Saving dataset tokens histogram for tokenizer: {self.args.tokenizer_type}")
            logger.info(f"Saving dataset tokens histogram")
            # See distribution of text len
            token_lens = []

            count_i = 0
            for txt in dataset_df.text:
                tokens = txt.split(" ")
                # tokens = self.tokenizer.encode(txt)
                token_lens.append(len(tokens))
                count_i = count_i + 1
                if count_i % 1000 == 0 and count_i > 0:
                    logger.info("Processed: " + str(count_i))

            max_len = max(token_lens)
            avg_len = np.mean(token_lens)
            cnt = Counter(token_lens)
            # sort by key
            cnt = sorted(cnt.items())
            print("Sentence len - Counts")

            dataset_name = args.dataset_name
            if dataset_name == 'combined':
                tmp = '-'.join(args.combined_datasets)
                dataset_name = dataset_name + '-' + tmp

            model_name = args.model_name
            model_name = model_name.replace('/', '-')
            # TODO
            tokenizer = 'Space'
            prefix = dataset_name + '_' + model_name + '-' + tokenizer + '-'
            histogram_file = os.path.join(self.dataset_loader.get_dataset_dir(), prefix + 'histogram.txt')

            with open(histogram_file, mode='w', encoding='utf-8') as f:
                f.write("Average len: {:.4f}".format(avg_len) + '\n')
                f.write("Max len: " + str(max_len) + '\n')
                f.write('length - count' + '\n')
                for (length, count) in cnt:
                    # print()
                    f.write(str(length) + ' - ' + str(count) + '\n')

            logger.info(f"Max tokens len: {max_len}")
            logger.info(f"Avg tokens len: {avg_len}")

            tokens_histogram_path = os.path.join(self.dataset_loader.get_dataset_dir(), prefix + 'tokens_histogram.png')
            logger.info(f"Tokens histogram image saved to: {tokens_histogram_path}")
            plt.figure()  # it resets the plot
            figure = sns.distplot(token_lens).get_figure()
            plt.xlim([0, 512])
            plt.xlabel(f"Token count, max len: {max_len}")
            figure.savefig(tokens_histogram_path, dpi=400)
            plt.figure()

    def perform_train_eval(self, model, args, train_time, curr_epoch):
        test_data_iterator = self.iterator_test
        test_data = self.dataset_loader.get_test_data()

        t0 = time.time()
        y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_iterator, test_data,
                                                                       self.device, args.eval_batch_size)
        eval_time = time.time() - t0
        print_time_info(eval_time, len(test_data), "entire_testing_evaluation", file=args.result_file)

        clas_report = classification_report(y_test, y_pred, target_names=self.dataset_loader.get_class_names())
        if args.print_error:
            print_error(args, y_review_texts, y_pred, y_test)
        print(clas_report)
        logger.info(clas_report)

        f1, accuracy, precision, recall = evaluate_predictions(y_pred, y_test)

        dataset_name = args.dataset_name
        result_string, only_results = get_table_result_string(
            f'{dataset_name}\tTrain/Test:{args.model_name} {args} curr_epoch:{str(curr_epoch)}',
            f1, accuracy, precision, recall, train_time)


        result_string = "\n-----------Test Results------------\n\t" + result_string

        # when we have the dev data
        if self.use_only_train_data is False:
            dev_data_iterator = self.iterator_dev
            dev_data = self.dataset_loader.get_dev_data()

            _, y_dev_pred, _, y_dev = get_predictions(model, dev_data_iterator, dev_data,
                                                      self.device, args.eval_batch_size)

            f1_dev, accuracy_dev, precision_dev, recall_dev = evaluate_predictions(y_dev_pred, y_dev)

            result_string_dev, only_results_dev = get_table_result_string(
                "Dev results", f1_dev, accuracy_dev, precision_dev, recall_dev, 0)

            only_results += "\t" + only_results_dev
            result_string += "\n-----------Dev Results------------\n" + result_string_dev

        # eval cross-lingual results
        if self.dataset_loader.get_lang_mode() == CROSS_LINGUAL_MODE:
            data_dev_target_iterator = self.iterator_dev_target_lang
            data_dev_target = self.dataset_loader.get_target_lang_dev_data()

            _, y_dev_pred_trg, _, y_dev_trg = get_predictions(model, data_dev_target_iterator, data_dev_target,
                                                              self.device, args.eval_batch_size)

            f1_dev_trg, accuracy_dev_trg, precision_dev_trg, recall_dev_trg = evaluate_predictions(y_dev_pred_trg,
                                                                                                   y_dev_trg)
            target_lang = self.dataset_loader.get_target_lang()
            source_lang = self.dataset_loader.get_source_lang()

            result_string_dev_trg, only_results_dev_trg = get_table_result_string(
                "Dev results target: " + str(target_lang),
                f1_dev_trg,
                accuracy_dev_trg,
                precision_dev_trg,
                recall_dev_trg,
                0)

            print(70 * '*-')
            print("Test {} Test F1: {:.4f}".format(target_lang, f1))
            print("Test {} Test accuracy: {:.4f}".format(target_lang, accuracy))
            print("Test {} Test precision: {:.4f}".format(target_lang, precision))
            print("Test {} Test recall: {:.4f}".format(target_lang, recall))
            print(70 * '*-')

            print(70 * '*-')
            print("DEV {} Test F1: {:.4f}".format(target_lang, f1_dev_trg))
            print("DEV {} Test accuracy: {:.4f}".format(target_lang, accuracy_dev_trg))
            print("DEV {} Test precision: {:.4f}".format(target_lang, precision_dev_trg))
            print("DEV {} Test recall: {:.4f}".format(target_lang, recall_dev_trg))
            print(70 * '*-')

            try:
                print(70 * '*-')
                print("DEV {}  F1: {:.4f}".format(source_lang, f1_dev))
                print("DEV {}  accuracy: {:.4f}".format(source_lang, accuracy_dev))
                print("DEV {}  precision: {:.4f}".format(source_lang, precision_dev))
                print("DEV {}  recall: {:.4f}".format(source_lang, recall_dev))
                print(70 * '*-')
            except Exception:
                logger.info("No dev data for source lang: " + str(source_lang))

            only_results += "\t" + only_results_dev_trg
            result_string += "\n-----------target: " + str(
                target_lang) + " Dev Results------------\n" + result_string_dev_trg

        print("\n\n\n-----------Save results------------\n" + str(only_results) + "\n\n\n")
        results_file = args.result_file

        # write to disk
        with open(results_file, "a", encoding='utf-8') as f:
            f.write(only_results + "\n")


        print(result_string)
        logger.info(result_string)

        # save_model()


DATASET_LOADERS = {
    "fb": CzechFBDatasetLoader,
    "csfd": CzechCSFDDatasetLoader,
    "mallcz": CzechMALLCZDatasetLoader,
    "imdb": EnglishIMDBDatasetLoader,
    "sst": EnglishSSTDatasetLoader,
    "allocine": FrenchAllocineDatasetLoader,
    "imdb-csfd": CrossLingualIMDBCSFDDataset,
    "csfd-imdb": CrossLingualCSFDIMDBDataset,
    "sst-csfd": CrossLingualSSTCSFDDataset,
    "csfd-sst": CrossLingualCSFDSSTDataset,
    "allocine-sst": CrossLingualAllocineSSTDataset,
    "allocine-imdb": CrossLingualAllocineIMDBDataset,
    "allocine-csfd": CrossLingualAllocineCSFDDataset,
    "csfd-allocine": CrossLingualCSFDAllocineDataset,
    "imdb-allocine": CrossLingualIMDBAllocineDataset,
    "sst-allocine": CrossLingualSSTAllocineDataset
}
