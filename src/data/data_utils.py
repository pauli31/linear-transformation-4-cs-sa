from config import PAD_TOKEN, FRENCH
from nltk import WhitespaceTokenizer, word_tokenize, ToktokTokenizer
from corpy.morphodita import Tokenizer


def get_pad_to_min_max_len_fn(min_length, max_length):
    """
    Aligns data to minimum length - fill in blank characters for shorter data, trims when maximum length is exceeded

    :param min_length: minimal length to align data
    :param max_length: maximum length, if exceeded the data is trimmed
    :return: aligned data
    """

    def pad_to_min_max_len(batch, vocab, minimal_length=min_length, maximal_length=max_length):
        """
        Aligns data to minimal length by adding a <pad> token, trims when max length is exceeded

        :param batch: data
        :param vocab: vocabulary of field
        :param minimal_length: minimal length
        :param maximal_length: maximal length
        :return: aligned data
        """
        pad_idx = vocab.stoi[PAD_TOKEN]
        for idx, ex in enumerate(batch):
            if len(ex) < minimal_length:
                batch[idx] = ex + [pad_idx] * (minimal_length - len(ex))
            elif 0 < maximal_length < len(ex):
                batch[idx] = batch[idx][:maximal_length]
        return batch

    return pad_to_min_max_len


def tokenize_nn(tokenizer):
    """
    Tokenization of text by a given tokenizer for the needs of neural networks

    :param tokenizer: tokenizer
    :return: tokenized text
    """

    def tokenize_nn_text(text):
        """
        Tokenizes given text

        :param text: text to be tokenized
        :return: tokenized text
        """
        tokenized_all = [x for x in tokenizer.tokenize(text) if len(x) >= 1]
        return tokenized_all

    return tokenize_nn_text


def get_tokenizer(tokenizer_name, lang):
    if tokenizer_name == 'white_space' or tokenizer_name is None:
        tokenizer = WhitespaceTokenizer()
    elif tokenizer_name == 'toktok':
        tokenizer = ToktokTokenizer()
    elif tokenizer_name == 'word_tokenizer' or lang == FRENCH:
        tokenizer = WordTokenizerMock(lang)
    elif tokenizer_name == 'corpy':
        tokenizer = Tokenizer(lang)
    else:
        raise Exception('Not valid tokenizer: ' + tokenizer_name)

    return tokenizer


class WordTokenizerMock:
    def __init__(self, language):
        self.language = language

    def tokenize(self, text):
        words = [x for x in word_tokenize(text, language=self.language, preserve_line=True) if len(x) >= 1]
        return words


def dataset_get_x_y(my_df, x_col, y_col):
    x = my_df[x_col]
    y = my_df[y_col]

    return x, y


def get_sum_info(X, y, classes):
    ret = 'set has total ' + str(len(X)) + ' entries with \n'
    for i, clazz in enumerate(classes):
        tmp = '{0:.2f}% ({1:d}) - ' + clazz + '\n'
        class_len = len(X[y == clazz])
        tmp = tmp.format((class_len / (len(X) * 1.)) * 100, class_len)
        ret = ret + tmp
    ret = ret + '------------'

    return ret


