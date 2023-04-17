import pandas as pd
import logging
import os
import math
from sklearn.model_selection import train_test_split

from config import FACEBOOK_DATASET_TRAIN, FACEBOOK_DATASET_TEST, FACEBOOK_DATASET_DEV, FACEBOOK_DATASET, \
    CSFD_DATASET_TRAIN, CSFD_DATASET_TEST, FACEBOOK_DATASET_DIR, CSFD_DATASET_DEV, CSFD_DATASET, CSFD_DATASET_DIR, \
    MALL_DATASET_TRAIN, MALL_DATASET_TEST, MALL_DATASET_DEV, MALL_DATASET, MALL_DATASET_DIR, CZECH, \
    CROSS_LINGUAL_MODE, \
    MODES, MONOLINGUAL_MODE, IMDB_DATASET_DIR, IMDB_DATASET_TEST, IMDB_DATASET_TRAIN, IMDB_DATASET, ENGLISH, \
    SST_DATASET_DIR, SST_DATASET_TEST, SST_DATASET_DEV, SST_DATASET, SST_DATASET_TRAIN, IMDB_DATASET_CL_DIR, \
    IMDB_DATASET_CL_TRAIN, IMDB_DATASET_CL_TEST, IMDB_DATASET_CL, IMDB_DATASET_DEV, RANDOM_SEED, ALLOCINE_DATASET, \
    ALLOCINE_DATASET_DEV, ALLOCINE_DATASET_TEST, ALLOCINE_DATASET_TRAIN, FRENCH, ALLOCINE_DATASET_DIR

logger = logging.getLogger(__name__)


class DatasetLoader(object):
    def __init__(self, max_train_data, binary, lang_mode, source_lang, target_lang=None):
        self.binary = binary
        self.max_train_data = max_train_data
        # Cross lingual or monolingual
        # if monolingual only the source_lang should be set
        # if cross lingual both the source_lang and target_lang should be set
        if lang_mode not in MODES:
            raise ValueError("Unknown mode: " + str(lang_mode))
        else:
            self.lang_mode = lang_mode

        # Source language of the dataset
        self.source_lang = source_lang

        # Target language of the dataset only, if lang_mode is cross_lingual
        self.target_lang = target_lang

        self.train_data = None
        self.test_data = None
        self.dev_data = None

    def get_lang_mode(self):
        return self.lang_mode

    def get_source_lang(self):
        return self.source_lang

    def get_target_lang(self):
        return self.target_lang

    def get_dev_data(self):
        if self.dev_data is None:
            self.load_data()
        return self.dev_data

    def get_test_data(self):
        if self.test_data is None:
            self.load_data()
        return self.test_data

    def get_cut_train_data(self, data_df):

        data_to_return = data_df
        total_size = len(data_df)
        logger.info("The size of training dataset is: " + str(total_size))
        logger.info("Applying cutting to train data with value: " + str(self.max_train_data))

        new_size = -1
        if self.max_train_data <= 0:
            logger.info("No cutting is performed")
        elif 0 < self.max_train_data <= 1:
            logger.info("Cutting in percentages")
            new_size = total_size * self.max_train_data
            new_size = math.ceil(new_size)
        elif self.max_train_data > 1:
            logger.info("Cutting in absolute numbers")
            new_size = self.max_train_data
            new_size = math.ceil(new_size)
        else:
            raise Exception("Unknown value for max_train_data, the value: " + str(self.max_train_data))

        logger.info("New size is: " + str(new_size))
        if new_size > 1:
            data_to_return = data_to_return.head(new_size)
            data_to_return.reset_index(drop=True, inplace=True)

        return data_to_return

    def get_train_data(self):
        if self.train_data is None:
            self.load_data()

        data_to_return = self.get_cut_train_data(self.train_data)

        return data_to_return

    def get_train_dev_data(self):
        if self.train_data is None:
            self.load_data()
        if self.dev_data is None:
            self.load_data()

        df = pd.concat([self.train_data, self.dev_data], axis=0)
        df.reset_index(drop=True, inplace=True)

        data_to_return = self.get_cut_train_data(df)

        return data_to_return

    def load_entire_dataset(self):
        raise NotImplementedError()

    def load_data(self):
        """Loads all data"""
        raise NotImplementedError()

    def get_dataset_dir(self):
        """Returns dir of the dataset"""
        raise NotImplementedError()

    def get_class_num(self):
        """Returns number of classes"""
        if self.binary is True:
            return 2
        else:
            return 3

    def get_classes(self):
        """Returns possible classes as numbers"""
        if self.binary is True:
            return [0, 1]
        else:
            return [0, 1, 2]

    def get_class_names(self):
        """
        Returns possible names of classes
        Indices corresponds to the ones returned by method get_classes
        In default it returns three classes, can be overridden
        """
        if self.binary is True:
            return ['negative', 'positive']
        else:
            return ['negative', 'positive', 'neutral']

    @staticmethod
    def get_text4label(label):
        """
        Returns text for numerical label
        :param label: numerical label
        :return:  textual label 0 - negative, 1 - positive, 2 - neutral
        """
        ret = ''
        if label == 0:
            ret = 'negative'
        elif label == 1:
            ret = 'positive'
        elif label == 2:
            ret = 'neutral'
        else:
            raise Exception("Unknown label: " + str(label))

        return ret

    def get_label4text(self, text_label):
        """
        Returns numerical label for text label
        :param text_label: text label
        :return: negative - 0, positive - 1, neutral - 2
        """
        ret = ''
        if text_label == 'negative':
            ret = 0
        elif text_label == 'positive':
            ret = 1
        elif text_label == 'neutral':
            ret = 2
        else:
            raise Exception("Unknown text label: " + str(text_label))

        return ret

    def filter_neutral(self, df):
        if self.binary:
            df = df[df.label != 2]
            df.reset_index(drop=True, inplace=True)
        return df


# Implementations

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$ Monolingual $$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

class CzechFBDatasetLoader(DatasetLoader):
    def __init__(self, max_train_data, binary=False):
        super().__init__(max_train_data, binary, MONOLINGUAL_MODE, CZECH)

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(FACEBOOK_DATASET_TRAIN))
        self.train_data = self.filter_neutral(self.train_data)

        self.test_data = pd.read_csv(os.path.join(FACEBOOK_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data = pd.read_csv(os.path.join(FACEBOOK_DATASET_DEV))
        self.dev_data = self.filter_neutral(self.dev_data)

    def load_entire_dataset(self):
        df = pd.read_csv(os.path.join(FACEBOOK_DATASET))
        df = self.filter_neutral(df)
        return df

    def get_dataset_dir(self):
        return FACEBOOK_DATASET_DIR


class CzechCSFDDatasetLoader(DatasetLoader):
    def __init__(self, max_train_data, binary=False):
        super().__init__(max_train_data, binary, MONOLINGUAL_MODE, CZECH)

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(CSFD_DATASET_TRAIN))
        self.train_data = self.filter_neutral(self.train_data)

        self.test_data = pd.read_csv(os.path.join(CSFD_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data = pd.read_csv(os.path.join(CSFD_DATASET_DEV))
        self.dev_data = self.filter_neutral(self.dev_data)

    def load_entire_dataset(self):
        df = pd.read_csv(os.path.join(CSFD_DATASET))
        df = self.filter_neutral(df)
        return df

    def get_dataset_dir(self):
        return CSFD_DATASET_DIR


class CzechMALLCZDatasetLoader(DatasetLoader):
    def __init__(self, max_train_data, binary=False):
        super().__init__(max_train_data, binary, MONOLINGUAL_MODE, CZECH)

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(MALL_DATASET_TRAIN))
        self.train_data = self.filter_neutral(self.train_data)

        self.test_data = pd.read_csv(os.path.join(MALL_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data = pd.read_csv(os.path.join(MALL_DATASET_DEV))
        self.dev_data = self.filter_neutral(self.dev_data)

    def load_entire_dataset(self):
        df = pd.read_csv(os.path.join(MALL_DATASET))
        df = self.filter_neutral(df)
        return df

    def get_dataset_dir(self):
        return MALL_DATASET_DIR


class EnglishIMDBDatasetLoader(DatasetLoader):
    def __init__(self, max_train_data, binary=True):
        if not binary:
            raise Exception("Imdb dataset is a only binary dataset")
        super().__init__(max_train_data, binary, MONOLINGUAL_MODE, ENGLISH)

    def load_data(self):
        self.test_data = pd.read_csv(os.path.join(IMDB_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        tmp_train_data = pd.read_csv(os.path.join(IMDB_DATASET_TRAIN))
        tmp_train_data = self.filter_neutral(tmp_train_data)

        # The original dev data are empty so we take part from the train data
        # self.dev_data = pd.read_csv(os.path.join(IMDB_DATASET_DEV))
        # self.dev_data = self.filter_neutral(self.dev_data)
        dev_size = 0.1
        self.train_data, self.dev_data = train_test_split(tmp_train_data, test_size=dev_size, random_state=RANDOM_SEED)
        self.train_data.reset_index(drop=True, inplace=True)
        self.dev_data.reset_index(drop=True, inplace=True)

    def load_entire_dataset(self):
        df = pd.read_csv(os.path.join(IMDB_DATASET))
        df = self.filter_neutral(df)
        return df

    def get_dataset_dir(self):
        return IMDB_DATASET_DIR


class EnglishSSTDatasetLoader(DatasetLoader):
    def __init__(self, max_train_data, binary=False):
        super().__init__(max_train_data, binary, MONOLINGUAL_MODE, ENGLISH)

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(SST_DATASET_TRAIN))
        self.train_data = self.filter_neutral(self.train_data)

        self.test_data = pd.read_csv(os.path.join(SST_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data = pd.read_csv(os.path.join(SST_DATASET_DEV))
        self.dev_data = self.filter_neutral(self.dev_data)

    def load_entire_dataset(self):
        df = pd.read_csv(os.path.join(SST_DATASET))
        df = self.filter_neutral(df)
        return df

    def get_dataset_dir(self):
        return SST_DATASET_DIR


class FrenchAllocineDatasetLoader(DatasetLoader):
    def __init__(self, max_train_data, binary=False):
        super().__init__(max_train_data, binary, MONOLINGUAL_MODE, ENGLISH)

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(ALLOCINE_DATASET_TRAIN))
        self.train_data = self.filter_neutral(self.train_data)

        self.test_data = pd.read_csv(os.path.join(ALLOCINE_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data = pd.read_csv(os.path.join(ALLOCINE_DATASET_DEV))
        self.dev_data = self.filter_neutral(self.dev_data)

    def load_entire_dataset(self):
        df = pd.read_csv(os.path.join(ALLOCINE_DATASET))
        df = self.filter_neutral(df)
        return df

    def get_dataset_dir(self):
        return ALLOCINE_DATASET_DIR


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$ Cross-lingual $$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

class CrossLingualDataset(DatasetLoader):
    """
    As dev data the subclass dataset MUST return the source language data
    As test data the subclass dataset MUST return the target language data
    """

    def __init__(self, max_train_data, binary, source_lang, target_lang):
        super().__init__(max_train_data, binary, CROSS_LINGUAL_MODE, source_lang, target_lang)

    def get_source_lang_dev_data(self):
        """Returns the dev data for the source language"""
        self.get_dev_data()

    def get_target_lang_dev_data(self):
        """Returns the dev data for the target language"""
        raise NotImplementedError()

    def load_entire_source_lang_dataset(self):
        raise NotImplementedError()

    def load_entire_target_lang_dataset(self):
        raise NotImplementedError()

    def load_entire_dataset(self):
        df_source = self.load_entire_source_lang_dataset()
        df_target = self.load_entire_target_lang_dataset()

        df = pd.concat([df_source, df_target], axis=0)
        df.reset_index(drop=True, inplace=True)
        return df


class CrossLingualCSFDIMDBDataset(CrossLingualDataset):
    """
    It returns CSFD Czech train and test dataset as train data
    As dev data it returns CSFD Czech dev data
    As test data it returns IMDB original test data
    As special dev data it returns IMDB original train data

    Source = CSFD, Czech
    Target = IMDB, English
    """

    def __init__(self, max_train_data, binary=True):
        if not binary:
            raise Exception("Imdb dataset is a only binary dataset")
        super().__init__(max_train_data, binary, CZECH, ENGLISH)

    def get_dataset_dir(self):
        return IMDB_DATASET_DIR

    def load_data(self):
        tmp_train_data_csfd = pd.read_csv(os.path.join(CSFD_DATASET_TRAIN))
        tmp_train_data_csfd = self.filter_neutral(tmp_train_data_csfd)

        tmp_test_data_csfd = pd.read_csv(os.path.join(CSFD_DATASET_TEST))
        tmp_test_data_csfd = self.filter_neutral(tmp_test_data_csfd)

        self.train_data = pd.concat([tmp_train_data_csfd, tmp_test_data_csfd], axis=0)
        self.train_data.reset_index(drop=True, inplace=True)

        self.dev_data = pd.read_csv(os.path.join(CSFD_DATASET_DEV))
        self.dev_data = self.filter_neutral(self.dev_data)

        self.test_data = pd.read_csv(os.path.join(IMDB_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data_target = pd.read_csv(os.path.join(IMDB_DATASET_TRAIN))
        self.dev_data_target = self.filter_neutral(self.dev_data_target)

    def get_target_lang_dev_data(self):
        if self.dev_data_target is None:
            self.load_data()
        return self.dev_data_target

    def load_entire_source_lang_dataset(self):
        df = pd.read_csv(os.path.join(CSFD_DATASET))
        df = self.filter_neutral(df)
        return df

    def load_entire_target_lang_dataset(self):
        df = pd.read_csv(os.path.join(IMDB_DATASET))
        df = self.filter_neutral(df)
        return df


class CrossLingualCSFDSSTDataset(CrossLingualDataset):
    """
    It returns CSFD Czech train and test dataset as train data
    As dev data it returns CSFD Czech dev data
    As test data it returns original SST test data
    As special dev data it returns SST original dev data

    Source = CSFD, Czech
    Target = SST, English

    """

    def __init__(self, max_train_data, binary):
        super().__init__(max_train_data, binary, CZECH, ENGLISH)

    def get_dataset_dir(self):
        return SST_DATASET_DIR

    def load_data(self):
        # train data
        tmp_train_data_csfd = pd.read_csv(CSFD_DATASET_TRAIN)
        tmp_train_data_csfd = self.filter_neutral(tmp_train_data_csfd)

        tmp_test_data_csfd = pd.read_csv(CSFD_DATASET_TEST)
        tmp_test_data_csfd = self.filter_neutral(tmp_test_data_csfd)

        self.train_data = pd.concat([tmp_train_data_csfd, tmp_test_data_csfd], axis=0)
        self.train_data.reset_index(drop=True, inplace=True)

        # dev data
        self.dev_data = pd.read_csv(CSFD_DATASET_DEV)
        self.dev_data = self.filter_neutral(self.dev_data)

        # test data
        self.test_data = pd.read_csv(SST_DATASET_TEST)
        self.test_data = self.filter_neutral(self.test_data)

        # special english data
        self.dev_data_target = pd.read_csv(SST_DATASET_DEV)
        self.dev_data_target = self.filter_neutral(self.dev_data_target)

    def get_target_lang_dev_data(self):
        if self.dev_data_target is None:
            self.load_data()
        return self.dev_data_target

    def load_entire_source_lang_dataset(self):
        df = pd.read_csv(os.path.join(CSFD_DATASET))
        df = self.filter_neutral(df)
        return df

    def load_entire_target_lang_dataset(self):
        df = pd.read_csv(SST_DATASET)
        df = self.filter_neutral(df)
        return df


class CrossLingualSSTCSFDDataset(CrossLingualDataset):
    """
    This dataset

    It returns SST English train + dev data as train data
    As dev data it returns SST English test data
    As test data it returns CSFD Czech test data
    There is additional function that returns czech CSFD Dev Data

    Source = SST, English
    Target = CSFD, Czech
    """

    def __init__(self, max_train_data, binary=False):
        super().__init__(max_train_data, binary, ENGLISH, CZECH)

    def get_dataset_dir(self):
        return SST_DATASET_DIR

    def load_data(self):
        # train data
        tmp_train_data_sst = pd.read_csv(SST_DATASET_TRAIN)
        tmp_train_data_sst = self.filter_neutral(tmp_train_data_sst)

        tmp_dev_data_sst = pd.read_csv(SST_DATASET_DEV)
        tmp_dev_data_sst = self.filter_neutral(tmp_dev_data_sst)

        self.train_data = pd.concat([tmp_train_data_sst, tmp_dev_data_sst], axis=0)
        self.train_data.reset_index(drop=True, inplace=True)

        # dev data - these are the SST original test data
        self.dev_data = pd.read_csv(SST_DATASET_TEST)
        self.dev_data = self.filter_neutral(self.dev_data)

        # test data
        self.test_data = pd.read_csv(CSFD_DATASET_TEST)
        self.test_data = self.filter_neutral(self.test_data)

        # special czech dev data
        self.dev_data_target = pd.read_csv(CSFD_DATASET_DEV)
        self.dev_data_target = self.filter_neutral(self.dev_data_target)

    def get_target_lang_dev_data(self):
        """Returns the dev data for the target language"""
        if self.dev_data_target is None:
            self.load_data()
        return self.dev_data_target

    def load_entire_source_lang_dataset(self):
        df = pd.read_csv(SST_DATASET)
        df = self.filter_neutral(df)
        return df

    def load_entire_target_lang_dataset(self):
        df = pd.read_csv(os.path.join(CSFD_DATASET))
        df = self.filter_neutral(df)
        return df


class CrossLingualIMDBCSFDDataset(CrossLingualDataset):
    """
    This dataset use different split than the original one

    It returns IMDB English data as train data
    As dev data it returns IMDB English test data (these are different from the original one)
    As test data it returns CSFD Czech test data
    There is additional function that returns czech CSFD Dev Data

    The entire dataset is loaded as IMDB English dataset

    Source = IMDB, English
    Target = CSFD, Czech
    """

    def __init__(self, max_train_data, binary=True):
        if binary == False:
            raise Exception("Imdb dataset is only binary dataset")
        super().__init__(max_train_data, binary, ENGLISH, CZECH)

    def get_dataset_dir(self):
        return IMDB_DATASET_CL_DIR

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(IMDB_DATASET_CL_TRAIN))
        self.train_data = self.filter_neutral(self.train_data)

        # it is correct we return test data for english as dev here
        self.dev_data = pd.read_csv(os.path.join(IMDB_DATASET_CL_TEST))
        self.dev_data = self.filter_neutral(self.dev_data)

        self.test_data = pd.read_csv(os.path.join(CSFD_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data_target = pd.read_csv(os.path.join(CSFD_DATASET_DEV))
        self.dev_data_target = self.filter_neutral(self.dev_data_target)

    def get_target_lang_dev_data(self):
        if self.dev_data_target is None:
            self.load_data()
        return self.dev_data_target

    def load_entire_source_lang_dataset(self):
        df = pd.read_csv(os.path.join(IMDB_DATASET_CL))
        df = self.filter_neutral(df)
        return df

    def load_entire_target_lang_dataset(self):
        df = pd.read_csv(os.path.join(CSFD_DATASET))
        df = self.filter_neutral(df)
        return df


class CrossLingualAllocineCSFDDataset(CrossLingualDataset):
    """
    This dataset use different split than the original one

    It returns Allocine French data as train data
    As dev data it returns Allocine French test data (these are different from the original one)
    As test data it returns CSFD Czech test data
    There is additional function that returns czech CSFD Dev Data

    The entire dataset is loaded as Allocine French dataset

    Source = Allocine, French
    Target = CSFD, Czech
    """

    def __init__(self, max_train_data, binary=True):
        if binary == False:
            raise Exception("Allocine dataset is only binary dataset")
        super().__init__(max_train_data, binary, FRENCH, CZECH)

    def get_dataset_dir(self):
        return IMDB_DATASET_CL_DIR

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(ALLOCINE_DATASET_TRAIN))
        self.train_data = self.filter_neutral(self.train_data)

        # it is correct we return test data for english as dev here
        self.dev_data = pd.read_csv(os.path.join(ALLOCINE_DATASET_TEST))
        self.dev_data = self.filter_neutral(self.dev_data)

        self.test_data = pd.read_csv(os.path.join(CSFD_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data_target = pd.read_csv(os.path.join(CSFD_DATASET_DEV))
        self.dev_data_target = self.filter_neutral(self.dev_data_target)

    def get_target_lang_dev_data(self):
        if self.dev_data_target is None:
            self.load_data()
        return self.dev_data_target

    def load_entire_source_lang_dataset(self):
        df = pd.read_csv(os.path.join(ALLOCINE_DATASET))
        df = self.filter_neutral(df)
        return df

    def load_entire_target_lang_dataset(self):
        df = pd.read_csv(os.path.join(CSFD_DATASET))
        df = self.filter_neutral(df)
        return df


class CrossLingualAllocineSSTDataset(CrossLingualDataset):
    """
    This dataset use different split than the original one

    It returns Allocine French data as train data
    As dev data it returns Allocine French test data (these are different from the original one)
    As test data it returns SST English test data
    There is additional function that returns czech SST English Data

    The entire dataset is loaded as Allocine French dataset

    Source = Allocine, French
    Target = SST, English
    """

    def __init__(self, max_train_data, binary=True):
        if binary == False:
            raise Exception("Allocine dataset is only binary dataset")
        super().__init__(max_train_data, binary, FRENCH, ENGLISH)

    def get_dataset_dir(self):
        return IMDB_DATASET_CL_DIR

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(ALLOCINE_DATASET_TRAIN))
        self.train_data = self.filter_neutral(self.train_data)

        # it is correct we return test data for english as dev here
        self.dev_data = pd.read_csv(os.path.join(ALLOCINE_DATASET_TEST))
        self.dev_data = self.filter_neutral(self.dev_data)

        self.test_data = pd.read_csv(os.path.join(SST_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data_target = pd.read_csv(os.path.join(SST_DATASET_DEV))
        self.dev_data_target = self.filter_neutral(self.dev_data_target)

    def get_target_lang_dev_data(self):
        if self.dev_data_target is None:
            self.load_data()
        return self.dev_data_target

    def load_entire_source_lang_dataset(self):
        df = pd.read_csv(os.path.join(ALLOCINE_DATASET))
        df = self.filter_neutral(df)
        return df

    def load_entire_target_lang_dataset(self):
        df = pd.read_csv(os.path.join(SST_DATASET))
        df = self.filter_neutral(df)
        return df


class CrossLingualAllocineIMDBDataset(CrossLingualDataset):
    """
    This dataset use different split than the original one

    It returns Allocine French data as train data
    As dev data it returns Allocine French test data (these are different from the original one)
    As test data it returns IMDB English test data
    There is additional function that returns czech IMDB English Data

    The entire dataset is loaded as Allocine French dataset

    Source = Allocine, French
    Target = IMDB, English
    """

    def __init__(self, max_train_data, binary=True):
        if binary == False:
            raise Exception("Imdb dataset is only binary dataset")
        super().__init__(max_train_data, binary, FRENCH, ENGLISH)

    def get_dataset_dir(self):
        return IMDB_DATASET_CL_DIR

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(ALLOCINE_DATASET_TRAIN))
        self.train_data = self.filter_neutral(self.train_data)

        # it is correct we return test data for english as dev here
        self.dev_data = pd.read_csv(os.path.join(ALLOCINE_DATASET_TEST))
        self.dev_data = self.filter_neutral(self.dev_data)

        self.test_data = pd.read_csv(os.path.join(IMDB_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data_target = pd.read_csv(os.path.join(IMDB_DATASET_TRAIN))
        self.dev_data_target = self.filter_neutral(self.dev_data_target)

    def get_target_lang_dev_data(self):
        if self.dev_data_target is None:
            self.load_data()
        return self.dev_data_target

    def load_entire_source_lang_dataset(self):
        df = pd.read_csv(os.path.join(ALLOCINE_DATASET))
        df = self.filter_neutral(df)
        return df

    def load_entire_target_lang_dataset(self):
        df = pd.read_csv(os.path.join(IMDB_DATASET))
        df = self.filter_neutral(df)
        return df


class CrossLingualCSFDAllocineDataset(CrossLingualDataset):
    """
    It returns CSFD Czech train and test dataset as train data
    As dev data it returns CSFD Czech dev data
    As test data it returns original Allocine test data
    As special dev data it returns Allocine original dev data

    Source = CSFD, Czech
    Target = Allocine, French

    """

    def __init__(self, max_train_data, binary=True):
        super().__init__(max_train_data, binary, CZECH, FRENCH)

    def get_dataset_dir(self):
        return SST_DATASET_DIR

    def load_data(self):
        # train data
        tmp_train_data_csfd = pd.read_csv(CSFD_DATASET_TRAIN)
        tmp_train_data_csfd = self.filter_neutral(tmp_train_data_csfd)

        tmp_test_data_csfd = pd.read_csv(CSFD_DATASET_TEST)
        tmp_test_data_csfd = self.filter_neutral(tmp_test_data_csfd)

        self.train_data = pd.concat([tmp_train_data_csfd, tmp_test_data_csfd], axis=0)
        self.train_data.reset_index(drop=True, inplace=True)

        # dev data
        self.dev_data = pd.read_csv(CSFD_DATASET_DEV)
        self.dev_data = self.filter_neutral(self.dev_data)

        # test data
        self.test_data = pd.read_csv(ALLOCINE_DATASET_TEST)
        self.test_data = self.filter_neutral(self.test_data)

        # special english data
        self.dev_data_target = pd.read_csv(ALLOCINE_DATASET_DEV)
        self.dev_data_target = self.filter_neutral(self.dev_data_target)

    def get_target_lang_dev_data(self):
        if self.dev_data_target is None:
            self.load_data()
        return self.dev_data_target

    def load_entire_source_lang_dataset(self):
        df = pd.read_csv(os.path.join(CSFD_DATASET))
        df = self.filter_neutral(df)
        return df

    def load_entire_target_lang_dataset(self):
        df = pd.read_csv(ALLOCINE_DATASET)
        df = self.filter_neutral(df)
        return df


class CrossLingualIMDBAllocineDataset(CrossLingualDataset):
    """
    This dataset use different split than the original one

    It returns IMDB English data as train data
    As dev data it returns IMDB English test data (these are different from the original one)
    As test data it returns Allocine French test data
    There is additional function that returns french Allocine Dev Data

    The entire dataset is loaded as IMDB English dataset

    Source = IMDB, English
    Target = Allocine, French
    """

    def __init__(self, max_train_data, binary=True):
        if binary == False:
            raise Exception("Imdb dataset is only binary dataset")
        super().__init__(max_train_data, binary, ENGLISH, FRENCH)

    def get_dataset_dir(self):
        return IMDB_DATASET_CL_DIR

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(IMDB_DATASET_CL_TRAIN))
        self.train_data = self.filter_neutral(self.train_data)

        # it is correct we return test data for english as dev here
        self.dev_data = pd.read_csv(os.path.join(IMDB_DATASET_CL_TEST))
        self.dev_data = self.filter_neutral(self.dev_data)

        self.test_data = pd.read_csv(os.path.join(ALLOCINE_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data_target = pd.read_csv(os.path.join(ALLOCINE_DATASET_DEV))
        self.dev_data_target = self.filter_neutral(self.dev_data_target)

    def get_target_lang_dev_data(self):
        if self.dev_data_target is None:
            self.load_data()
        return self.dev_data_target

    def load_entire_source_lang_dataset(self):
        df = pd.read_csv(os.path.join(IMDB_DATASET_CL))
        df = self.filter_neutral(df)
        return df

    def load_entire_target_lang_dataset(self):
        df = pd.read_csv(os.path.join(ALLOCINE_DATASET))
        df = self.filter_neutral(df)
        return df


class CrossLingualSSTAllocineDataset(CrossLingualDataset):
    """
    This dataset

    It returns SST English train + dev data as train data
    As dev data it returns SST English test data
    As test data it returns Allocine French test data
    There is additional function that returns french Allocine Dev Data

    Source = SST, English
    Target = Allocine, French
    """

    def __init__(self, max_train_data, binary=True):
        super().__init__(max_train_data, binary, ENGLISH, FRENCH)

    def get_dataset_dir(self):
        return SST_DATASET_DIR

    def load_data(self):
        # train data
        tmp_train_data_sst = pd.read_csv(SST_DATASET_TRAIN)
        tmp_train_data_sst = self.filter_neutral(tmp_train_data_sst)

        tmp_dev_data_sst = pd.read_csv(SST_DATASET_DEV)
        tmp_dev_data_sst = self.filter_neutral(tmp_dev_data_sst)

        self.train_data = pd.concat([tmp_train_data_sst, tmp_dev_data_sst], axis=0)
        self.train_data.reset_index(drop=True, inplace=True)

        # dev data - these are the SST original test data
        self.dev_data = pd.read_csv(SST_DATASET_TEST)
        self.dev_data = self.filter_neutral(self.dev_data)

        # test data
        self.test_data = pd.read_csv(ALLOCINE_DATASET_TEST)
        self.test_data = self.filter_neutral(self.test_data)

        # special czech dev data
        self.dev_data_target = pd.read_csv(ALLOCINE_DATASET_DEV)
        self.dev_data_target = self.filter_neutral(self.dev_data_target)

    def get_target_lang_dev_data(self):
        """Returns the dev data for the target language"""
        if self.dev_data_target is None:
            self.load_data()
        return self.dev_data_target

    def load_entire_source_lang_dataset(self):
        df = pd.read_csv(SST_DATASET)
        df = self.filter_neutral(df)
        return df

    def load_entire_target_lang_dataset(self):
        df = pd.read_csv(os.path.join(ALLOCINE_DATASET))
        df = self.filter_neutral(df)
        return df
