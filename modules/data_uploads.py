import numpy as np
from sklearn.model_selection import train_test_split
from sagemaker import TrainingInput

from config import data_dir, s3_data_loc, s3_root
from modules.utils import push_sm_csv


class DataUploader:

    def __init__(self, df, test_size=0.2, val_size=0.1,  stratify_on=None, id_col=None):
        self.df = df
        self.train, self.test, self.val = self._split_data(test_size, val_size, stratify_on, id_col)

    def _split_data(self, test_size, val_size, stratify_on, id_col):

        if stratify_on is not None:
            stratify_input = self.df[stratify_on]

        train, test = train_test_split(self.df, test_size=test_size, random_state=42, stratify=stratify_input)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        val_size = val_size / (1 - test_size)
        if stratify_on is not None:
            stratify_input = train[stratify_on]
        train, val = train_test_split(train, test_size=val_size, random_state=42, stratify=stratify_input)
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)

        if id_col is not None:
            train = train.drop(columns=id_col)
            val = val.drop(columns=id_col)
            test = test.drop(columns=id_col)

        test = test.drop(columns='y')

        return train, test, val

    def save_local(self):
        if self.train is None:
            self.df.to_csv(data_dir / 'model.csv', index=False, header=False)
        else:
            self.train.to_csv(data_dir / 'train.csv', index=False, header=False)
            self.test.to_csv(data_dir / 'test.csv', index=False, header=False)
            self.val.to_csv(data_dir / 'val.csv', index=False, header=False)

    def save_s3(self):
        if self.train is None:
            push_sm_csv(s3_data_loc / 'model.csv', self.df, index=False, header=False)
        else:
            push_sm_csv(s3_data_loc / 'train.csv', self.train, index=False, header=False)
            push_sm_csv(s3_data_loc / 'test.csv', self.test, index=False, header=False)
            push_sm_csv(s3_data_loc / 'val.csv', self.val, index=False, header=False)

    def s3_train_in(self):
        return TrainingInput("s3://{}/{}".format(s3_data_loc, "/train.csv"), content_type='csv')

    def s3_val_in(self):
        return TrainingInput("s3://{}/{}".format(s3_data_loc, "/val.csv"), content_type='csv')

