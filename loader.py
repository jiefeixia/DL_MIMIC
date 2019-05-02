import os

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn

import pandas as pd
import numpy as np

from tqdm import tqdm

EMBEDDING_DIM = 256


class IdxData(Dataset):
    """
    only used for traing embedding layer
    """

    def __init__(self, context_size, file):
        filepath = os.path.join(check_sys_path(), file)
        print("loading data from", filepath)
        notes = np.load(filepath)

        if context_size > 0:  # load discharge notes to train embedding layer
            self.data = []
            for note in notes:
                for i in range(len(note) - context_size):
                    self.data.append(note[i:i + context_size + 1])
            self.data = np.array(self.data)
        else:  # load admission notes to transform it from idx to embeddings
            self.data = notes

    @staticmethod
    def get_vacab_size():
        with open(os.path.join(check_sys_path(), "word_idx.txt")) as f:
            vocab_size = len(f.readlines())
        return vocab_size

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class Data(Dataset):
    """
    used to traing main prediction model
    """

    def __init__(self, dataset):
        self.name = dataset

        if dataset == "train":
            x_path = "train_idx.npy"
            y_path = "train_label.npy"
        elif dataset == "validation":
            x_path = "val_idx.npy"
            y_path = "val_label.npy"
        else:
            x_path = "test_idx.npy"
            y_path = "test_label.npy"

        X = np.load(os.path.join(check_sys_path(), x_path))
        y = np.load(os.path.join(check_sys_path(), y_path))
        self.proportion = np.sum(y, axis=0) / np.sum(y)

        self.X = []
        self.y = []
        for i, x in enumerate(X):
            if x.shape[0] > 0:
                self.X.append(x)
                self.y.append(y[i])

    @staticmethod
    def get_embedding_dim():
        return EMBEDDING_DIM

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


class EmbeddingData(Dataset):
    def __init__(self, dataset):
        if dataset == "train":
            x_path = "train_%dembedding.npy" % EMBEDDING_DIM
            y_path = "train_label.npy"
        else:
            x_path = "val_%dembedding.npy" % EMBEDDING_DIM
            y_path = "val_label.npy"

        print("padding ...")
        X = np.load(os.path.join(check_sys_path(), x_path))
        max_len = np.max([len(note) for note in X])
        self.X = np.zeros((X.shape[0], EMBEDDING_DIM, max_len))
        for i, note in tqdm(enumerate(X)):
            self.X[i, :, 0:len(note)] = np.moveaxis(note, 0, 1)
        self.y = np.load(os.path.join(check_sys_path(), y_path))

    @staticmethod
    def get_embedding_dim():
        return EMBEDDING_DIM

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

    def __len__(self):
        return self.X.shape[0]


def check_sys_path():
    """
    :return: absolute path of the folder to store data
    """
    cwd = os.getcwd()
    if "jeffy" in cwd.lower():  # local env
        return "C:\\Users\\Jeffy\\Downloads\\Data\\project"
    else:  # aws env
        return "data"


def get_metrics_df():
    df = pd.DataFrame({"p": [-1] * 8, "r": [-1] * 8, "f1": [-1] * 8})
    with open(os.path.join(check_sys_path(), "med_idx.txt")) as f:
        medicines = [line.split(":")[0] for line in f]
    df.index = medicines
    return df

word2idx = dict()
with open(os.path.join(check_sys_path(), "word2idx.txt")) as f:
    for line in f:
        word, idx = line.strip().split(":")
        word2idx[word] = int(idx)

if __name__ == '__main__':
    data = Data("validation")
    data.__getitem__(100)

    train_loader = DataLoader(data,
                              batch_size=128,
                              num_workers=4,
                              shuffle=True,
                              collate_fn=collate)

    for batch_idx, (x, seq_len, y) in enumerate(train_loader):
        print(x)
        print(y)
