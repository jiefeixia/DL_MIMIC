import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import csv
import pandas as pd
import numpy as np
from loader import *

VOCAB_SIZE = IdxData.get_vacab_size()


def load_pretrained_embedding(word2vec_path, word2idx_path="word2idx.txt"):
    """load pre trained word embedding vector"""
    print("loading pretrained embedding weight...")
    pre_trained = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE)
    word2idx = dict()
    with open(word2idx_path) as f:
        for l in f:
            word, idx = l.strip().split(":")
            word2idx[word] = int(idx)

    embed_size = pre_trained.shape[1] - 1
    weight = np.zeros((VOCAB_SIZE, embed_size))
    pre_trained["idx"] = pre_trained.iloc[:, 0].map(word2idx)
    pre_trained["idx"] = pre_trained["idx"].fillna(-1).astype("int")
    pre_trained = pre_trained[pre_trained["idx"] > 0]
    weight[pre_trained["idx"]] = pre_trained.iloc[:, 1:-1].values
    return torch.from_numpy(weight)


class PureCNN(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(PureCNN, self).__init__()
        self.name = "CNN"

        self.cnns = nn.ModuleList([nn.Sequential(nn.Conv1d(embedding_dim, 64, kernel),
                                                 nn.BatchNorm1d(64),
                                                 nn.ReLU())
                                   for kernel in range(3, 6)])

        self.linear = nn.Linear(3 * 64, 64)
        self.dropout = nn.Dropout(p=0.3)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = [cnn(x) for cnn in self.cnns]
        out = [F.max_pool1d(i, i.shape[2]) for i in out]
        out = torch.cat(tuple(out), 1)[:, :, 0]

        out = F.relu(self.linear(out))
        out = self.dropout(out)
        out = self.fc(out)
        return out


class NGramLanguageModeler(nn.Module):
    """
    for training embedding layer
    """

    def __init__(self, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.embeddings = nn.Embedding(VOCAB_SIZE, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, VOCAB_SIZE)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((-1, self.context_size * self.embedding_dim))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


class CNN(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(CNN, self).__init__()
        self.name = "Embed_CNN"

        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim)
        self.cnns = nn.ModuleList([nn.Sequential(nn.Conv1d(embedding_dim, 64, kernel),
                                                 nn.BatchNorm1d(64),
                                                 nn.ReLU())
                                   for kernel in range(3, 6)])

        self.linear = nn.Linear(3 * 64, 64)
        self.dropout = nn.Dropout(p=0.3)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, seq_len):
        out = self.embedding(x)  # (B, padded seq Length, Embedding)
        out = out.permute(0, 2, 1)  # (B, E, L)
        out = [cnn(out) for cnn in self.cnns]
        out = [F.max_pool1d(i, i.shape[2]) for i in out]
        out = torch.cat(tuple(out), 1)[:, :, 0]

        out = F.relu(self.linear(out))
        out = self.dropout(out)
        out = self.fc(out)
        return out


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, layers, dropout, num_classes):
        super(LSTM, self).__init__()
        self.name = "LSTM"

        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=layers,
                            bidirectional=True,
                            dropout=dropout,
                            batch_first=True)

        self.linear = nn.Linear(hidden_size * 2, 64)
        self.dropout = nn.Dropout(p=0.2)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, seq_len):
        out = self.embedding(x)  # (B, padded seq Length, Embedding)

        out = rnn.pack_padded_sequence(out, seq_len, batch_first=True)
        out, _ = self.lstm(out)
        out, seq_len = rnn.pad_packed_sequence(out, batch_first=True)  # (B, padded seq Length, Hidden)

        out = out.permute(0, 2, 1)  # (B, Hidden, padded seq Length)
        out = F.max_pool1d(out, out.shape[2])  # (B, Hidden, 1)
        out = out.resize(out.shape[0], out.shape[1])  # (B, Hidden)

        out = F.relu(self.linear(out))
        out = self.dropout(out)
        out = self.fc(out)
        return out


class HAN(nn.Module):
    def __init__(self, word2vec_path, hidden_size, attention_size, num_classes):
        super(HAN, self).__init__()

        # load pre-trained embedding layer
        dict = self.load_pretrained_embedding(word2vec_path)
        self.embedding = nn.Embedding.from_pretrained(dict)
        embed_dim = self.embedding.weight.shape[1]
        self.word_att = AttGRU(input_size=embed_dim, hidden_size=hidden_size, att_size=attention_size)
        self.sent_att = AttGRU(input_size=hidden_size, hidden_size=hidden_size, att_size=attention_size)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, docs):
        """
        N stands for batch size,
        S stands for num of sentences per document, it is padded per batch
        W stands for num of words per sentence, it is padded per document
        :param: docs: list(padded S, ), inside is sent: tensor(N, W)
        :return:  out, word_att_weights (N, S, W), sent_att_weight (N, S)
        """
        docs = x.permute(1, 0, 2)  # x (S, N, W)

        sent_vecs = []
        word_att_weights = []
        for i, sent in enumerate(x):  # sent (N, W)
            word_vec = self.embedding(sent)  # sent (N, L, D)
            sent_vec, word_att_weight = self.word_att(word_vec)  # sent_vec (N, H), word_att_weight (N, W)
            sent_vecs.append(sent_vec)
            word_att_weights.append(word_att_weight)

        word_att_weights = torch.stack(word_att_weights).premute(1, 0, 2)  # word_att_weight (N, S, L)
        sent_vecs = torch.stack(sent_vecs).premute(1, 0, 2)  # sent_vecs (N, S, H)

        doc_vec, sent_att_weight = self.sent_att(sent_vecs)  # doc_vec (N, H)
        out = self.fc(doc_vec)
        return out, word_att_weights, sent_att_weight


class AttGRU(nn.Module):
    def __init__(self, input_size, hidden_size, att_size):
        super(AttGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.key_fc = nn.Linear(hidden_size * 2, att_size)

        # query is a fixed vector storing the information "which is the important information in the value"
        self.query = nn.Parameter(torch.empty(att_size))  # query (A)
        torch.nn.init.xavier_normal_(self.query)

    def forward(self, x):
        h, h_n = self.gru(x)  # both key and value of attention are h

        batch_size, padded_len, hidden_size = h.shape
        key = h.view(batch_size * padded_len, hidden_size)  # key (N*L, H)
        key = F.tanh(self.key_fc(key))  # key (N*L, A)
        key = key.view(batch_size, padded_len, -1)  # h (N, L, A)

        query = self.query.repeat(batch_size, 1).unsqueeze(2)  # query (N, A, 1)
        energy = torch.bmm(key, query)  # energy (N, L, 1)
        energy = energy.squeeze(2)  # energy (N, L, 1)
        att_weight = F.softmax(energy, dim=1)  # att_weight (N, L)
        # for simplicity, ignore the mask for different length

        context = torch.bmm(att_weight, h)  # (N, H)

        return context, att_weight


if __name__ == '__main__':
    # for debug

    x = torch.randint(0, 3000, (200, 1000)).cuda()
    net = HAN("glove.6B.50d.txt", 512, 128, 8)
    net = net.cuda()

    out, word_aw, sent_aw = net(x)
    print(out)
