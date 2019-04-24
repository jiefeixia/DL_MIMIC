import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn


class PureCNN(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(CNN, self).__init__()
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

        out = self.dropout(out)
        out = F.relu(self.linear(out))
        
        out = self.fc(out)
#         out = out.LogSoftmax(1)
        return out


class NGramLanguageModeler(nn.Module):
    """
    for training embedding layer
    """

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((-1, self.context_size * self.embedding_dim))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(Embed_CNN, self).__init__()
        self.name = "Embed_CNN"

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.cnns = nn.ModuleList([nn.Sequential(nn.Conv1d(embedding_dim, 64, kernel),
                                                 nn.BatchNorm1d(64),
                                                 nn.ReLU())
                                   for kernel in range(3, 6)])

        self.linear = nn.Linear(3 * 64, 64)
        self.dropout = nn.Dropout(p=0.3)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out = [cnn(out) for cnn in self.cnns]
        out = [F.max_pool1d(i, i.shape[2]) for i in out]
        out = torch.cat(tuple(out), 1)[:, :, 0]
        out = self.dropout(out)
        out = self.linear(out)
        out = self.fc(out)
#         out = out.LogSoftmax(1)
        return out


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, layers, dropout, num_classes):
        super(LSTM, self).__init__()
        self.name = "LSTM"

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=layers,
                            bidirectional=True,
                            dropout=dropout,
                            batch_first=True)

        self.linear = nn.Linear(3 * 64, 64)
        self.dropout = nn.Dropout(p=0.3)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.embedding(x)

        out = rnn.pack_sequence(out)
        out, _ = self.lstm(out)
        out = rnn.pad_packed_sequence(out, batch_first=True)

        out = [F.max_pool1d(i, i.shape[2]) for i in out]
        out = torch.cat(tuple(out), 1)[:, :, 0]

        out = F.relu(self.linear(out))
        out = self.dropout(out)
        out = self.fc(out)
#         out = out.LogSoftmax(1)
        return out
