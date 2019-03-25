import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
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

        out = F.relu(self.linear(out))
        out = self.dropout(out)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out


class NGramLanguageModeler(nn.Module):
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
