import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
# from util import *


class Embedding(nn.Module):
    """
    input: (batch_size, seq_length)
    output: (batch_size, seq_length, embedding_dim)
    """
    def __init__(self, num_embeddings, embedding_dim, init_embedding=None, trainable=False):
        # init_embedding: 2d matrix of pre-trained word embedding
        super(Embedding, self).__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))

        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2
        self.scale_grad_by_freq = False
        self.sparse = False

        self.reset_parameters(init_embedding)
        self.weight.requires_grad = trainable

    def reset_parameters(self, init_embedding=None):
        if not (init_embedding is None):
            self.weight.data.copy_(torch.from_numpy(init_embedding))
        else:
            self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def forward(self, input):
        padding_idx = self.padding_idx
        if padding_idx is None:
            padding_idx = -1
        return self._backend.Embedding.apply(
            input, self.weight,
            padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse
        )

    def __repr__(self):
        return nn.Embedding.__repr__(self)


class UtteranceEncoder(nn.Module):
    """
    input: (batch_size, seq_len)
    output: (batch_size, hidden_size * direction)
    """
    def __init__(self, init_embedding, hidden_size, rnn_mode='BiLSTM'):
        super(UtteranceEncoder, self).__init__()
        self.input_size = init_embedding.shape[1]
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.embedding = Embedding(init_embedding.shape[0], init_embedding.shape[1], init_embedding)
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                           bidirectional=True, batch_first=True)

    def forward(self, input):
        embedding = self.embedding(input)
        output, _ = self.rnn(embedding, self.init_hidden(input.size()[0]))
        return output[:, -1, :]

    def init_hidden(self, batch):
        h = Variable(torch.zeros(self.num_layers * 2, batch, self.hidden_size))
        c = Variable(torch.zeros(self.num_layers * 2, batch, self.hidden_size))
        return h, c


class ContextEncoder(nn.Module):
    """
    input: (batch_size, input_size)
    output: (batch_size, hidden_size)
    """
    def __init__(self, input_size, hidden_size, batch_size):
        super(ContextEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_size = batch_size
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.hidden = self.init_hidden()

    def forward(self, input):
        output, hn = self.rnn(input.view(input.size()[0], 1, self.input_size), self.hidden)
        self.hidden = hn
        return output.view(-1, self.hidden_size)

    def init_hidden(self):
        return Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self):
        pass

    def init_hidden(self):
        pass
