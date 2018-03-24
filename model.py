import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
# from util import *


class Embedding(nn.Module):
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
    def __init__(self, init_embedding, hidden_size, rnn_mode='BiLSTM'):
        super(UtteranceEncoder, self).__init__()
        self.input_size = init_embedding.shape[1]
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.embedding = Embedding(init_embedding.shape[0], init_embedding.shape[1], init_embedding)
        # if rnn_mode == 'BiLSTM':
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=True)
        # elif rnn_mode == 'GRU':
        #     self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers)

    def forward(self, input):
        embedding = self.embedding(input)#.view(-1, 1, self.input_size)
        print(embedding)
        output, hn = self.rnn(embedding)
        return output, hn

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size)), Variable(torch.zeros(1, 1, self.hidden_size))


class ContextEncoder(nn.Module):
    def __init__(self):
        super(ContextEncoder, self).__init__()

    def forward(self):
        pass

    def init_hidden(self):
        pass


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self):
        pass

    def init_hidden(self):
        pass
