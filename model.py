import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from util import get_length
# from unit_test import *
#from dataset import *


class Embedding(nn.Module):
    """
    input: (batch_size, seq_length)
    output: (batch_size, seq_length, embedding_dim)
    """
    def __init__(self, num_embeddings, embedding_dim, init_embedding=None, trainable=False):
        # init_embedding: 2d matrix of pre-trained word embedding
        # row 0 used for padding
        super(Embedding, self).__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings + 1, embedding_dim
        self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))

        self.padding_idx = 0
        self.max_norm = None
        self.norm_type = 2
        self.scale_grad_by_freq = False
        self.sparse = False

        self.reset_parameters(init_embedding)
        self.weight.requires_grad = trainable

    def reset_parameters(self, init_embedding=None):
        if not (init_embedding is None):
            self.weight[1:].data.copy_(torch.from_numpy(init_embedding))
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
    input: (batch_size, max_turn, max_len, embedding_dim)
    output: (batch_size, max_turn, hidden_size * direction)
    """
    def __init__(self, batch_size, max_turn, max_len, input_size, hidden_size, rnn_mode='BiLSTM'):
        super(UtteranceEncoder, self).__init__()
        self.batch_size, self.max_turn, self.max_len = batch_size, max_turn, max_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                           bidirectional=True, batch_first=True)

    def forward(self, input, length):
        print(input.size())
        input = input.view(self.batch_size * self.max_turn, self.max_len, -1)
        sorted_len = sorted()
        print(input.size())
        input = pack_padded_sequence(input, length)
        output, _ = self.rnn(input, self.init_hidden(self.batch_size * self.max_turn))
        return output[:, -1, :]

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size))
        c = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size))
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
        # self.hidden = self.init_hidden()

    def forward(self, input, hidden):
        output, hn = self.rnn(input.view(input.size()[0], 1, self.input_size), hidden)
        return output.view(-1, self.hidden_size), hn

    def init_hidden(self):
        return Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))


class HREDDecoder(nn.Module):
    """
    input: (batch_size, context_size) and (batch_size, seq_length, input_size)
    output: (batch_size, seq_length, output_size)
    """
    def __init__(self, input_size, context_size, hidden_size, output_size):
        super(HREDDecoder, self).__init__()
        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = 1
        self.max_seq_len = 30

        self.input_transform = nn.Linear(context_size, hidden_size, bias=True)
        self.rnn = nn.GRU(input_size, hidden_size, self.num_layers, batch_first=True)
        self.output_transform = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, context, word):
        hn = self.init_hidden(context)
        output, _ = self.rnn(word, hn)
        output = F.tanh(self.output_transform(output))
        return output

    def generate(self, context, word):
        pass

    def init_hidden(self, context):
        return F.tanh(self.input_transform(context)).view(1, 1, -1)


class VHREDDecoder(nn.Module):
    def __init__(self):
        super(VHREDDecoder, self).__init__()

    def forward(self):
        pass


def train():
    dataset = DummyDataset(4, 16, 5, 10, 50, 20)

    embed = Embedding(50, 20, dataset.get_embedding())
    uenc = UtteranceEncoder(4, 5, 10, 20, 20)
    cenc = ContextEncoder(40, 30, 1)
    dec = HREDDecoder(20, 30, 20, 50)

    params = list(uenc.parameters()) + list(cenc.parameters()) + list(dec.parameters())
    # print(params)
    optim = Adam(params)

    dataset.start_iter()
    batch_size = dataset.batch_size
    max_turn = dataset.max_turn
    max_len = dataset.max_len
    while True:
        batch = dataset.get_next_batch()
        if batch is None:
            break
        print(batch)
        len_turns, len_attrs = get_length(batch)
        print(len_turns, len_attrs)
        batch = torch.LongTensor(batch)
        embedded = embed(batch.view(batch_size * max_turn, max_len)).view(batch_size, max_turn, max_len, -1)
        print(embedded.size())

        # embedded = pack_padded_sequence(embedded, length, batch_first=True)
        # print(embedded.batch_sizes)
        u_repr = uenc(embedded, len_attrs)
    """
    for dialog in train_data:
        total_loss = 0
        hn = cenc.init_hidden()
        for i in range(len(dialog)-1):
            source, target = dialog[i], dialog[i+1]
            # print(source, target)
            source, target = torch.LongTensor(source).view(1, -1), torch.LongTensor(target).view(1, -1)
            # print(source, target)
            u_repr = uenc(embed(source))
            # print(u_repr.size())
            c_repr, hn = cenc(u_repr, hn)
            # print(c_repr.size())
            prdt = dec(c_repr, embed(target))[0]
            # print(prdt.size())
            loss = F.cross_entropy(F.softmax(prdt, dim=1), Variable(target).view(-1))
            # print(loss)
            total_loss += loss

        optim.zero_grad()
        total_loss.backward()
        optim.step()
    """


if __name__ == '__main__':
    train()
