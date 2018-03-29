import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from util import get_length
import pdb
# from unit_test import *
#from dataset import *
from masked_cel import compute_loss


class Embedding(nn.Module):
    """
    input: (batch_size, seq_length)
    output: (batch_size, seq_length, embedding_dim)
    """
    def __init__(self, num_embeddings, embedding_dim, init_embedding=None, trainable=False):
        # init_embedding: 2d matrix of pre-trained word embedding
        # row 0 used for padding
        super(Embedding, self).__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))

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
    input: (batch_size, seq_len, embedding_dim)
    output: (batch_size, hidden_size * direction)
    """
    def __init__(self, input_size, hidden_size):
        super(UtteranceEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                           bidirectional=True, batch_first=True)

    def forward(self, input):
        output, (h, c) = self.rnn(input)
        # return output
        return torch.transpose(h, 0, 1).contiguous().view(-1, 2 * self.hidden_size)

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size))
        c = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size))
        return h, c


class ContextEncoder(nn.Module):
    """
    input: (batch_size, seq_len, input_size)
    output: (batch_size, hidden_size)
    """
    def __init__(self, input_size, hidden_size):
        super(ContextEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)

    def forward(self, input):
        output, hn = self.rnn(input)
        # return output
        return torch.transpose(hn, 0, 1).contiguous().view(-1, self.hidden_size)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))


class HREDDecoder(nn.Module):
    """
    input: (batch_size, input_size) context: (batch_size, context_size)
    output: (batch_size, output_size) hn: (1, batch_size, hidden_size)
    one step at a time
    """
    def __init__(self, input_size, context_size, hidden_size, output_size):
        super(HREDDecoder, self).__init__()
        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = 1

        self.context_hidden_transform = nn.Linear(context_size, hidden_size, bias=True)
        self.rnn = nn.GRU(input_size, hidden_size, self.num_layers, batch_first=True)
        self.output_transform = nn.Linear(hidden_size, output_size)

    def forward(self, input, hn):
        output, hn = self.rnn(input.view(input.size()[0], 1, -1), hn)
        output = self.output_transform(output[:,0,:])
        return output, hn

    def generate(self, context, word):
        pass

    def init_hidden(self, context):
        return F.tanh(self.context_hidden_transform(context.view(1, context.size()[0], -1)))


class LatentVariableEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(LatentVariableEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.mean_transform = nn.Linear(input_size, output_size, bias=True)
        self.var_lin_transform = nn.Linear(input_size, output_size, bias=True)
        self.var_sp_transform = nn.Softplus()

    def mean(self, input):
        return self.mean_transform(input)

    def var(self, input):
        return self.var_sp_transform(self.var_lin_transform(input)) * 0.01

    def forward(self, input):
        return self.mean(input), self.var(input)

    def sample(self, input):
        mean = self.mean(input).cpu().data.numpy()
        var = self.var(input).cpu().data.numpy()
        output = np.zeros((mean.shape))
        for x in range(output.shape[0]):
            output[x] = np.random.multivariate_normal(mean[x], np.diag(var[x])) 
        return Variable(torch.from_numpy(output)).float()


class VHREDDecoder(nn.Module):
    def __init__(self):
        super(VHREDDecoder, self).__init__()

    def forward(self):
        pass


class HRED(nn.Module):
    def __init__(self, dictionary, vocab_size, dim_embedding, init_embedding, hidden_size):
        super(HRED, self).__init__()
        self.dictionary = dictionary
        self.embedding = Embedding(vocab_size, dim_embedding, init_embedding, trainable=True).cuda()
        self.u_encoder = UtteranceEncoder(dim_embedding, hidden_size).cuda()
        self.cenc_input_size = hidden_size * 2
        self.c_encoder = ContextEncoder(self.cenc_input_size, hidden_size).cuda()
        self.decoder = HREDDecoder(dim_embedding, hidden_size, hidden_size, len(dictionary)).cuda()
        self.hidden_size = hidden_size

    def parameters(self):
        return list(self.u_encoder.parameters()) + list(self.c_encoder.parameters()) \
               + list(self.decoder.parameters()) + list(self.embedding.parameters())

    def loss(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, ctc_lengths):
        src_seqs = self.embedding(Variable(src_seqs.cuda()))
        # src_seqs: (N, max_uttr_len, word_dim)
        uenc_packed_input = pack_padded_sequence(src_seqs, src_lengths, batch_first=True)
        uenc_output = self.u_encoder(uenc_packed_input)
        # output: (N, dim1)
        _batch_size = len(ctc_lengths)
        max_len = max(ctc_lengths)
        cenc_in = Variable(torch.zeros(_batch_size, max_len, self.cenc_input_size).float()).cuda()
        for i in range(len(indices)):
            x, y = indices[i]
            cenc_in[x, y, :] = uenc_output[i]
        # cenc_in: (batch_size, max_turn, dim1)
        ctc_lengths, perm_idx = torch.cuda.LongTensor(ctc_lengths).sort(0, descending=True)
        cenc_in = cenc_in[perm_idx, :, :]
        # cenc_in: (batch_size, max_turn, dim1)
        trg_seqs = trg_seqs.cuda()[perm_idx]
        trg_lengths = Variable(torch.cuda.LongTensor(trg_lengths))[perm_idx]
        max_len = trg_lengths.max().data[0]
        # trg_seqs: (batch_size, max_trg_len)
        cenc_packed_input = pack_padded_sequence(cenc_in, ctc_lengths.cpu().numpy(), batch_first=True)
        cenc_out = self.c_encoder(cenc_packed_input)
        # cenc_out: (batch_size, dim2)
        decoder_hidden = self.decoder.init_hidden(cenc_out)
        decoder_input = self.embedding(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>'])))
        decoder_outputs = Variable(torch.zeros(_batch_size, max_len - 1, len(self.dictionary))).cuda()
        for t in range(1, max_len):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden
            )
            decoder_outputs[:, t - 1, :] = decoder_output
            decoder_input = self.embedding(trg_seqs[:, t])

        loss = compute_loss(decoder_outputs, Variable(trg_seqs[:, 1:]), trg_lengths - 1)

        return loss

    def flatten_parameters(self):
        self.u_encoder.rnn.flatten_parameters()
        self.c_encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
        

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
