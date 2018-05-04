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
from masked_cel import compute_loss, compute_semantic_loss


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

        self.padding_idx = 0
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


class DiscUtteranceEncoder(nn.Module):
    """
    input: (batch_size, seq_len, embedding_dim)
    output: (batch_size, hidden_size * direction)
    """
    def __init__(self, input_size, hidden_size, type='gru', dp=0):
        super(DiscUtteranceEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.type = type
        if self.type == 'gru':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dp)
        else:
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dp)

    def forward(self, input):
        if self.type == 'gru':
            output, hn = self.rnn(input)
        else:
            output, (hn, cn) = self.rnn(input)
        # return output
        return torch.transpose(hn, 0, 1).contiguous().view(-1, self.hidden_size)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))


class ContextEncoder(nn.Module):
    """
    input: (batch_size, seq_len, input_size)
    output: (batch_size, hidden_size)
    """
    def __init__(self, input_size, hidden_size, type='lstm', dp=0.2):
        super(ContextEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        if self.type == 'gru':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dp)
        else:
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dp)

    def forward(self, input):
        if self.type == 'gru':
            output, hn = self.rnn(input)
        else:
            output, (hn, cn) = self.rnn(input)
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
        mean = self.mean(input)
        var = self.var(input)
        output = np.zeros((mean.shape))
        output = torch.normal(mean, torch.sqrt(var))
        return output


class HRED(nn.Module):
    def __init__(self, dictionary, vocab_size, dim_embedding, init_embedding, hidden_size):
        super(HRED, self).__init__()
        self.dictionary = dictionary
        self.embedding = Embedding(vocab_size, dim_embedding, init_embedding, trainable=False)
        self.u_encoder = UtteranceEncoder(dim_embedding, hidden_size, 'lstm', 0.5)
        self.uu_encoder = UtteranceEncoder(dim_embedding, hidden_size, 'lstm', 0.5)
        self.cenc_input_size = hidden_size
        self.c_encoder = ContextEncoder(self.cenc_input_size, hidden_size, type='gru', dp=0.5)
        self.hidden_size = hidden_size
        '''
        self.w_score = nn.Bilinear(hidden_size, hidden_size, 2)
        self.u_score = nn.Bilinear(hidden_size, hidden_size, 2)
        '''
        self.score = nn.Bilinear(hidden_size * 2, hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.conv = nn.Conv1d(dim_embedding, dim_embedding, 3, padding=1)
        self.pool = nn.MaxPool1d(3, padding=1, stride=1)

        for names in self.uu_encoder.rnn._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.uu_encoder.rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(10.)

        for names in self.u_encoder.rnn._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.u_encoder.rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(10.)

    def parameters(self):
        return list(self.u_encoder.parameters()) + list(self.c_encoder.parameters()) \
               + list(self.uu_encoder.parameters()) + list(self.score.parameters()) \
               + list(self.conv.parameters())

    def loss(self, src_seqs, src_lengths, indices, ctc_seqs, ctc_lengths, ctc_indices, trg_seqs, trg_lengths, trg_indices, turn_len, labels, weights):
        ctc_seqs = self.embedding(Variable(ctc_seqs.cuda()))
        ctc_packed_input = pack_padded_sequence(ctc_seqs, ctc_lengths, batch_first=True)
        ctc_output = self.u_encoder(ctc_packed_input)
        ctc_output = ctc_output[torch.from_numpy(np.argsort(ctc_indices)).cuda()]

        trg_seqs = self.embedding(Variable(trg_seqs.cuda()))
        trg_packed_input = pack_padded_sequence(trg_seqs, trg_lengths, batch_first=True)
        trg_output = self.u_encoder(trg_packed_input)
        trg_output = trg_output[torch.from_numpy(np.argsort(trg_indices)).cuda()]

        src_seqs = self.embedding(Variable(src_seqs.cuda()))
        src_seqs = self.conv(src_seqs.transpose(1, 2))
        src_seqs = self.pool(src_seqs).transpose(1, 2)
        src_packed_input = pack_padded_sequence(src_seqs, src_lengths, batch_first=True)
        src_output = self.uu_encoder(src_packed_input)

        _batch_size = len(turn_len)
        max_len = max(turn_len)
        cenc_in = Variable(torch.zeros(_batch_size, max_len, self.cenc_input_size).float()).cuda()
        for i in range(len(indices)):
            x, y = indices[i]
            cenc_in[x, y, :] = src_output[i]
        turn_len, perm_idx = torch.cuda.LongTensor(turn_len).sort(0, descending=True)
        cenc_in = cenc_in[perm_idx, :, :]
        cenc_packed_input = pack_padded_sequence(cenc_in, turn_len.cpu().numpy(), batch_first=True)
        cenc_out = self.c_encoder(cenc_packed_input)
        cenc_out = cenc_out[perm_idx.sort()[1]]
        logits = self.score(torch.cat((cenc_out, ctc_output), dim=1), trg_output)
        '''
        w_logits = self.w_score(ctc_output, trg_output)
        w_score = F.softmax(w_logits, dim=1)
        u_logits = self.u_score(cenc_out, trg_output)
        u_score = F.softmax(u_logits, dim=1)
        loss = w_score[row_ind, labels] * u_score[row_ind, 1-labels] + w_score[row_ind, 1-labels] * u_score[row_ind, labels]
        loss += w_score[row_ind, 1-labels] + u_score[row_ind, 1-labels]
        '''
        row_ind = torch.arange(_batch_size).long().cuda()
        labels = Variable(torch.cuda.LongTensor(labels))
        #loss = self.criterion(w_logits, labels) + self.criterion(u_logits, labels) / 2
        #loss = self.criterion(w_logits, Variable(torch.cuda.LongTensor(labels)) )
        score = F.log_softmax(logits, dim=1)
        weights = Variable(torch.cuda.FloatTensor(weights))
        loss = -score[row_ind, labels] * weights + F.binary_cross_entropy_with_logits(logits, F.softmax(logits, dim=1)) * (1 - weights)
        #loss = self.criterion(logits, labels)

        return loss.sum() / weights.sum()
        #return loss

    def evaluate(self, src_seqs, src_lengths, indices, ctc_seqs, ctc_lengths, ctc_indices, trg_seqs, trg_lengths, trg_indices, turn_len, labels, weights):
        ctc_seqs = self.embedding(Variable(ctc_seqs.cuda()))
        ctc_packed_input = pack_padded_sequence(ctc_seqs, ctc_lengths, batch_first=True)
        ctc_output = self.u_encoder(ctc_packed_input)
        ctc_output = ctc_output[torch.from_numpy(np.argsort(ctc_indices)).cuda()]

        trg_seqs = self.embedding(Variable(trg_seqs.cuda()))
        trg_packed_input = pack_padded_sequence(trg_seqs, trg_lengths, batch_first=True)
        trg_output = self.u_encoder(trg_packed_input)
        trg_output = trg_output[torch.from_numpy(np.argsort(trg_indices)).cuda()]

        src_seqs = self.embedding(Variable(src_seqs.cuda()))
        src_seqs = self.conv(src_seqs.transpose(1, 2)).transpose(1, 2)
        #src_seqs = F.relu(src_seqs)
        src_seqs = self.pool(src_seqs.transpose(1, 2)).transpose(1, 2)
        src_packed_input = pack_padded_sequence(src_seqs, src_lengths, batch_first=True)
        src_output = self.uu_encoder(src_packed_input)

        _batch_size = len(turn_len)
        max_len = max(turn_len)
        cenc_in = Variable(torch.zeros(_batch_size, max_len, self.cenc_input_size).float()).cuda()
        for i in range(len(indices)):
            x, y = indices[i]
            cenc_in[x, y, :] = src_output[i]
        turn_len, perm_idx = torch.cuda.LongTensor(turn_len).sort(0, descending=True)
        cenc_in = cenc_in[perm_idx, :, :]
        cenc_packed_input = pack_padded_sequence(cenc_in, turn_len.cpu().numpy(), batch_first=True)
        cenc_out = self.c_encoder(cenc_packed_input)
        cenc_out = cenc_out[perm_idx.sort()[1]]
        logits = self.score(torch.cat((cenc_out, ctc_output), dim=1), trg_output)
        score = F.softmax(logits, dim=1)
        '''
        w_logits = self.w_score(ctc_output, trg_output)
        w_score = F.softmax(w_logits, dim=1)
        u_logits = self.u_score(cenc_out, trg_output)
        u_score = F.softmax(u_logits, dim=1)
        u_logits = self.u_score(cenc_out, res_seqs)
        u_score = F.softmax(u_logits, dim=1)
        rev_perm_idx = perm_idx.sort()[1]
        '''

        #return (w_score + u_score)[:, 1], labels
        return score[:, 1]

    def flatten_parameters(self):
        self.u_encoder.rnn.flatten_parameters()
        self.uu_encoder.rnn.flatten_parameters()
        self.c_encoder.rnn.flatten_parameters()


class AugLSTM(nn.Module):
    def __init__(self, dictionary, vocab_size, dim_embedding, init_embedding, hidden_size, dp=0):
        super(AugLSTM, self).__init__()
        self.dictionary = dictionary
        self.embedding = Embedding(vocab_size, dim_embedding, init_embedding, trainable=True)
        self.u_encoder = UtteranceEncoder(dim_embedding, hidden_size, 'lstm', dp)
        self.hidden_size = hidden_size
        self.score = nn.Bilinear(hidden_size * 2, hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()

        for names in self.u_encoder.rnn._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.u_encoder.rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(10.)

    def parameters(self):
        return list(self.u_encoder.parameters()) + list(self.score.parameters())

    def wv_parameters(self):
        return self.embedding.parameters()

    def loss(self, src_seqs, aug_src_seqs, src_lengths, src_indices, trg_seqs, trg_lengths, trg_indices, labels, weights):
        src_seqs = self.embedding(Variable(src_seqs.cuda()))
        trg_seqs = self.embedding(Variable(trg_seqs.cuda()))
        src_input = pack_padded_sequence(src_seqs, src_lengths, batch_first=True)
        trg_input = pack_padded_sequence(trg_seqs, trg_lengths, batch_first=True)
        src_output = self.u_encoder(src_input)
        trg_output = self.u_encoder(trg_input)
        src_output = src_output[torch.from_numpy(np.argsort(src_indices)).cuda()]
        trg_output = trg_output[torch.from_numpy(np.argsort(trg_indices)).cuda()]
        aug_src_seqs = self.embedding(Variable(aug_src_seqs.cuda()))
        aug_src_input = pack_padded_sequence(aug_src_seqs, src_lengths, batch_first=True)
        aug_src_output = self.u_encoder(aug_src_input)
        aug_src_output = aug_src_output[torch.from_numpy(np.argsort(src_indices)).cuda()]
        batch_size = len(labels)
        row_ind = torch.arange(batch_size).long().cuda()
        logits = self.score(torch.cat((src_output, aug_src_output), dim=1), trg_output)
        loss = self.criterion(logits, Variable(torch.cuda.LongTensor(labels)))

        return loss

    def evaluate(self, src_seqs, aug_src_seqs, src_lengths, src_indices, trg_seqs, trg_lengths, trg_indices):
        src_seqs = self.embedding(Variable(src_seqs.cuda()))
        trg_seqs = self.embedding(Variable(trg_seqs.cuda()))
        src_input = pack_padded_sequence(src_seqs, src_lengths, batch_first=True)
        trg_input = pack_padded_sequence(trg_seqs, trg_lengths, batch_first=True)
        src_output = self.u_encoder(src_input)
        trg_output = self.u_encoder(trg_input)
        src_output = src_output[torch.from_numpy(np.argsort(src_indices)).cuda()]
        trg_output = trg_output[torch.from_numpy(np.argsort(trg_indices)).cuda()]
        aug_src_seqs = self.embedding(Variable(aug_src_seqs.cuda()))
        aug_src_input = pack_padded_sequence(aug_src_seqs, src_lengths, batch_first=True)
        aug_src_output = self.u_encoder(aug_src_input)
        aug_src_output = aug_src_output[torch.from_numpy(np.argsort(src_indices)).cuda()]
        #logits = self.score(src_output, trg_output)
        logits = self.score(torch.cat((src_output, aug_src_output), dim=1), trg_output)

        return F.softmax(logits, dim=1)[:, 1]

    def flatten_parameters(self):
        self.u_encoder.rnn.flatten_parameters()


class LSTM(nn.Module):
    def __init__(self, dictionary, vocab_size, dim_embedding, init_embedding, hidden_size, dp=0):
        super(LSTM, self).__init__()
        self.dictionary = dictionary
        self.embedding = Embedding(vocab_size, dim_embedding, init_embedding, trainable=True)
        self.u_encoder = DiscUtteranceEncoder(dim_embedding, hidden_size, 'lstm', dp)
        self.hidden_size = hidden_size
        self.score = nn.Bilinear(hidden_size, hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()

        for names in self.u_encoder.rnn._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.u_encoder.rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(10.)

    def parameters(self):
        return list(self.u_encoder.parameters()) + list(self.score.parameters())

    def wv_parameters(self):
        return self.embedding.parameters()

    def loss(self, src_seqs, src_lengths, src_indices, trg_seqs, trg_lengths, trg_indices, labels, weights):
        src_seqs = self.embedding(Variable(src_seqs.cuda()))
        trg_seqs = self.embedding(Variable(trg_seqs.cuda()))
        src_input = pack_padded_sequence(src_seqs, src_lengths, batch_first=True)
        trg_input = pack_padded_sequence(trg_seqs, trg_lengths, batch_first=True)
        src_output = self.u_encoder(src_input)
        trg_output = self.u_encoder(trg_input)
        src_output = src_output[torch.from_numpy(np.argsort(src_indices)).cuda()]
        trg_output = trg_output[torch.from_numpy(np.argsort(trg_indices)).cuda()]
        logits = self.score(src_output, trg_output)
        loss = self.criterion(logits, Variable(torch.cuda.LongTensor(labels)))

        return loss

    def evaluate(self, src_seqs, src_lengths, src_indices, trg_seqs, trg_lengths, trg_indices):
        src_seqs = self.embedding(Variable(src_seqs.cuda()))
        trg_seqs = self.embedding(Variable(trg_seqs.cuda()))
        src_input = pack_padded_sequence(src_seqs, src_lengths, batch_first=True)
        trg_input = pack_padded_sequence(trg_seqs, trg_lengths, batch_first=True)
        src_output = self.u_encoder(src_input)
        trg_output = self.u_encoder(trg_input)
        print(np.shape(src_indices))
        print(np.shape(np.argsort(src_indices)))
        src_output = src_output[torch.from_numpy(np.argsort(src_indices)).cuda()]
        trg_output = trg_output[torch.from_numpy(np.argsort(trg_indices)).cuda()]
        logits = self.score(src_output, trg_output)

        return F.softmax(logits, dim=1)[:, 1]

    def flatten_parameters(self):
        self.u_encoder.rnn.flatten_parameters()



if __name__ == '__main__':
    train()
