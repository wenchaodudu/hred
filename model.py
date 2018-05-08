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
from gumbel_softmax import gumbel_softmax
from discriminator import LSTM, DiscUtteranceEncoder


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

        self.context_fc1 = nn.Linear(context_size, hidden_size, bias=True)
        self.context_fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.rnn = nn.GRU(input_size, hidden_size, self.num_layers, batch_first=True)
        self.output_transform = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, input, hn):
        output, hn = self.rnn(input.contiguous().view(input.size()[0], 1, -1), hn)
        output = self.output_transform(output[:,0,:])
        return output, hn

    def init_hidden(self, context):
        #return F.tanh(self.context_hidden_transform(context.view(1, context.size()[0], -1)))
        return self.context_fc2(F.relu(self.context_fc1(context.view(1, context.size()[0], -1))))


class HREDMoSDecoder(nn.Module):
    """
    input: (batch_size, input_size) context: (batch_size, context_size)
    output: (batch_size, output_size) hn: (1, batch_size, hidden_size)
    one step at a time
    """
    def __init__(self, input_size, context_size, hidden_size, output_size):
        super(HREDMoSDecoder, self).__init__()
        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = 1

        self.context_hidden_transform = nn.Linear(context_size, hidden_size, bias=True)
        self.rnn = nn.GRU(input_size, hidden_size, self.num_layers, batch_first=True)
        self.output_transform = nn.Linear(hidden_size, output_size)
        self.K = 5
        self.prior = nn.Linear(hidden_size, self.K)
        self.W = nn.Linear(hidden_size, self.K * hidden_size)

    def forward(self, input, hn):
        output, hn = self.rnn(input.contiguous().view(input.size()[0], 1, -1), hn)
        pi = F.softmax(self.prior(output[:, 0, :]), dim=1)
        h = torch.tanh(self.W(output[:, 0, :])).view(-1, self.K, self.hidden_size)
        output = torch.bmm(pi.unsqueeze(1), F.softmax(self.output_transform(h), dim=1)).squeeze(1)
        return torch.log(output), hn

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


class QNetwork(nn.Module):
    def __init__(self, hidden_size, embedding_size):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(nn.Linear(hidden_size + embedding_size, 600),
                                   nn.ReLU(),
                                   nn.Linear(600, 450),
                                   nn.ReLU(), 
                                   nn.Linear(450, embedding_size),
                                  )

    def forward(self, input):
        return self.model(input)


class HRED(nn.Module):
    def __init__(self, dictionary, vocab_size, dim_embedding, init_embedding, hidden_size, discriminator=None, decoder='normal'):
        super(HRED, self).__init__()
        self.dictionary = dictionary
        self.embedding = Embedding(vocab_size, dim_embedding, init_embedding, trainable=False).cuda()
        self.u_encoder = UtteranceEncoder(dim_embedding, hidden_size).cuda()
        self.cenc_input_size = hidden_size * 2
        self.c_encoder = ContextEncoder(self.cenc_input_size, hidden_size).cuda()
        if decoder == 'mos':
            self.decoder = HREDMoSDecoder(dim_embedding, hidden_size, hidden_size, vocab_size).cuda()
        else:
            self.decoder = HREDDecoder(dim_embedding, hidden_size, hidden_size, vocab_size).cuda()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        if discriminator:
            self.discriminator = discriminator
        self.eou = self.dictionary['__eou__']
        self.gumbel_noise = Variable(torch.FloatTensor(64)).cuda()
        self.MSE = nn.MSELoss()
        
    '''
    def parameters(self):
        return list(self.u_encoder.parameters()) + list(self.c_encoder.parameters()) \
               + list(self.decoder.parameters())
    '''

    def encode(self, src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len):
        batch_size = len(turn_len)
        max_len = max(turn_len)

        src_embed = self.embedding(Variable(src_seqs.cuda()))
        if src_lengths[-1] == 0:
            nonempty = src_lengths.index(0)
        else:
            nonempty = len(src_lengths)
        src_packed_input = pack_padded_sequence(src_embed[:nonempty], src_lengths[:nonempty], batch_first=True)
        src_output = self.u_encoder(src_packed_input)
        if nonempty < len(src_lengths):
            src_output = torch.cat((src_output, Variable(torch.zeros(len(src_lengths) - nonempty, self.cenc_input_size)).cuda()), dim=0)
        src_output = src_output[torch.from_numpy(np.argsort(src_indices)).cuda()]
        src_output = src_output.view(batch_size, 5, self.cenc_input_size)

        turn_len, perm_idx = torch.cuda.LongTensor(turn_len).sort(0, descending=True)
        cenc_in = src_output[perm_idx, :, :]
        cenc_packed_input = pack_padded_sequence(cenc_in, turn_len.cpu().numpy(), batch_first=True)
        cenc_out = self.c_encoder(cenc_packed_input)
        cenc_out = cenc_out[perm_idx.sort()[1]]

        return cenc_out

    def gumbel_sampler(self, logits, noise, temperature=0.5):
        eps = 1e-20
        noise.data.add_(eps).log_().neg_()
        noise.data.add_(eps).log_().neg_()
        y = (logits + noise) / temperature
        y = F.softmax(y, dim=1)

        max_val, max_idx = torch.max(y, y.dim()-1)
        y_hard = y == max_val.view(-1,1).expand_as(y)
        y = (y_hard.float() - y).detach() + y

        return y, max_idx.view(1, -1)
        

    def decode(self, cenc_out, trg_seqs, trg_lengths, trg_indices, sampling_rate, gumbel=False):
        trg_seqs = self.embedding(Variable(trg_seqs.cuda()))
        trg_output = trg_seqs[torch.from_numpy(np.argsort(trg_indices)).cuda()]
        decoder_hidden = self.decoder.init_hidden(cenc_out)
        _batch_size = cenc_out.size(0)
        max_len = max(trg_lengths)
        decoder_input = self.embedding(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>'])))
        decoder_outputs = Variable(torch.zeros(_batch_size, max_len - 1, self.vocab_size)).cuda()
        gumbel_outputs = torch.zeros(_batch_size, max_len - 1).long().cuda()
        for t in range(1, max_len):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden
            )
            if gumbel:
                self.gumbel_noise.data.resize_(_batch_size, max_len, self.vocab_size)
                self.gumbel_noise.data.uniform_(0, 1)
                one_hot, idx = self.gumbel_sampler(decoder_output, self.gumbel_noise[:, t, :]) 
                decoder_outputs[:, t - 1, :] = decoder_output
                if np.random.uniform() > sampling_rate:
                    decoder_input = trg_seqs[:, t]
                else:
                    decoder_input = self.embedding(decoder_output.max(1)[1])
                gumbel_outputs[:, t - 1] = idx.squeeze(0).data
            else:
                decoder_outputs[:, t - 1, :] = decoder_output
                if np.random.uniform() > sampling_rate:
                    decoder_input = trg_seqs[:, t]
                else:
                    decoder_input = self.embedding(decoder_output.max(1)[1])
            
        if gumbel:
            return decoder_outputs, gumbel_outputs
        else:
            return decoder_outputs

    def loss(self, src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, trg_seqs, trg_lengths, trg_indices, turn_len, sampling_rate):
        cenc_out = self.encode(src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len)
        decoder_outputs = self.decode(cenc_out, trg_seqs, trg_lengths, trg_indices, sampling_rate)
        decoder_outputs = decoder_outputs[list(trg_indices)]
        trg_output = Variable(trg_seqs.cuda())
        loss = compute_loss(decoder_outputs, trg_output[:, 1:], Variable(torch.cuda.LongTensor(trg_lengths)) - 1)

        return loss

    def semantic_loss(self, src_seqs, src_lengths, indices, ctc_seqs, ctc_lengths, ctc_indices, trg_seqs, trg_lengths, trg_indices, turn_len):
        cenc_out = self.encode(src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len)
        decoder_outputs = self.decode(cenc_out, trg_seqs, trg_lengths, trg_indices, 0)
        decoder_outputs = decoder_outputs[list(trg_indices)]
        trg_output = Variable(trg_seqs.cuda())
        loss = compute_semantic_loss(decoder_outputs, trg_output[:, 1:], Variable(torch.cuda.LongTensor(trg_lengths)) - 1)

        return loss

    def augmented_loss(self, src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, trg_seqs, trg_lengths, trg_indices, turn_len, sampling_rate):
        cenc_out = self.encode(src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len)
        decoder_outputs, gumbel_outputs = self.decode(cenc_out, trg_seqs, trg_lengths, trg_indices, sampling_rate, True)
        decoder_outputs = decoder_outputs[list(trg_indices)]
        gumbel_outputs = gumbel_outputs[list(trg_indices)]
        trg_output = Variable(trg_seqs.cuda())
        nll_loss = compute_loss(decoder_outputs, trg_output[:, 1:], Variable(torch.cuda.LongTensor(trg_lengths)) - 1)
        d_loss = self.discriminator.evaluate(ctc_seqs, ctc_lengths, ctc_indices, gumbel_outputs, [ll - 1 for ll in trg_lengths], trg_indices)
        
        return nll_loss - torch.log(d_loss).sum() / trg_seqs.size(0) * .3


    def generate(self, src_seqs, src_lengths, indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len, max_len, beam_size, top_k):
        cenc_out = self.encode(src_seqs, src_lengths, indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len)
        decoder_hidden = self.decoder.init_hidden(cenc_out)
        _batch_size = cenc_out.size(0)
        decoder_input = self.embedding(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>'])))
        decoder_outputs = Variable(torch.zeros(_batch_size, max_len - 1, self.vocab_size)).cuda()
        # cenc_out: (batch_size, dim2)
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        decoder_input = self.embedding(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>'])))
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), beam_size, dim=1)
        #decoder_input = self.embedding(argtop[0])
        beam = Variable(torch.zeros(_batch_size, beam_size, max_len)).long().cuda()
        beam[:, :, 0] = argtop
        #beam_probs = logprobs[0].clone()
        beam_probs = logprobs.clone()
        #beam_eos = (argtop == self.eou)[0].data
        beam_eos = (argtop == self.eou).data
        decoder_hidden = decoder_hidden.unsqueeze(2).expand(1, _batch_size, beam_size, self.hidden_size).contiguous().view(1, _batch_size * beam_size, -1)
        decoder_input = self.embedding(argtop.view(-1))
        for t in range(max_len-1):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden
            )
            logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), top_k, dim=1)
            #best_probs, best_args = (beam_probs.repeat(top_k, 1).transpose(0, 1) + logprobs).view(-1).topk(beam_size)
            best_probs, best_args = (beam_probs.view(-1).unsqueeze(1).expand(_batch_size * beam_size, top_k) + logprobs).view(_batch_size, beam_size, -1).view(_batch_size, -1).topk(beam_size)
            decoder_hidden = decoder_hidden.view(1, _batch_size, beam_size, -1)
            for x in range(_batch_size):
                last = (best_args / top_k)[x]
                curr = (best_args % top_k)[x]
                beam[x, :, :] = beam[x][last, :]
                beam[x, :, t+1] = argtop.view(_batch_size, beam_size, top_k)[x][last.data, curr.data] * Variable(~beam_eos[x]).long() + eos_filler * Variable(beam_eos[x]).long()
                beam_eos[x, :] = beam_eos[x][last.data]
                beam_probs[x, :] = beam_probs[x][last.data]
                ind1 = torch.cuda.LongTensor(beam_size).fill_(x)
                ind2 = (~beam_eos[x]).long()
                beam_probs[ind1, ind2] = (beam_probs[ind1, ind2] * (t+1) + best_probs[ind1, ind2]) / (t+2)
                decoder_hidden[:, x, :, :] = decoder_hidden[:, x, :, :][:, last, :]
            beam_eos = beam_eos | (beam[:, :, t+1] == self.eou).data
            decoder_hidden = decoder_hidden.view(1, _batch_size * beam_size, -1)
            decoder_input = self.embedding(beam[:, :, t+1].contiguous().view(-1))
        best, best_arg = beam_probs.max(1)
        generations = beam[torch.arange(_batch_size).long().cuda(), best_arg.data].data.cpu()
        return generations, best


    def random_sample(self, src_seqs, src_lengths, indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len, max_len, beam_size, top_k):
        cenc_out = self.encode(src_seqs, src_lengths, indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len)
        decoder_hidden = self.decoder.init_hidden(cenc_out)
        _batch_size = cenc_out.size(0)
        decoder_input = self.embedding(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>'])))
        decoder_outputs = Variable(torch.zeros(_batch_size, max_len - 1, self.vocab_size)).cuda()
        # cenc_out: (batch_size, dim2)
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        decoder_input = self.embedding(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>'])))
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), beam_size, dim=1)
        #decoder_input = self.embedding(argtop[0])
        beam = Variable(torch.zeros(_batch_size, beam_size, max_len)).long().cuda()
        beam[:, :, 0] = argtop
        #beam_probs = logprobs[0].clone()
        beam_probs = logprobs.clone()
        #beam_eos = (argtop == self.eou)[0].data
        beam_eos = (argtop == self.eou).data
        decoder_hidden = decoder_hidden.unsqueeze(2).expand(1, _batch_size, beam_size, self.hidden_size).contiguous().view(1, _batch_size * beam_size, -1)
        decoder_input = self.embedding(argtop.view(-1))
        for t in range(max_len-1):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden
            )
            word_probs = F.softmax(decoder_output, dim=1)
            argtop = torch.multinomial(word_probs, top_k, replacement=False)
            logprobs = Variable(torch.zeros(_batch_size * beam_size, top_k).float().cuda())
            for x in range(top_k):
                logprobs[:, x] = torch.log(word_probs[torch.arange(_batch_size * beam_size).long().cuda(), argtop[:, x].data])
            #logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), top_k, dim=1)
            best_probs, best_args = (beam_probs.view(-1).unsqueeze(1).expand(_batch_size * beam_size, top_k) + logprobs).view(_batch_size, beam_size, -1).view(_batch_size, -1).topk(beam_size)
            decoder_hidden = decoder_hidden.view(1, _batch_size, beam_size, -1)
            for x in range(_batch_size):
                last = (best_args / top_k)[x]
                curr = (best_args % top_k)[x]
                beam[x, :, :] = beam[x][last, :]
                beam[x, :, t+1] = argtop.view(_batch_size, beam_size, top_k)[x][last.data, curr.data] * Variable(~beam_eos[x]).long() + eos_filler * Variable(beam_eos[x]).long()
                beam_eos[x, :] = beam_eos[x][last.data]
                beam_probs[x, :] = beam_probs[x][last.data]
                ind1 = torch.cuda.LongTensor(beam_size).fill_(x)
                ind2 = (~beam_eos[x]).long()
                beam_probs[ind1, ind2] = (beam_probs[ind1, ind2] * (t+1) + best_probs[ind1, ind2]) / (t+2)
                decoder_hidden[:, x, :, :] = decoder_hidden[:, x, :, :][:, last, :]
            beam_eos = beam_eos | (beam[:, :, t+1] == self.eou).data
            decoder_hidden = decoder_hidden.view(1, _batch_size * beam_size, -1)
            decoder_input = self.embedding(beam[:, :, t+1].contiguous().view(-1))
        best, best_arg = beam_probs.max(1)
        generations = beam[torch.arange(_batch_size).long().cuda(), best_arg.data].data.cpu()
        return generations, best


    def train_decoder(self, src_seqs, src_lengths, indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len, max_len, beam_size, top_k, discriminator, qnet, weight):
        cenc_out = self.encode(src_seqs, src_lengths, indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len)
        decoder_hidden = self.decoder.init_hidden(cenc_out)
        _batch_size = cenc_out.size(0)
        decoder_input = self.embedding(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>'])))
        decoder_outputs = Variable(torch.zeros(_batch_size, max_len - 1, self.vocab_size)).cuda()
        # cenc_out: (batch_size, dim2)
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        decoder_input = self.embedding(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>'])))
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), beam_size, dim=1)
        #decoder_input = self.embedding(argtop[0])
        beam = Variable(torch.zeros(_batch_size, beam_size, max_len)).long().cuda()
        beam[:, :, 0] = argtop
        #beam_probs = logprobs[0].clone()
        beam_probs = logprobs.clone()
        #beam_eos = (argtop == self.eou)[0].data
        beam_eos = (argtop == self.eou).data
        hidden_states = torch.zeros(_batch_size, beam_size, max_len, self.hidden_size).float().cuda()
        decoder_hidden = decoder_hidden.unsqueeze(2).expand(1, _batch_size, beam_size, self.hidden_size).contiguous()
        hidden_states[:, :, 0, :] = decoder_hidden[0].data
        decoder_hidden = decoder_hidden.view(1, _batch_size * beam_size, -1)
        decoder_input = self.embedding(argtop.view(-1))
        disc_ctc = discriminator.encode_context(ctc_seqs, ctc_lengths, ctc_indices)
        disc_ctc_encoding = disc_ctc.unsqueeze(1).expand(_batch_size, beam_size * top_k, disc_ctc.size(1)).contiguous().view(-1, disc_ctc.size(1))
        for t in range(max_len-1):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden
            )
            word_probs = F.softmax(decoder_output, dim=1)
            argtop = torch.multinomial(word_probs, top_k, replacement=False)
            logprobs = Variable(torch.zeros(_batch_size * beam_size, top_k).float().cuda())
            for x in range(top_k):
                logprobs[:, x] = torch.log(word_probs[torch.arange(_batch_size * beam_size).long().cuda(), argtop[:, x].data])
            pred_input = self.embedding(argtop.view(-1))
            d_score = torch.log(discriminator.score_(disc_ctc_encoding, pred_input))
            #logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), top_k, dim=1)
            logprobs += weight * d_score.contiguous().view(-1, top_k)
            best_probs, best_args = (beam_probs.view(-1).unsqueeze(1).expand(_batch_size * beam_size, top_k) + logprobs).view(_batch_size, beam_size, -1).view(_batch_size, -1).topk(beam_size)
            decoder_hidden = decoder_hidden.view(1, _batch_size, beam_size, -1)
            for x in range(_batch_size):
                last = (best_args / top_k)[x]
                curr = (best_args % top_k)[x]
                beam[x, :, :] = beam[x][last, :]
                beam[x, :, t+1] = argtop.view(_batch_size, beam_size, top_k)[x][last.data, curr.data] * Variable(~beam_eos[x]).long() + eos_filler * Variable(beam_eos[x]).long()
                beam_eos[x, :] = beam_eos[x][last.data]
                beam_probs[x, :] = beam_probs[x][last.data]
                ind1 = torch.cuda.LongTensor(beam_size).fill_(x)
                ind2 = (~beam_eos[x]).long()
                beam_probs[ind1, ind2] = (beam_probs[ind1, ind2] * (t+1) + best_probs[ind1, ind2]) / (t+2)
                decoder_hidden[:, x, :, :] = decoder_hidden[:, x, :, :][:, last, :]
                hidden_states[x, :, :, :] = hidden_states[x, :, :, :][last.data, :, :]
                hidden_states[x, :, t+1, :] = decoder_hidden[0, x, :, :].data
            beam_eos = beam_eos | (beam[:, :, t+1] == self.eou).data
            decoder_hidden = decoder_hidden.view(1, _batch_size * beam_size, -1)
            decoder_input = self.embedding(beam[:, :, t+1].contiguous().view(-1))
        best, best_arg = beam_probs.max(1)
        generations = beam[torch.arange(_batch_size).long().cuda(), best_arg.data]

        hidden_states = hidden_states[torch.arange(_batch_size).long().cuda(), best_arg.data]
        # a hack for getting the lengths of generated responses
        eous, lengths = generations.min(1)
        lengths[eous != 2] = max_len
        lengths[lengths == 0] = 1
        len_ind = torch.max((lengths - 3).data, torch.zeros(_batch_size).long().cuda()+1)
        states = hidden_states[torch.arange(_batch_size).long().cuda(), len_ind, :]
        inputs = self.embedding(generations[torch.arange(_batch_size).long().cuda(), len_ind+1])
        inputs = torch.cat((Variable(states), inputs), dim=1)

        lengths, perm_idx = lengths.sort(0, descending=True)
        generations = generations[perm_idx]
        inputs = inputs[perm_idx]
        packed_input = pack_padded_sequence(self.embedding(generations), lengths.data.cpu().numpy(), batch_first=True)
        output = discriminator.u_encoder(packed_input).detach()
        pred = qnet(inputs)
        loss = self.MSE(pred, output)
        output = output[perm_idx.sort()[1]]
        generations = generations[perm_idx.sort()[1]]
        score = (weight * torch.log(discriminator.score_(disc_ctc, output)) + best).sum() / _batch_size
        return loss, score, generations.data.cpu()

    def flatten_parameters(self):
        self.u_encoder.rnn.flatten_parameters()
        self.c_encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()


class VHRED(nn.Module):
    def __init__(self, dictionary, vocab_size, dim_embedding, init_embedding, hidden_size):
        super(VHRED, self).__init__()
        self.dictionary = dictionary
        self.embedding = Embedding(vocab_size, dim_embedding, init_embedding, trainable=True).cuda()
        self.u_encoder = UtteranceEncoder(dim_embedding, hidden_size).cuda()
        self.cenc_input_size = hidden_size * 2
        self.c_encoder = ContextEncoder(self.cenc_input_size, hidden_size).cuda()
        self.decoder = HREDDecoder(dim_embedding, hidden_size * 2, hidden_size, len(dictionary)).cuda()
        self.hidden_size = hidden_size
        self.prior_enc = LatentVariableEncoder(hidden_size, hidden_size).cuda()
        self.post_enc = LatentVariableEncoder(hidden_size * 2, hidden_size).cuda()
        self.vocab_size = vocab_size

    def parameters(self):
        return list(self.u_encoder.parameters()) + list(self.c_encoder.parameters()) \
               + list(self.decoder.parameters()) + list(self.embedding.parameters()) \
               + list(self.prior_enc.parameters()) + list(self.post_enc.parameters())

    def loss(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, ctc_lengths, kl_weight=1):
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
        decoder_input = self.embedding(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>'])))
        decoder_outputs = Variable(torch.zeros(_batch_size, max_len - 1, self.vocab_size)).cuda()
        sample_prior = self.prior_enc.sample(cenc_out).cuda()
        decoder_hidden = self.decoder.init_hidden(torch.stack((cenc_out, sample_prior), dim=1))

        for t in range(1, max_len):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden
            )
            decoder_outputs[:, t - 1, :] = decoder_output
            decoder_input = self.embedding(trg_seqs[:, t])

        loss = compute_loss(decoder_outputs, Variable(trg_seqs[:, 1:]), trg_lengths - 1)

        trg_lengths, perm_idx = trg_lengths.data.sort(0, descending=True)
        trg_seqs = trg_seqs[perm_idx]
        cenc_out = cenc_out[perm_idx]
        trg_packed = pack_padded_sequence(self.embedding(Variable(trg_seqs)), trg_lengths.cpu().numpy(), batch_first=True)
        trg_encoded = self.u_encoder(trg_packed)
        post_mean, post_var = self.post_enc(trg_encoded)
        prior_mean, prior_var = self.prior_enc(cenc_out)
        kl_loss = torch.sum(torch.log(prior_var)) - torch.sum(torch.log(post_var))
        kl_loss += torch.sum((prior_mean - post_mean)**2 / prior_var) 
        kl_loss += torch.sum(post_var / prior_var)
        loss += kl_loss / (2 * _batch_size) * kl_weight

        return loss

    def semantic_loss(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, ctc_lengths):
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
        decoder_input = self.embedding(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>'])))
        decoder_outputs = Variable(torch.zeros(_batch_size, max_len - 1, self.vocab_size)).cuda()
        sample_prior = self.prior_enc.sample(cenc_out).cuda()
        decoder_hidden = self.decoder.init_hidden(torch.stack((cenc_out, sample_prior), dim=1))

        for t in range(1, max_len):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden
            )
            decoder_outputs[:, t - 1, :] = decoder_output
            decoder_input = self.embedding(trg_seqs[:, t])

        loss = compute_semantic_loss(decoder_outputs, Variable(trg_seqs[:, 1:]), trg_lengths - 1)

        return loss

    def generate(self, src_seqs, src_lengths, indices, ctc_lengths, max_len, beam_size, top_k):
        src_seqs = self.embedding(Variable(src_seqs.cuda()))
        # src_seqs: (N, max_uttr_len, word_dim)
        uenc_packed_input = pack_padded_sequence(src_seqs, src_lengths, batch_first=True)
        uenc_output = self.u_encoder(uenc_packed_input)
        # output: (N, dim1)
        _batch_size = len(ctc_lengths)
        max_ctc_len = max(ctc_lengths)
        cenc_in = Variable(torch.zeros(_batch_size, max_ctc_len, self.cenc_input_size).float()).cuda()
        for i in range(len(indices)):
            x, y = indices[i]
            cenc_in[x, y, :] = uenc_output[i]
        # cenc_in: (batch_size, max_turn, dim1)
        ctc_lengths, perm_idx = torch.cuda.LongTensor(ctc_lengths).sort(0, descending=True)
        cenc_in = cenc_in[perm_idx, :, :]
        # cenc_in: (batch_size, max_turn, dim1)
        cenc_packed_input = pack_padded_sequence(cenc_in, ctc_lengths.cpu().numpy(), batch_first=True)
        cenc_out = self.c_encoder(cenc_packed_input)
        # cenc_out: (batch_size, dim2)
        generations = torch.zeros(_batch_size, max_len).long()
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.dictionary['<end>']))
        sample_prior = self.prior_enc.sample(cenc_out).cuda()
        for x in range(_batch_size):
            decoder_hidden = self.decoder.init_hidden(torch.stack((cenc_out[x].unsqueeze(0), sample_prior[x].unsqueeze(0)), dim=1))
            decoder_input = self.embedding(Variable(torch.zeros(1).long().cuda().fill_(self.dictionary['<start>'])))
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), top_k, dim=1)
            decoder_input = self.embedding(argtop[0])
            beam = Variable(torch.zeros(beam_size, max_len)).long().cuda()
            beam[:, 0] = argtop
            beam_probs = logprobs[0].clone()
            beam_eos = (argtop == self.dictionary['<end>'])[0].data
            decoder_hidden = decoder_hidden.repeat(1, beam_size, 1)
            for t in range(max_len-1):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden
                )
                logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), top_k, dim=1)
                best_probs, best_args = (beam_probs.repeat(top_k, 1).transpose(0, 1) + logprobs).view(-1).topk(beam_size)
                beam = beam[best_args / top_k, :]
                beam_eos = beam_eos[(best_args / top_k).data]
                beam_probs = beam_probs[(best_args / top_k).data]
                beam[:, t+1] = argtop[(best_args/ top_k).data, (best_args % top_k).data] * Variable(~beam_eos).long() + \
                               eos_filler * Variable(beam_eos).long()
                beam_probs[~beam_eos] = (beam_probs[~beam_eos] * (t+1) + best_probs[~beam_eos]) / (t+2)
                decoder_hidden = decoder_hidden[:, best_args / top_k, :]
                decoder_input = self.embedding(beam[:, t+1])
                beam_eos = beam_eos | (beam[:, t+1] == self.dictionary['<end>']).data
            best, best_arg = beam_probs.max(0)
            generations[x] = beam[best_arg.data].data.cpu()
        return generations

    def flatten_parameters(self):
        self.u_encoder.rnn.flatten_parameters()
        self.c_encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
        

class AttnDecoderRNN(nn.Module):
    def __init__(self, dictionary, vocab_size, word_embedding_dim, word_vectors, hidden_size, encoder_type='rnn'):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.input_size = word_embedding_dim
        self.dropout_p = 0.0
        self.max_length = 50
        self.dictionary = dictionary

        self.embedding = Embedding(vocab_size, word_embedding_dim, word_vectors, trainable=False)
        self.attn = nn.Linear(self.hidden_size + self.input_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        if encoder_type == 'rnn':
            self.encoder = nn.GRU(self.input_size, self.hidden_size)
        elif encoder_type == 'cnn':
            self.encoder = nn.Conv1d(word_embedding_dim, word_embedding_dim, 3, padding=1)
        else:
            raise NotImplementedError
        self.decoder = nn.GRU(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.encoder_type = encoder_type

    def forward(self, src_seqs, src_lengths, trg_seqs, trg_lengths, sampling_rate):
        src_seqs = self.embedding(Variable(src_seqs.cuda()))
        max_len = max(trg_lengths)
        trg_lengths = Variable(torch.cuda.LongTensor(trg_lengths))
        # src_seqs: (N, max_uttr_len, word_dim)
        if self.encoder_type == 'rnn':
            uenc_packed_input = pack_padded_sequence(src_seqs, src_lengths, batch_first=True)
            encoder_outputs, decoder_hidden = self.encoder(uenc_packed_input)
            encoder_outputs, _ = pad_packed_sequence(encoder_outputs, batch_first=True)
            encoder_outputs = encoder_outputs[:, -self.max_length:, :]
        elif self.encoder_type == 'cnn':
            encoder_outputs = F.relu(self.encoder(src_seqs.transpose(1, 2)).transpose(1, 2))
            encoder_outputs = encoder_outputs[:, -self.max_length:, :]
            decoder_hidden = src_seqs.sum(dim=1) / Variable(torch.FloatTensor(src_lengths).unsqueeze(1).cuda())
            decoder_hidden = decoder_hidden.unsqueeze(0)
        else:
            raise NotImplementedError

        batch_size = len(src_lengths)
        #decoder_input = self.embedding(Variable(torch.zeros(batch_size).long().cuda().fill_(self.dictionary['<start>'])))
        decoder_input = self.embedding(Variable(trg_seqs[:, 0].cuda()))
        decoder_outputs = Variable(torch.zeros(batch_size, max_len - 1, self.output_size)).cuda()
        for t in range(1, max_len):
            attn_weights = F.softmax(self.attn(torch.cat((decoder_input, decoder_hidden[0]), 1)), dim=1)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)[:, 0, :]
            output = torch.cat((decoder_input, attn_applied), 1)
            output = self.attn_combine(output).unsqueeze(0)
            output = F.relu(output)
            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(0), output)
            decoder_output = self.out(decoder_output[0])
            decoder_outputs[:, t - 1, :] = decoder_output
            if np.random.uniform() > sampling_rate:
                decoder_input = self.embedding(Variable(trg_seqs[:, t].cuda()))
            else:
                decoder_input = self.embedding(decoder_output.max(1)[1])

        loss = compute_loss(decoder_outputs, Variable(trg_seqs[:, 1:].cuda()), trg_lengths - 1)

        return loss

    def generate(self, src_seqs, src_lengths, max_len, beam_size, top_k):
        src_seqs = self.embedding(Variable(src_seqs.cuda()))
        # src_seqs: (N, max_uttr_len, word_dim)
        uenc_packed_input = pack_padded_sequence(src_seqs, src_lengths, batch_first=True)
        if self.encoder_type == 'rnn':
            uenc_packed_input = pack_padded_sequence(src_seqs, src_lengths, batch_first=True)
            encoder_outputs, decoder_initial_hidden = self.encoder(uenc_packed_input)
            encoder_outputs, _ = pad_packed_sequence(encoder_outputs, batch_first=True)
            encoder_outputs = encoder_outputs[:, -self.max_length:, :]
        elif self.encoder_type == 'cnn':
            encoder_outputs = F.relu(self.encoder(src_seqs.transpose(1, 2)).transpose(1, 2))
            encoder_outputs = encoder_outputs[:, -self.max_length:, :]
            decoder_initial_hidden = src_seqs.sum(dim=1) / Variable(torch.FloatTensor(src_lengths).unsqueeze(1).cuda())
            decoder_initial_hidden = decoder_initial_hidden.unsqueeze(0)
        else:
            raise NotImplementedError

        batch_size = len(src_seqs)
        initial_input = self.embedding(Variable(torch.zeros(batch_size).long().cuda().fill_(self.dictionary['<start>'])))
        decoder_outputs = Variable(torch.zeros(batch_size, max_len - 1, self.output_size)).cuda()
        generations = torch.zeros(batch_size, max_len).long()
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.dictionary['__eou__']))
        for x in range(batch_size):
            attn_weights = F.softmax(self.attn(torch.cat((initial_input[x].unsqueeze(0), decoder_initial_hidden[0, x, :].unsqueeze(0)), 1)), dim=1)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs[x].unsqueeze(0))[:, 0, :]
            output = torch.cat((initial_input[x].unsqueeze(0), attn_applied), 1)
            output = self.attn_combine(output).unsqueeze(0)
            output = F.relu(output)
            decoder_output, decoder_hidden = self.decoder(initial_input[x].unsqueeze(0).unsqueeze(0), output)
            decoder_output = self.out(decoder_output[0])

            logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), top_k, dim=1)
            decoder_input = self.embedding(argtop[0])
            beam = Variable(torch.zeros(beam_size, max_len)).long().cuda()
            beam[:, 0] = argtop
            beam_probs = logprobs[0].clone()
            beam_eos = (argtop == self.dictionary['__eou__'])[0].data
            decoder_hidden = decoder_hidden.repeat(1, beam_size, 1)
            for t in range(max_len-1):
                attn_weights = F.softmax(self.attn(torch.cat((decoder_input, decoder_hidden[0]), 1)), dim=1)
                attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs[x].repeat(beam_size, 1, 1))[:, 0, :]
                output = torch.cat((decoder_input, attn_applied), 1)
                output = self.attn_combine(output).unsqueeze(0)
                output = F.relu(output)
                decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(0), output)
                decoder_output = self.out(decoder_output[0])
                logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), top_k, dim=1)
                best_probs, best_args = (beam_probs.repeat(top_k, 1).transpose(0, 1) + logprobs).view(-1).topk(beam_size)
                beam = beam[best_args / top_k, :]
                beam_eos = beam_eos[(best_args / top_k).data]
                beam_probs = beam_probs[(best_args / top_k).data]
                beam[:, t+1] = argtop[(best_args/ top_k).data, (best_args % top_k).data] * Variable(~beam_eos).long() + \
                               eos_filler * Variable(beam_eos).long()
                beam_probs[~beam_eos] = (beam_probs[~beam_eos] * (t+1) + best_probs[~beam_eos]) / (t+2)
                decoder_hidden = decoder_hidden[:, best_args / top_k, :]
                decoder_input = self.embedding(beam[:, t+1])
                beam_eos = beam_eos | (beam[:, t+1] == self.dictionary['__eou__']).data
            best, best_arg = beam_probs.max(0)
            generations[x] = beam[best_arg.data].data.cpu()

        return generations

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters()) \
               + list(self.attn.parameters()) + list(self.attn_combine.parameters())

    def flatten_parameters(self):
        if self.encoder_type == 'rnn':
            self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()


 
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
