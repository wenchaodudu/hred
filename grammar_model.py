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
from masked_cel import compute_loss, compute_perplexity
from gumbel_softmax import gumbel_softmax
from discriminator import LSTM, DiscUtteranceEncoder
from ast import literal_eval
from collections import defaultdict
from anytree import Node, RenderTree
from copy import deepcopy


Preterminal = literal_eval(open('preterminal.txt').readlines()[0])
CC = ['for', 'and', 'nor', 'but', 'or', 'yet', 'so']
DT = []
EX = ['there']
IN = []
MD = ['can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'must']
TO = ['to']
WDT = ['what', 'which', 'whose']
WP = ['who', 'what', 'whether', 'which']
WPP = ['whose']
WRB = ['how', 'when', 'whence', 'where', 'why']
PRP = ['you', 'i', 'he', 'she', 'it', 'we', 'they']
PRPP = ['yours', 'mine', 'his', 'hers', 'its', 'ours', 'theirs']

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
            self.weight.data.uniform_(-0.1, 0.1)
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


class AttnDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, word_vectors, dictionary):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.embed = Embedding(vocab_size, input_size, word_vectors, trainable=True)
        self.encoder = nn.LSTM(input_size, hidden_size)
        self.decoder = nn.LSTM(input_size + hidden_size, hidden_size, batch_first=True)
        self.key_size = 100
        self.q_key = nn.Linear(hidden_size, self.key_size)
        self.q_value = nn.Linear(hidden_size, hidden_size)
        self.a_key = nn.Linear(hidden_size, self.key_size)
        self.max_len = 21
        self.out = nn.Linear(hidden_size * 2, input_size)
        self.word_dist = nn.Linear(input_size, vocab_size)
        self.context_fc1 = nn.Linear(hidden_size * 2, hidden_size // 2 * 3)
        self.context_fc2 = nn.Linear(hidden_size // 2 * 3, hidden_size)
        self.hidden_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.hidden_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.cell_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.discriminator = None
        self.dictionary = dictionary
        self.eou = dictionary['__eou__']
        self.word_dist.weight = self.embed.weight

        for names in self.encoder._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.encoder, name)
                n = bias.size(0)
                start, end = n//4, n//2 
                bias.data[start:end].fill_(0.)

        for names in self.decoder._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.decoder, name)
                n = bias.size(0)
                start, end = n//4, n//2 
                bias.data[start:end].fill_(0.)

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()

    def context_transform(self, context):
        return self.context_fc2(F.relu(self.context_fc1(context)))

    def hidden_transform(self, hidden):
        return self.hidden_fc2(F.relu(self.hidden_fc1(hidden)))

    def cell_transform(self, cell):
        return self.cell_fc2(F.relu(self.cell_fc1(cell)))

    def init_hidden(self, src_hidden):
        hidden = src_hidden[0]
        cell = src_hidden[1]
        hidden = self.hidden_transform(hidden)
        cell = self.cell_transform(cell)
        return (hidden, cell)
        '''
        return src_hidden
        '''

    def forward(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, sampling_rate):
        batch_size = src_seqs.size(0)

        src_embed = self.embed(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 

        length = src_hidden.size(1)
        ans_embed = self.embed(Variable(trg_seqs).cuda())
        trg_l = ans_embed.size(1)

        decoder_input = ans_embed[:, 0, :].unsqueeze(1)
        decoder_outputs = Variable(torch.FloatTensor(batch_size, trg_l - 1, self.vocab_size).cuda())
        for step in range(trg_l - 1):
            a_key = F.tanh(self.a_key(decoder_hidden[0].squeeze(0)))

            q_key = F.tanh(self.q_key(src_hidden))
            q_value = self.q_value(src_hidden)
            q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
            q_mask  = torch.arange(length).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(length, 1).transpose(0, 1)
            q_energy[~q_mask] = -np.inf
            q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
            q_context = torch.bmm(q_weights, q_value).squeeze(1)

            #context = torch.cat((q_context, i_context), dim=1)
            context = q_context.unsqueeze(1)
            decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2), decoder_hidden)
            decoder_outputs[:, step, :] = self.word_dist(self.out(torch.cat((decoder_output.squeeze(1), context.squeeze(1)), dim=1)))
            #decoder_outputs[:, step, :] = decoder_output.squeeze(1)
            if np.random.uniform() < sampling_rate and step < self.max_len - 2:
                decoder_input = ans_embed[:, step+1, :].unsqueeze(1)
            else:
                words = decoder_outputs[:, step, :].max(dim=1)[1]
                decoder_input = self.embed(words).unsqueeze(1)

        return decoder_outputs

    def generate(self, src_seqs, src_lengths, indices, max_len, beam_size, top_k):
        src_embed = self.embed(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 

        cenc_out = src_last_hidden
        _batch_size = src_embed.size(0)
        # cenc_out: (batch_size, dim2)
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        decoder_input = self.embed(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>']))).unsqueeze(1)
        length = src_hidden.size(1)

        decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2), decoder_hidden)
        decoder_output = self.word_dist(self.out(torch.cat((decoder_output.squeeze(1), context.squeeze(1)), dim=1)))
        # only non-terminal at the beginning
        logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), beam_size, dim=1)
        #decoder_input = self.embedding(argtop[0])
        beam = Variable(torch.zeros(_batch_size, beam_size, max_len)).long().cuda()
        beam[:, :, 0] = argtop
        #beam_probs = logprobs[0].clone()
        beam_probs = logprobs.clone()
        #beam_eos = (argtop == self.eou)[0].data
        beam_eos = (argtop == self.eou).data
        hidden = (decoder_hidden[0].clone(), decoder_hidden[1].clone())
        decoder_hidden = (decoder_hidden[0].unsqueeze(2).expand(1, _batch_size, beam_size, self.hidden_size).contiguous().view(1, _batch_size * beam_size, -1),
                          decoder_hidden[1].unsqueeze(2).expand(1, _batch_size, beam_size, self.hidden_size).contiguous().view(1, _batch_size * beam_size, -1))
        decoder_input = self.embed(argtop.view(-1)).unsqueeze(1)
        for t in range(max_len-1):
            decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2), decoder_hidden)
            decoder_output = self.word_dist(self.out(torch.cat((decoder_output.squeeze(1), context.squeeze(1)), dim=1)))

            logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), top_k, dim=1)
            #best_probs, best_args = (beam_probs.repeat(top_k, 1).transpose(0, 1) + logprobs).view(-1).topk(beam_size)
            best_probs, best_args = (beam_probs.view(-1).unsqueeze(1).expand(_batch_size * beam_size, top_k) + logprobs).view(_batch_size, beam_size, -1).view(_batch_size, -1).topk(beam_size)
            decoder_hidden = (decoder_hidden[0].view(1, _batch_size, beam_size, -1),
                              decoder_hidden[1].view(1, _batch_size, beam_size, -1))
            for x in range(_batch_size):
                last = (best_args / top_k)[x]
                curr = (best_args % top_k)[x]
                beam[x, :, :] = beam[x][last, :]
                beam_eos[x, :] = beam_eos[x][last.data]
                beam_probs[x, :] = beam_probs[x][last.data]
                beam[x, :, t+1] = argtop.view(_batch_size, beam_size, top_k)[x][last.data, curr.data] * Variable(~beam_eos[x]).long() + eos_filler * Variable(beam_eos[x]).long()
                mask = torch.cuda.ByteTensor(_batch_size, beam_size).fill_(0)
                mask[x] = ~beam_eos[x]
                beam_probs[mask] = (beam_probs[mask] * (t+1) + best_probs[mask]) / (t+2)
                decoder_hidden[0][:, x, :, :] = decoder_hidden[0][:, x, :, :][:, last, :]
                decoder_hidden[1][:, x, :, :] = decoder_hidden[1][:, x, :, :][:, last, :]
            beam_eos = beam_eos | (beam[:, :, t+1] == self.eou).data
            decoder_hidden = (decoder_hidden[0].view(1, _batch_size * beam_size, -1),
                              decoder_hidden[1].view(1, _batch_size * beam_size, -1))
            decoder_input = self.embed(beam[:, :, t+1].contiguous().view(-1)).unsqueeze(1)
        best, best_arg = beam_probs.max(1)
        generations = beam[torch.arange(_batch_size).long().cuda(), best_arg.data].data.cpu()
        return generations, best

    def loss(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, sampling_rate):
        decoder_outputs = self.forward(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, sampling_rate)
        loss = compute_loss(decoder_outputs, Variable(trg_seqs[:, 1:].cuda()), Variable(torch.cuda.LongTensor(trg_lengths)) - 1)

        return loss

    def evaluate(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths):
        decoder_outputs = self.forward(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, 1)
        loss = compute_perplexity(decoder_outputs, Variable(trg_seqs[:, 1:].cuda()), Variable(torch.cuda.LongTensor(trg_lengths)) - 1)

        return loss


class GrammarDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, word_vectors, dictionary):
        super(GrammarDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.embed = Embedding(vocab_size, input_size, word_vectors, trainable=True)
        self.encoder = nn.LSTM(input_size, hidden_size)
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, input_size)
        self.word_dist = nn.Linear(input_size, vocab_size)
        self.hidden_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.hidden_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.cell_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.discriminator = None
        self.dictionary = dictionary
        self.eou = dictionary['__eou__']
        self.REDUCE = dictionary['REDUCE']
        self.word_dist.weight = self.embed.weight

        self.init_forget_bias(self.encoder, 0)
        self.init_forget_bias(self.decoder, 10)

        nonterminal = [False for x in range(vocab_size)]
        for k, v in dictionary.items():
            if k[:3] == 'NT-':
                nonterminal[v] = True
        self.nonterminal = torch.cuda.ByteTensor(nonterminal)

        preterminal = [False for x in range(vocab_size)]
        for k, v in dictionary.items():
            if k[3:] in Preterminal:
                preterminal[v] = True
        self.preterminal = torch.cuda.ByteTensor(preterminal)

        self.nonpreterminal = self.nonterminal & ~self.preterminal

        slots = [False for x in range(vocab_size)]
        for k, v in dictionary.items():
            if len(k) > 7 and k[3:7] == 'SLOT':
                slots[v] = True
        self.slots = torch.cuda.ByteTensor(slots)

    def init_rules(self):
        rules = [False for x in range(self.vocab_size)]
        rules_by_head = defaultdict(list)
        self.rules_by_id = defaultdict()
        for k, v in self.dictionary.items():
            if k[:4] == 'RULE':
                rules[v] = True
                nt = k.split()
                rules_by_head[nt[1]].append(v)
                self.rules_by_id[v] = k
        for k, v in rules_by_head.items():
            t = torch.zeros(self.vocab_size).cuda().byte()
            t[v] = True
            rules_by_head[k] = t
        self.rules = torch.cuda.ByteTensor(rules)
        self.rules_by_head = rules_by_head

        self.preterminal = defaultdict(bool)
        for pt in Preterminal:
            self.preterminal['NT-{}'.format(pt)] = True

    def init_forget_bias(self, rnn, b):
        for names in rnn._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2 
                bias.data[start:end].fill_(b)

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()

    def hidden_transform(self, hidden):
        return self.hidden_fc2(F.relu(self.hidden_fc1(hidden)))

    def cell_transform(self, cell):
        return self.cell_fc2(F.relu(self.cell_fc1(cell)))

    def init_hidden(self, src_hidden):
        hidden = src_hidden[0]
        cell = src_hidden[1]
        hidden = self.hidden_transform(hidden)
        cell = self.cell_transform(cell)
        return (hidden, cell)

    def forward(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, sampling_rate):
        batch_size = src_seqs.size(0)

        src_embed = self.embed(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 

        length = src_hidden.size(1)
        ans_embed = self.embed(Variable(trg_seqs).cuda())
        trg_l = ans_embed.size(1)

        decoder_input = ans_embed[:, 0, :].unsqueeze(1)
        decoder_outputs = Variable(torch.FloatTensor(batch_size, trg_l - 1, self.vocab_size).cuda())
        for step in range(trg_l - 1):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs[:, step, :] = self.word_dist(self.out(decoder_output.squeeze(1)))
            if np.random.uniform() < sampling_rate and step < trg_l - 2:
                decoder_input = ans_embed[:, step+1, :].unsqueeze(1)
            else:
                words = decoder_outputs[:, step, :].max(dim=1)[1]
                decoder_input = self.embed(words).unsqueeze(1)

        return decoder_outputs

    def generate(self, src_seqs, src_lengths, indices, max_len, beam_size, top_k, grammar=True):
        src_embed = self.embed(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 

        _batch_size = src_embed.size(0)
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        decoder_input = self.embed(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>']))).unsqueeze(1)
        length = src_hidden.size(1)

        nonpreterminal = self.nonterminal & ~self.preterminal
        START = self.dictionary['NT-S']
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        decoder_output = self.word_dist(self.out(decoder_output))

        if grammar:
            decoder_output[~nonpreterminal.expand(_batch_size, 1, self.vocab_size).cuda()] = -np.inf
        else:
            decoder_output[self.nonterminal.expand(_batch_size, 1, self.vocab_size).cuda()] = -np.inf
        logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), beam_size, dim=1)
        beam = Variable(torch.zeros(_batch_size, beam_size, max_len)).long().cuda()
        beam[:, :, 0] = argtop
        beam_probs = logprobs.clone()
        beam_eos = (argtop == self.eou).data
        hidden = (decoder_hidden[0].clone(), decoder_hidden[1].clone())
        decoder_hidden = (decoder_hidden[0].unsqueeze(2).expand(1, _batch_size, beam_size, self.hidden_size).contiguous().view(1, _batch_size * beam_size, -1),
                          decoder_hidden[1].unsqueeze(2).expand(1, _batch_size, beam_size, self.hidden_size).contiguous().view(1, _batch_size * beam_size, -1))
        decoder_input = self.embed(argtop.view(-1)).unsqueeze(1)
        stacks = torch.zeros(_batch_size, beam_size).long().cuda() + 1
        reduce_mask = torch.zeros(_batch_size * beam_size, 1, self.vocab_size).byte().cuda()
        eou_mask = torch.zeros(_batch_size * beam_size, 1, self.vocab_size).byte().cuda()
        reduce_mask[:, :, self.REDUCE] = 1
        eou_mask[:, :, self.eou] = 1
        eou_mask[:, :, START] = 1
        for t in range(max_len-1):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = self.word_dist(self.out(decoder_output))

            if grammar:
                r_mask = (stacks == 0).view(-1).expand(self.vocab_size, 1, _batch_size * beam_size).transpose(0, 2) * ~nonpreterminal
                r_mask[:, 0, self.REDUCE] |= self.nonterminal[beam[0, :, t].data]
                e_mask = (stacks > 0).view(-1).expand(self.vocab_size, 1, _batch_size * beam_size).transpose(0, 2) * eou_mask
                pt_mask = self.preterminal[beam[:, :, t].contiguous().view(-1).data].expand(self.vocab_size, 1, _batch_size * beam_size).transpose(0, 2) * (self.nonterminal + reduce_mask[0, 0, :])
                nt_mask = nonpreterminal[beam[:, :, t].contiguous().view(-1).data].expand(self.vocab_size, 1, _batch_size * beam_size).transpose(0, 2) * ~self.nonterminal
                t_mask = ~self.nonterminal[beam[:, :, t].contiguous().view(-1).data].expand(self.vocab_size, 1, _batch_size * beam_size).transpose(0, 2) * ~self.nonterminal
                t_mask[:, :, self.REDUCE] = 0
                decoder_output[r_mask] = -np.inf
                decoder_output[e_mask] = -np.inf
                decoder_output[pt_mask] = -np.inf
                decoder_output[nt_mask] = -np.inf
                decoder_output[t_mask] = -np.inf
            else:
                decoder_output[self.nonterminal.expand(_batch_size * beam_size, 1, self.vocab_size)] = -np.inf
                decoder_output[:, :, self.REDUCE] = -np.inf

            logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), top_k, dim=1)
            best_probs, best_args = (beam_probs.view(-1).unsqueeze(1).expand(_batch_size * beam_size, top_k) + logprobs).view(_batch_size, beam_size, -1).view(_batch_size, -1).topk(beam_size)
            decoder_hidden = (decoder_hidden[0].view(1, _batch_size, beam_size, -1),
                              decoder_hidden[1].view(1, _batch_size, beam_size, -1))
            for x in range(_batch_size):
                last = (best_args / top_k)[x]
                curr = (best_args % top_k)[x]
                beam[x, :, :] = beam[x][last, :]
                beam_eos[x, :] = beam_eos[x][last.data]
                beam_probs[x, :] = beam_probs[x][last.data]
                stacks[x, :] = stacks[x][last.data]
                beam[x, :, t+1] = argtop.view(_batch_size, beam_size, top_k)[x][last.data, curr.data] * Variable(~beam_eos[x]).long() + eos_filler * Variable(beam_eos[x]).long()
                '''
                ind1 = torch.cuda.LongTensor(beam_size).fill_(x)
                ind2 = (~beam_eos[x]).long()
                beam_probs[ind1, ind2] = (beam_probs[ind1, ind2] * (t+1) + best_probs[ind1, ind2]) / (t+2)
                '''
                mask = torch.cuda.ByteTensor(_batch_size, beam_size).fill_(0)
                mask[x] = ~beam_eos[x]
                beam_probs[mask] = (beam_probs[mask] * (t+1) + best_probs[mask]) / (t+2)
                decoder_hidden[0][:, x, :, :] = decoder_hidden[0][:, x, :, :][:, last, :]
                decoder_hidden[1][:, x, :, :] = decoder_hidden[1][:, x, :, :][:, last, :]
            beam_eos = beam_eos | (beam[:, :, t+1] == self.eou).data
            decoder_hidden = (decoder_hidden[0].view(1, _batch_size * beam_size, -1),
                              decoder_hidden[1].view(1, _batch_size * beam_size, -1))
            decoder_input = self.embed(beam[:, :, t+1].contiguous().view(-1)).unsqueeze(1)
            if grammar:
                #stacks[beam[:, :, t+1].data == self.REDUCE] -= 1
                nonterminal = self.nonterminal.cuda()[beam[:, :, t+1].data.contiguous().view(-1)].view(_batch_size, beam_size)
                stacks[nonterminal] += 1
                stacks[~nonterminal] -= 1
        best, best_arg = beam_probs.max(1)
        generations = beam[torch.arange(_batch_size).long().cuda(), best_arg.data].data.cpu()
        return generations, best

    def fsm_all_generate(self, src_seqs, src_lengths, indices, slots, max_len, beam_size, top_k):
        src_embed = self.embed(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 

        _batch_size = src_embed.size(0)
        assert _batch_size == 1
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        decoder_input = self.embed(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>']))).unsqueeze(1)
        length = src_hidden.size(1)

        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        decoder_output = self.word_dist(self.out(decoder_output))

        slots = slots.cuda()
        non_slots = self.slots.clone()
        non_slots[slots] = 0
        states_num = 2 ** len(slots)
        slots_num = slots.size(0)
        decoder_output[~self.nonterminal.expand(_batch_size, 1, self.vocab_size).cuda()] = -np.inf
        decoder_output[non_slots.expand(_batch_size, 1, self.vocab_size)] = -np.inf
        slots_logprobs = F.log_softmax(decoder_output.squeeze(1), dim=1)[:, slots]
        decoder_output[:, :, slots] = -np.inf
        logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), beam_size, dim=1)
        argtop = Variable(torch.cat((argtop.data, slots.expand(_batch_size, slots_num)), dim=1)).squeeze(0)
        logprobs = Variable(torch.cat((logprobs.data, slots_logprobs.data), dim=1)).squeeze(0)
        state_codes = np.zeros(beam_size + slots_num, dtype=np.int32)
        logprobs, perm_idx = logprobs.sort(descending=True)
        argtop = argtop[perm_idx]
        state_codes = state_codes[perm_idx.cpu().data.numpy()]
        for x in range(slots_num):
            state_codes[beam_size+x] = 2 ** x
        beam = Variable(torch.zeros(states_num, beam_size, max_len)).long().cuda()
        beam_probs = torch.zeros(states_num, beam_size).cuda().fill_(-np.inf)
        beam_eos = torch.zeros(states_num, beam_size).byte().cuda()
        for x in range(states_num):
            if (state_codes == x).any():
                state_going = torch.from_numpy(np.where(state_codes == x)[0]).cuda()
                state_going_num = min(beam_size, state_going.shape[0])
                beam[x, :state_going_num, 0] = argtop[state_going[:state_going_num]]
                beam_probs[x, :state_going_num] = logprobs[state_going[:state_going_num]].data
                beam_eos[x, :state_going_num] = (argtop[state_going[:state_going_num]] == self.eou).data
        decoder_hidden = (decoder_hidden[0].unsqueeze(2).expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1),
                          decoder_hidden[1].unsqueeze(2).expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1))
        decoder_input = self.embed(beam[:, :, 0].contiguous().view(-1)).unsqueeze(1)

        stacks = torch.zeros(states_num, beam_size).long().cuda() + 1
        nonterminal = self.nonterminal.cuda()[beam[:, :, 0].data.contiguous().view(-1)].view(states_num, beam_size)
        stacks[nonterminal] += 1
        reduce_mask = torch.zeros(states_num * beam_size, 1, self.vocab_size).byte().cuda()
        eou_mask = torch.zeros(states_num * beam_size, 1, self.vocab_size).byte().cuda()
        reduce_mask[:, :, self.REDUCE] = 1
        eou_mask[:, :, self.eou] = 1

        state_codes = np.zeros(beam_size * states_num, dtype=np.int32)
        trans_state_codes = np.zeros(top_k + slots_num, dtype=np.int32)
        for x in range(slots_num):
            trans_state_codes[top_k+x] = 2 ** x 
        for x in range(states_num):
            state_codes[x*beam_size:(x+1)*beam_size] = x

        for t in range(max_len-1):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = self.word_dist(self.out(decoder_output))
            decoder_output[non_slots.expand(states_num * beam_size, 1, self.vocab_size)] = -np.inf

            r_mask = (stacks == 0).view(-1).expand(self.vocab_size, 1, states_num * beam_size).transpose(0, 2) * reduce_mask
            e_mask = (stacks > 0).view(-1).expand(self.vocab_size, 1, states_num * beam_size).transpose(0, 2) * eou_mask
            slot_mask = self.slots[beam[:, :, t].contiguous().view(-1).data].expand(self.vocab_size, 1, states_num * beam_size).transpose(0, 2) * ~reduce_mask
            '''
            decoder_output[r_mask] = -np.inf
            decoder_output[e_mask] = -np.inf
            decoder_output[slot_mask] = -np.inf
            slots_logprobs = F.log_softmax(decoder_output.squeeze(1), dim=1)[:, slots]
            decoder_output[:, :, slots] = -np.inf
            '''
            pre_logprobs = F.log_softmax(decoder_output.squeeze(1), dim=1)
            slots_logprobs = pre_logprobs[:, slots]
            pre_logprobs[r_mask] = -np.inf
            pre_logprobs[e_mask] = -np.inf
            pre_logprobs[slot_mask] = -np.inf

            logprobs, argtop = torch.topk(F.log_softmax(pre_logprobs.squeeze(1), dim=1), top_k, dim=1)
            argtop = Variable(torch.cat((argtop.data, slots.expand(states_num * beam_size, slots_num)), dim=1)).squeeze(0)
            logprobs = Variable(torch.cat((logprobs.data, slots_logprobs.data), dim=1)).squeeze(0)
            curr_logprobs = (beam_probs.view(-1).unsqueeze(1).expand(states_num * beam_size, top_k + slots_num) + logprobs.data).view(-1)
            transition = np.bitwise_or(np.repeat(state_codes, top_k + slots_num), np.tile(trans_state_codes, states_num * beam_size))
            transition = torch.cuda.LongTensor(transition)
            for x in range(states_num):
                if (transition == x).any():
                    _logprobs = curr_logprobs.clone()
                    _logprobs[transition != x] = -np.inf
                    best_probs, best_args = _logprobs.topk(beam_size)
                    last = (best_args / (top_k + slots_num))
                    curr = (best_args % (top_k + slots_num))
                    beam[x, :, :] = beam.view(-1, max_len)[last, :]
                    beam_eos[x, :] = beam_eos.view(-1)[last]
                    beam_probs[x, :] = beam_probs.view(-1)[last]
                    stacks[x, :] = stacks.view(-1)[last]
                    beam[x, :, t+1] = argtop[last, curr] * Variable(~beam_eos[x]).long() + eos_filler * Variable(beam_eos[x]).long()
                    mask = torch.cuda.ByteTensor(states_num, beam_size).fill_(0)
                    mask[x] = ~beam_eos[x]
                    beam_probs[mask] = (beam_probs[mask] * (t+1) + best_probs[~beam_eos[x]]) / (t+2)
                    decoder_hidden[0][:, x*beam_size:(x+1)*beam_size, :] = decoder_hidden[0][:, last, :]
                    decoder_hidden[1][:, x*beam_size:(x+1)*beam_size, :] = decoder_hidden[1][:, last, :]
            beam_eos = beam_eos | (beam[:, :, t+1] == self.eou).data
            decoder_input = self.embed(beam[:, :, t+1].contiguous().view(-1)).unsqueeze(1)
            stacks[beam[:, :, t+1].data == self.REDUCE] -= 1
            nonterminal = self.nonterminal.cuda()[beam[:, :, t+1].data.contiguous().view(-1)].view(states_num, beam_size)
            stacks[nonterminal] += 1
        best, best_arg = beam_probs[-1].max(0)
        generations = beam[-1][best_arg].data.cpu()
        return generations, best

    def fsm_one_generate(self, src_seqs, src_lengths, indices, slots, max_len, beam_size, top_k, grammar=True):
        src_embed = self.embed(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 

        _batch_size = src_embed.size(0)
        assert _batch_size == 1
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        decoder_input = self.embed(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>']))).unsqueeze(1)
        length = src_hidden.size(1)

        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        decoder_output = self.word_dist(self.out(decoder_output))

        nonpreterminal = self.nonterminal & ~self.preterminal
        START = self.dictionary['NT-S']
        non_slots = self.slots.clone()
        non_slots[slots] = 0
        if grammar:
            states_num = 3
            slots.append(self.dictionary['NT-NN'])
        else:
            states_num = 2
        slots_num = len(slots)
        slots = torch.cuda.LongTensor(slots)
        slots_logprobs = F.log_softmax(decoder_output.squeeze(1), dim=1).squeeze(0)
        decoder_output[:, :, slots] = -np.inf
        if grammar:
            decoder_output[~nonpreterminal.expand(_batch_size, 1, self.vocab_size).cuda()] = -np.inf
            slots_logprobs[~nonpreterminal] = -np.inf
        else:
            decoder_output[self.nonterminal.expand(_batch_size, 1, self.vocab_size).cuda()] = -np.inf
            slots_logprobs[self.nonterminal] = -np.inf
        slots_logprobs = slots_logprobs[slots]

        logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), beam_size, dim=1)
        argtop = argtop.squeeze(0)
        logprobs = logprobs.squeeze(0)
        argtop = Variable(torch.cat((argtop.data, slots)))
        logprobs = Variable(torch.cat((logprobs.data, slots_logprobs.data)))

        transition = np.zeros((states_num, slots_num + top_k), dtype=np.int32)
        state_codes = np.zeros(beam_size + slots_num, dtype=np.int32)
        if grammar:
            transition[0, -1] = 1
            transition[1, :top_k] = 0
            transition[1, top_k:top_k+slots_num-1] = 2
            transition[2, :] = 2
            state_codes[-1] = 1
        else:
            for x in range(slots_num):
                transition[0, top_k + x] = 1
            transition[1, :] = 1
            state_codes[beam_size:] = 1
        transition = np.repeat(transition[:, np.newaxis, :], beam_size, axis=1)
        transition = torch.cuda.LongTensor(transition).view(-1)

        beam = Variable(torch.zeros(states_num, beam_size, max_len)).long().cuda()
        beam_probs = torch.zeros(states_num, beam_size).cuda().fill_(-np.inf)
        beam_eos = torch.zeros(states_num, beam_size).byte().cuda()
        for x in range(states_num):
            if (state_codes == x).any():
                state_going = torch.from_numpy(np.where(state_codes == x)[0]).cuda()
                state_going_num = min(beam_size, state_going.shape[0])
                beam[x, :state_going_num, 0] = argtop[state_going[:state_going_num]]
                beam_probs[x, :state_going_num] = logprobs[state_going[:state_going_num]].data
                beam_eos[x, :state_going_num] = (argtop[state_going[:state_going_num]] == self.eou).data
        decoder_hidden = (decoder_hidden[0].unsqueeze(2).expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1),
                          decoder_hidden[1].unsqueeze(2).expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1))
        decoder_input = self.embed(beam[:, :, 0].contiguous().view(-1)).unsqueeze(1)

        stacks = torch.zeros(states_num, beam_size).long().cuda() + 1
        nonterminal = self.nonterminal.cuda()[beam[:, :, 0].data.contiguous().view(-1)].view(states_num, beam_size)
        stacks[nonterminal] += 1
        reduce_mask = torch.zeros(states_num * beam_size, 1, self.vocab_size).byte().cuda()
        eou_mask = torch.zeros(states_num * beam_size, 1, self.vocab_size).byte().cuda()
        reduce_mask[:, :, self.REDUCE] = 1
        eou_mask[:, :, self.eou] = 1
        eou_mask[:, :, START] = 1

        for t in range(max_len-1):
            new_beam = beam.clone()
            new_beam_probs = beam_probs.clone()
            new_beam_eos = beam_eos.clone()
            new_stacks = stacks.clone()

            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = self.word_dist(self.out(decoder_output))
            decoder_output[non_slots.expand(states_num * beam_size, 1, self.vocab_size)] = -np.inf
            slots_logprobs = F.log_softmax(decoder_output, dim=2)
            decoder_output[:, :, slots] = -np.inf
            if grammar:
                r_mask = (stacks == 0).view(-1).expand(self.vocab_size, 1, states_num * beam_size).transpose(0, 2) * ~nonpreterminal
                e_mask = (stacks > 0).view(-1).expand(self.vocab_size, 1, states_num * beam_size).transpose(0, 2) * eou_mask
                #slot_mask = self.slots[beam[:, :, t].contiguous().view(-1).data].expand(self.vocab_size, 1, states_num * beam_size).transpose(0, 2) * ~reduce_mask
                pt_mask = self.preterminal[beam[:, :, t].contiguous().view(-1).data].expand(self.vocab_size, 1, states_num * beam_size).transpose(0, 2) * (self.nonterminal + reduce_mask[0, 0, :])
                nt_mask = nonpreterminal[beam[:, :, t].contiguous().view(-1).data].expand(self.vocab_size, 1, states_num * beam_size).transpose(0, 2) * ~self.nonterminal
                t_mask = ~self.nonterminal[beam[:, :, t].contiguous().view(-1).data].expand(self.vocab_size, 1, states_num * beam_size).transpose(0, 2) * ~self.nonterminal
                t_mask[:, :, self.REDUCE] = 0
                decoder_output[r_mask] = -np.inf
                decoder_output[e_mask] = -np.inf
                #decoder_output[slot_mask] = -np.inf
                decoder_output[pt_mask] = -np.inf
                decoder_output[nt_mask] = -np.inf
                decoder_output[t_mask] = -np.inf
                slots_logprobs[r_mask] = -np.inf
                slots_logprobs[e_mask] = -np.inf
                #slots_logprobs[slot_mask] = -np.inf
                slots_logprobs[pt_mask] = -np.inf
                slots_logprobs[nt_mask] = -np.inf
                slots_logprobs[t_mask] = -np.inf
            else:
                decoder_output[self.nonterminal.expand(states_num * beam_size, 1, self.vocab_size)] = -np.inf
                decoder_output[:, :, self.REDUCE] = -np.inf
            slots_logprobs = slots_logprobs.squeeze(1)[:, slots]

            #logprobs, argtop = torch.topk(F.log_softmax(pre_logprobs.squeeze(1), dim=1), top_k, dim=1)
            logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), top_k, dim=1)
            argtop = Variable(torch.cat((argtop.data, slots.expand(states_num * beam_size, slots_num)), dim=1)).squeeze(0)
            logprobs = Variable(torch.cat((logprobs.data, slots_logprobs.data), dim=1)).squeeze(0)
            curr_logprobs = (beam_probs.view(-1).unsqueeze(1).expand(states_num * beam_size, top_k + slots_num) + logprobs.data).view(-1)
            for x in range(states_num):
                _logprobs = curr_logprobs.clone()
                _logprobs[transition != x] = -np.inf
                if (_logprobs != -np.inf).any():
                    best_probs, best_args = _logprobs.topk(beam_size)
                    last = (best_args / (top_k + slots_num))
                    curr = (best_args % (top_k + slots_num))
                    new_beam[x, :, :] = beam.view(-1, max_len)[last, :]
                    new_beam_eos[x, :] = beam_eos.view(-1)[last]
                    new_beam_probs[x, :] = beam_probs.view(-1)[last]
                    new_stacks[x, :] = stacks.view(-1)[last]
                    new_beam[x, :, t+1] = argtop[last, curr] * Variable(~new_beam_eos[x]).long() + eos_filler * Variable(new_beam_eos[x]).long()
                    mask = torch.cuda.ByteTensor(states_num, beam_size).fill_(0)
                    mask[x] = ~new_beam_eos[x]
                    new_beam_probs[mask] = (new_beam_probs[mask] * (t+1) + best_probs[~new_beam_eos[x]]) / (t+2)
                    decoder_hidden[0][:, x*beam_size:(x+1)*beam_size, :] = decoder_hidden[0][:, last, :]
                    decoder_hidden[1][:, x*beam_size:(x+1)*beam_size, :] = decoder_hidden[1][:, last, :]
            beam_eos = new_beam_eos | (new_beam[:, :, t+1] == self.eou).data
            decoder_input = self.embed(new_beam[:, :, t+1].contiguous().view(-1)).unsqueeze(1)
            if grammar:
                #stacks[beam[:, :, t+1].data == self.REDUCE] -= 1
                nonterminal = self.nonterminal.cuda()[new_beam[:, :, t+1].data.contiguous().view(-1)].view(states_num, beam_size)
                new_stacks[nonterminal] += 1
                new_stacks[~nonterminal] -= 1
            beam = new_beam
            beam_probs = new_beam_probs
            stacks = new_stacks
        best, best_arg = beam_probs[-1].max(0)
        generations = beam[-1][best_arg].data.cpu()
        return generations, best

    def grid_generate(self, src_seqs, src_lengths, indices, slots, max_len, beam_size, top_k):
        src_embed = self.embed(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 

        _batch_size = src_embed.size(0)
        assert _batch_size == 1
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        decoder_input = self.embed(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>']))).unsqueeze(1)
        length = src_hidden.size(1)

        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        decoder_output = self.word_dist(self.out(decoder_output))

        slots = slots.cuda()
        non_slots = self.slots.clone()
        non_slots[slots] = 0
        slots_num = slots.size(0)
        states_num = slots_num + 1
        decoder_output[~self.nonterminal.expand(_batch_size, 1, self.vocab_size).cuda()] = -np.inf
        decoder_output[non_slots.expand(_batch_size, 1, self.vocab_size)] = -np.inf
        slots_logprobs = F.log_softmax(decoder_output.squeeze(1), dim=1)[:, slots]
        decoder_output[:, :, slots] = -np.inf
        logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), beam_size, dim=1)
        argtop = Variable(torch.cat((argtop.data, slots.expand(_batch_size, slots_num)), dim=1)).squeeze(0)
        logprobs = Variable(torch.cat((logprobs.data, slots_logprobs.data), dim=1)).squeeze(0)
        state_codes = np.zeros(beam_size + slots_num, dtype=np.int32)
        logprobs, perm_idx = logprobs.sort(descending=True)
        argtop = argtop[perm_idx]
        state_codes = state_codes[perm_idx.cpu().data.numpy()]
        for x in range(slots_num):
            state_codes[beam_size+x] = 1
        beam = Variable(torch.zeros(states_num, beam_size, max_len)).long().cuda()
        beam_probs = torch.zeros(states_num, beam_size).cuda().fill_(-np.inf)
        beam_eos = torch.zeros(states_num, beam_size).byte().cuda()
        for x in range(states_num):
            if (state_codes == x).any():
                state_going = torch.from_numpy(np.where(state_codes == x)[0]).cuda()
                state_going_num = min(beam_size, state_going.shape[0])
                beam[x, :state_going_num, 0] = argtop[state_going[:state_going_num]]
                beam_probs[x, :state_going_num] = logprobs[state_going[:state_going_num]].data
                beam_eos[x, :state_going_num] = (argtop[state_going[:state_going_num]] == self.eou).data
        decoder_hidden = (decoder_hidden[0].unsqueeze(2).expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1),
                          decoder_hidden[1].unsqueeze(2).expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1))
        decoder_input = self.embed(beam[:, :, 0].contiguous().view(-1)).unsqueeze(1)

        stacks = torch.zeros(states_num, beam_size).long().cuda() + 1
        nonterminal = self.nonterminal.cuda()[beam[:, :, 0].data.contiguous().view(-1)].view(states_num, beam_size)
        stacks[nonterminal] += 1
        reduce_mask = torch.zeros(states_num * beam_size, 1, self.vocab_size).byte().cuda()
        eou_mask = torch.zeros(states_num * beam_size, 1, self.vocab_size).byte().cuda()
        reduce_mask[:, :, self.REDUCE] = 1
        eou_mask[:, :, self.eou] = 1

        state_codes = np.zeros((states_num, beam_size, slots_num), dtype=np.int32)
        trans_state_codes = np.zeros((states_num, beam_size, top_k + slots_num, slots_num), dtype=np.int32)
        for x in range(1, states_num):
            state_codes[x, :, x-1] = 1
            trans_state_codes[:, :, top_k+x-1, x-1] = 1

        for t in range(max_len-1):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = self.word_dist(self.out(decoder_output))
            decoder_output[non_slots.expand(states_num * beam_size, 1, self.vocab_size)] = -np.inf

            r_mask = (stacks == 0).view(-1).expand(self.vocab_size, 1, states_num * beam_size).transpose(0, 2) * reduce_mask
            e_mask = (stacks > 0).view(-1).expand(self.vocab_size, 1, states_num * beam_size).transpose(0, 2) * eou_mask
            slot_mask = self.slots[beam[:, :, t].contiguous().view(-1).data].expand(self.vocab_size, 1, states_num * beam_size).transpose(0, 2) * ~reduce_mask
            '''
            decoder_output[r_mask] = -np.inf
            decoder_output[e_mask] = -np.inf
            decoder_output[slot_mask] = -np.inf
            slots_logprobs = F.log_softmax(decoder_output.squeeze(1), dim=1)[:, slots]
            decoder_output[:, :, slots] = -np.inf
            '''

            pre_logprobs = F.log_softmax(decoder_output.squeeze(1), dim=1)
            slots_logprobs = pre_logprobs[:, slots]
            pre_logprobs[r_mask] = -np.inf
            pre_logprobs[e_mask] = -np.inf
            pre_logprobs[slot_mask] = -np.inf

            logprobs, argtop = torch.topk(F.log_softmax(pre_logprobs.squeeze(1), dim=1), top_k, dim=1)
            argtop = Variable(torch.cat((argtop.data, slots.expand(states_num * beam_size, slots_num)), dim=1)).squeeze(0)
            logprobs = Variable(torch.cat((logprobs.data, slots_logprobs.data), dim=1)).squeeze(0)
            curr_logprobs = (beam_probs.view(-1).unsqueeze(1).expand(states_num * beam_size, top_k + slots_num) + logprobs.data).view(-1)
            #transition = np.bitwise_or(np.repeat(state_codes, top_k + slots_num), np.tile(trans_state_codes, states_num * beam_size))
            transition = np.repeat(state_codes[:, :, np.newaxis, :], top_k + slots_num, axis=2) | trans_state_codes
            new_state_codes = transition.reshape(states_num * beam_size * (top_k + slots_num), slots_num)
            transition = torch.cuda.LongTensor(transition).sum(3).view(-1)
            for x in range(states_num):
                if (transition == x).any():
                    _logprobs = curr_logprobs.clone()
                    _logprobs[transition != x] = -np.inf
                    best_probs, best_args = _logprobs.topk(beam_size)
                    last = (best_args / (top_k + slots_num))
                    curr = (best_args % (top_k + slots_num))
                    beam[x, :, :] = beam.view(-1, max_len)[last, :]
                    beam_eos[x, :] = beam_eos.view(-1)[last]
                    beam_probs[x, :] = beam_probs.view(-1)[last]
                    stacks[x, :] = stacks.view(-1)[last]
                    state_codes[x, :, :] = new_state_codes[best_args.cpu().numpy(), :]
                    beam[x, :, t+1] = argtop[last, curr] * Variable(~beam_eos[x]).long() + eos_filler * Variable(beam_eos[x]).long()
                    mask = torch.cuda.ByteTensor(states_num, beam_size).fill_(0)
                    mask[x] = ~beam_eos[x]
                    beam_probs[mask] = (beam_probs[mask] * (t+1) + best_probs[~beam_eos[x]]) / (t+2)
                    decoder_hidden[0][:, x*beam_size:(x+1)*beam_size, :] = decoder_hidden[0][:, last, :]
                    decoder_hidden[1][:, x*beam_size:(x+1)*beam_size, :] = decoder_hidden[1][:, last, :]
            beam_eos = beam_eos | (beam[:, :, t+1] == self.eou).data
            decoder_input = self.embed(beam[:, :, t+1].contiguous().view(-1)).unsqueeze(1)
            stacks[beam[:, :, t+1].data == self.REDUCE] -= 1
            nonterminal = self.nonterminal.cuda()[beam[:, :, t+1].data.contiguous().view(-1)].view(states_num, beam_size)
            stacks[nonterminal] += 1
        best, best_arg = beam_probs[-1].max(0)
        generations = beam[-1][best_arg].data.cpu()
        return generations, best

    def rules_generate(self, src_seqs, src_lengths, indices, max_len, beam_size, top_k):
        src_embed = self.embed(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 

        _batch_size = src_embed.size(0)
        assert _batch_size == 1
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        decoder_input = self.embed(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>']))).unsqueeze(1)
        length = src_hidden.size(1)

        START = self.dictionary['NT-S']
        start_rule = self.rules_by_head['NT-S']
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        decoder_output = self.word_dist(self.out(decoder_output))

        decoder_output[~start_rule.expand(_batch_size, 1, self.vocab_size).cuda()] = -np.inf
        logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), beam_size, dim=1)
        beam = Variable(torch.zeros(_batch_size, beam_size, max_len)).long().cuda()
        beam[:, :, 0] = argtop
        beam_probs = logprobs.clone()
        beam_eos = (argtop == self.eou).data
        hidden = (decoder_hidden[0].clone(), decoder_hidden[1].clone())
        decoder_hidden = (decoder_hidden[0].unsqueeze(2).expand(1, _batch_size, beam_size, self.hidden_size).contiguous().view(1, _batch_size * beam_size, -1),
                          decoder_hidden[1].unsqueeze(2).expand(1, _batch_size, beam_size, self.hidden_size).contiguous().view(1, _batch_size * beam_size, -1))
        decoder_input = self.embed(argtop.view(-1)).unsqueeze(1)
        trees = np.array([None for x in range(beam_size)])
        current = np.array([None for x in range(beam_size)])
        for y in range(beam_size):
            rule = self.rules_by_id[beam.data[0, y, 0]].split()
            root = Node(rule[1])
            for nt in rule[3:]:
                ch = Node(nt, parent=root)
            #trees[y] = root
            current[y] = root.children[0]
            
        for t in range(max_len-1):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = self.word_dist(self.out(decoder_output))

            for x in range(beam_size):
                ind = ~torch.zeros(beam_size, 1, self.vocab_size).cuda().byte()
                ind[:, :, self.eou] = False
                if current[x] == None:
                    ind[x, 0, :] = start_rule
                    ind[x, 0, self.eou] = True
                else:
                    name = current[x].name
                    if self.preterminal[name]:
                        ind[x, 0, :] = ~self.rules
                    else:
                        cand = self.rules_by_head[name]
                        ind[x, 0, :] = cand
                decoder_output[~ind] = -np.inf

            logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), top_k, dim=1)
            best_probs, best_args = (beam_probs.view(-1).unsqueeze(1).expand(_batch_size * beam_size, top_k) + logprobs).view(_batch_size, beam_size, -1).view(_batch_size, -1).topk(beam_size)
            decoder_hidden = (decoder_hidden[0].view(1, _batch_size, beam_size, -1),
                              decoder_hidden[1].view(1, _batch_size, beam_size, -1))
            for x in range(_batch_size):
                last = (best_args / top_k)[x]
                curr = (best_args % top_k)[x]
                #trees = trees[last.data.tolist()]
                current = current[last.data.tolist()]
                beam[x, :, :] = beam[x][last, :]
                beam_eos[x, :] = beam_eos[x][last.data]
                beam_probs[x, :] = beam_probs[x][last.data]
                beam[x, :, t+1] = argtop.view(_batch_size, beam_size, top_k)[x][last.data, curr.data] * Variable(~beam_eos[x]).long() + eos_filler * Variable(beam_eos[x]).long()
                mask = torch.cuda.ByteTensor(_batch_size, beam_size).fill_(0)
                mask[x] = ~beam_eos[x]
                beam_probs[mask] = (beam_probs[mask] * (t+1) + best_probs[mask]) / (t+2)
                decoder_hidden[0][:, x, :, :] = decoder_hidden[0][:, x, :, :][:, last, :]
                decoder_hidden[1][:, x, :, :] = decoder_hidden[1][:, x, :, :][:, last, :]
            beam_eos = beam_eos | (beam[:, :, t+1] == self.eou).data
            decoder_hidden = (decoder_hidden[0].view(1, _batch_size * beam_size, -1),
                              decoder_hidden[1].view(1, _batch_size * beam_size, -1))
            decoder_input = self.embed(beam[:, :, t+1].contiguous().view(-1)).unsqueeze(1)

            for x in range(beam_size):
                current[x] = deepcopy(current[x])
                if current[x] is None:
                    if not (beam.data[0, x, :] == self.eou).any():
                        rule = self.rules_by_id[beam.data[0, x, t+1]].split()
                        root = Node(rule[1])
                        for nt in rule[3:]:
                            ch = Node(nt, parent=root)
                        current[x] = root.children[0]
                    continue
                name = current[x].name
                if self.preterminal[name]:
                    while True:
                        if current[x].parent is None:
                            current[x] = None
                            break
                        pos = current[x].parent.children.index(current[x])
                        if pos >= len(current[x].parent.children) - 1:
                            current[x] = current[x].parent
                        else:
                            current[x] = current[x].parent.children[pos+1]
                            break
                else:
                    rule = self.rules_by_id[beam.data[0, x, t+1]].split()
                    for nt in rule[3:]:
                        ch = Node(nt, parent=current[x])
                    current[x] = current[x].children[0]

        best, best_arg = beam_probs.max(1)
        generations = beam[torch.arange(_batch_size).long().cuda(), best_arg.data].data.cpu()
        return generations, best

    def rules_generate_cstr(self, src_seqs, src_lengths, indices, max_len, beam_size, top_k, slots):
        src_embed = self.embed(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 

        _batch_size = src_embed.size(0)
        assert _batch_size == 1
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        decoder_input = self.embed(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>']))).unsqueeze(1)
        length = src_hidden.size(1)
 
        states_num = 2
        slots_num = len(slots)
        slots = torch.cuda.LongTensor(slots)
        transition = np.zeros((states_num, slots_num + top_k), dtype=np.int32)
        state_codes = np.zeros(beam_size + slots_num, dtype=np.int32)
        transition[0, top_k:top_k+slots_num] = 1
        transition[1, :] = 1
        state_codes[beam_size:] = 1
        transition = np.repeat(transition[:, np.newaxis, :], beam_size, axis=1)
        transition = torch.cuda.LongTensor(transition).view(-1)

        START = self.dictionary['NT-S']
        start_rule = self.rules_by_head['NT-S']
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        decoder_output = self.word_dist(self.out(decoder_output))

        decoder_output[~start_rule.expand(_batch_size, 1, self.vocab_size).cuda()] = -np.inf
        logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), beam_size, dim=1)
        logprobs = logprobs[0]
        argtop = argtop[0]
        beam = Variable(torch.zeros(states_num, beam_size, max_len)).long().cuda()
        beam_probs = torch.zeros(states_num, beam_size).cuda().fill_(-np.inf)
        beam_eos = torch.zeros(states_num, beam_size).byte().cuda()
        for x in range(1):
            if (state_codes == x).any():
                state_going = torch.from_numpy(np.where(state_codes == x)[0]).cuda()
                state_going_num = min(beam_size, state_going.shape[0])
                beam[x, :state_going_num, 0] = argtop[state_going[:state_going_num]]
                beam_probs[x, :state_going_num] = logprobs[state_going[:state_going_num]].data
                beam_eos[x, :state_going_num] = (argtop[state_going[:state_going_num]] == self.eou).data
        decoder_hidden = (decoder_hidden[0].unsqueeze(2).expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1),
                          decoder_hidden[1].unsqueeze(2).expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1))
        decoder_input = self.embed(beam[:, :, 0].contiguous().view(-1)).unsqueeze(1)

        current = np.array([[None for x in range(beam_size)] for y in range(states_num)])
        for x in range(1):
            for y in range(beam_size):
                rule = self.rules_by_id[beam.data[x, y, 0]].split()
                root = Node(rule[1])
                for nt in rule[3:]:
                    ch = Node(nt, parent=root)
                #trees[y] = root
                current[x, y] = root.children[0]
        
        for t in range(max_len-1):
            new_beam = beam.clone()
            new_beam_probs = beam_probs.clone()
            new_beam_eos = beam_eos.clone()
            new_current = deepcopy(current)

            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = self.word_dist(self.out(decoder_output))
            slots_logprobs = F.log_softmax(decoder_output, dim=2)[:, :, slots]
            decoder_output[:, :, slots] = -np.inf
            for x in range(states_num):
                for y in range(beam_size):
                    ind = ~torch.zeros(states_num * beam_size, 1, self.vocab_size).cuda().byte()
                    ind[:, :, self.eou] = False
                    if current[x, y] == None:
                        ind[x * beam_size + y, 0, :] = start_rule
                        ind[x * beam_size + y, 0, self.eou] = True
                        slots_logprobs[x * beam_size + y, :, :] = -np.inf
                    else:
                        name = current[x, y].name
                        if self.preterminal[name]:
                            ind[x * beam_size + y, 0, :] = ~self.rules
                        else:
                            cand = self.rules_by_head[name]
                            ind[x * beam_size + y, 0, :] = cand
                            slots_logprobs[x * beam_size + y, :, :] = -np.inf
                    decoder_output[~ind] = -np.inf

            '''
            logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), top_k, dim=1)
            best_probs, best_args = (beam_probs.view(-1).unsqueeze(1).expand(_batch_size * beam_size, top_k) + logprobs).view(_batch_size, beam_size, -1).view(_batch_size, -1).topk(beam_size)
            decoder_hidden = (decoder_hidden[0].view(1, _batch_size, beam_size, -1),
                              decoder_hidden[1].view(1, _batch_size, beam_size, -1))
            '''

            logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), top_k, dim=1)
            argtop = Variable(torch.cat((argtop.data, slots.expand(states_num * beam_size, slots_num)), dim=1))
            logprobs = Variable(torch.cat((logprobs.data, slots_logprobs.squeeze(1).data), dim=1))
            curr_logprobs = (beam_probs.view(-1).unsqueeze(1).expand(states_num * beam_size, top_k + slots_num) + logprobs.data).view(-1)
            current = current.reshape(states_num * beam_size)
            for x in range(states_num):
                _logprobs = curr_logprobs.clone()
                _logprobs[transition != x] = -np.inf
                if (_logprobs != -np.inf).any():
                    best_probs, best_args = _logprobs.topk(beam_size)
                    last = (best_args / (top_k + slots_num))
                    curr = (best_args % (top_k + slots_num))
                    new_current[x] = current[last.tolist()]
                    new_beam[x, :, :] = beam.view(-1, max_len)[last, :]
                    new_beam_eos[x, :] = beam_eos.view(-1)[last]
                    new_beam_probs[x, :] = beam_probs.view(-1)[last]
                    new_beam[x, :, t+1] = argtop[last, curr] * Variable(~new_beam_eos[x]).long() + eos_filler * Variable(new_beam_eos[x]).long()
                    mask = torch.cuda.ByteTensor(states_num, beam_size).fill_(0)
                    mask[x] = ~new_beam_eos[x]
                    new_beam_probs[mask] = (new_beam_probs[mask] * (t+1) + best_probs[~new_beam_eos[x]]) / (t+2)
                    decoder_hidden[0][:, x*beam_size:(x+1)*beam_size, :] = decoder_hidden[0][:, last, :]
                    decoder_hidden[1][:, x*beam_size:(x+1)*beam_size, :] = decoder_hidden[1][:, last, :]

            '''
            for x in range(_batch_size):
                last = (best_args / top_k)[x]
                curr = (best_args % top_k)[x]
                #trees = trees[last.data.tolist()]
                current = current[last.data.tolist()]
                beam[x, :, :] = beam[x][last, :]
                beam_eos[x, :] = beam_eos[x][last.data]
                beam_probs[x, :] = beam_probs[x][last.data]
                beam[x, :, t+1] = argtop.view(_batch_size, beam_size, top_k)[x][last.data, curr.data] * Variable(~beam_eos[x]).long() + eos_filler * Variable(beam_eos[x]).long()
                mask = torch.cuda.ByteTensor(_batch_size, beam_size).fill_(0)
                mask[x] = ~beam_eos[x]
                beam_probs[mask] = (beam_probs[mask] * (t+1) + best_probs[mask]) / (t+2)
                decoder_hidden[0][:, x, :, :] = decoder_hidden[0][:, x, :, :][:, last, :]
                decoder_hidden[1][:, x, :, :] = decoder_hidden[1][:, x, :, :][:, last, :]
            '''
            beam_eos = new_beam_eos | (new_beam[:, :, t+1] == self.eou).data
            decoder_input = self.embed(new_beam[:, :, t+1].contiguous().view(-1)).unsqueeze(1)
            beam = new_beam
            beam_probs = new_beam_probs
            current = new_current

            for x in range(states_num):
                for y in range(beam_size):
                    current[x, y] = deepcopy(current[x, y])
                    if beam_probs[x, y] == -np.inf:
                        continue
                    if current[x, y] is None:
                        if not (beam.data[x, y, :] == self.eou).any():
                            rule = self.rules_by_id[beam.data[x, y, t+1]].split()
                            root = Node(rule[1])
                            for nt in rule[3:]:
                                ch = Node(nt, parent=root)
                            current[x, y] = root.children[0]
                        continue
                    name = current[x, y].name
                    if self.preterminal[name]:
                        while True:
                            if current[x, y].parent is None:
                                current[x, y] = None
                                break
                            pos = current[x, y].parent.children.index(current[x, y])
                            if pos >= len(current[x, y].parent.children) - 1:
                                current[x, y] = current[x, y].parent
                            else:
                                current[x, y] = current[x, y].parent.children[pos+1]
                                break
                    else:
                        rule = self.rules_by_id[beam.data[x, y, t+1]].split()
                        for nt in rule[3:]:
                            ch = Node(nt, parent=current[x, y])
                        current[x, y] = current[x, y].children[0]

        best, best_arg = beam_probs[-1].max(0)
        generations = beam[-1][best_arg].data.cpu()
        return generations, best

    def loss(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, sampling_rate):
        decoder_outputs = self.forward(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, sampling_rate)
        loss = compute_perplexity(decoder_outputs, Variable(trg_seqs[:, 1:].cuda()), Variable(torch.cuda.LongTensor(trg_lengths)) - 1)

        return loss

    def evaluate(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths):
        decoder_outputs = self.forward(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, 1)
        loss = compute_perplexity(decoder_outputs, Variable(trg_seqs[:, 1:].cuda()), Variable(torch.cuda.LongTensor(trg_lengths)) - 1)

        return loss


class PersonaGrammarDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, word_vectors, dictionary):
        super(PersonaGrammarDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.embed = Embedding(vocab_size, input_size, word_vectors, trainable=True)
        self.encoder = nn.LSTM(input_size, hidden_size)
        self.decoder = nn.LSTM(input_size + hidden_size, hidden_size, batch_first=True)
        self.key_size = 100
        self.q_key = nn.Linear(input_size, self.key_size)
        self.q_value = nn.Linear(input_size, hidden_size)
        self.a_key = nn.Linear(hidden_size, self.key_size)
        self.max_len = 21
        self.out = nn.Linear(hidden_size * 2, input_size)
        self.word_dist = nn.Linear(input_size, vocab_size)
        self.context_fc1 = nn.Linear(hidden_size * 2, hidden_size // 2 * 3)
        self.context_fc2 = nn.Linear(hidden_size // 2 * 3, hidden_size)
        self.hidden_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.hidden_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.cell_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.discriminator = None
        self.dictionary = dictionary
        self.eou = dictionary['__eou__']
        self.word_dist.weight = self.embed.weight

        for names in self.encoder._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.encoder, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(0.)

        for names in self.decoder._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.decoder, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(0.)

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()

    def context_transform(self, context):
        return self.context_fc2(F.relu(self.context_fc1(context)))

    def hidden_transform(self, hidden):
        return self.hidden_fc2(F.relu(self.hidden_fc1(hidden)))

    def cell_transform(self, cell):
        return self.cell_fc2(F.relu(self.cell_fc1(cell)))

    def init_hidden(self, src_hidden):
        hidden = src_hidden[0]
        cell = src_hidden[1]
        hidden = self.hidden_transform(hidden)
        cell = self.cell_transform(cell)
        return (hidden, cell)
        '''
        return src_hidden
        '''
    def embed_context(self, ctc_seq):
        embed = Variable(torch.cuda.FloatTensor(len(ctc_seq), self.input_size))
        for x in range(len(ctc_seq)):
            embed[x] = self.embed(Variable(torch.cuda.LongTensor(ctc_seq[x]))).sum(0) / len(ctc_seq[x])
        return embed

    def forward(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, ctc_seqs, ctc_lengths, sampling_rate):
        batch_size = src_seqs.size(0)

        src_embed = self.embed(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden)

        length = src_hidden.size(1)
        max_ctc_len = max(ctc_lengths)
        ans_embed = self.embed(Variable(trg_seqs).cuda())
        trg_l = ans_embed.size(1)

        decoder_input = ans_embed[:, 0, :].unsqueeze(1)
        decoder_outputs = Variable(torch.FloatTensor(batch_size, trg_l - 1, self.vocab_size).cuda())
        context_embeddings = Variable(torch.cuda.FloatTensor(batch_size, max_ctc_len, self.input_size))
        for x in range(batch_size):
            context_embeddings[x, :ctc_lengths[x], :] = self.embed_context(ctc_seqs[x])

        q_key = F.tanh(self.q_key(context_embeddings))
        q_value = self.q_value(context_embeddings)
        for step in range(trg_l - 1):
            a_key = F.tanh(self.a_key(decoder_hidden[0].squeeze(0)))

            '''
            q_key = F.tanh(self.q_key(src_hidden))
            q_value = self.q_value(src_hidden)
            q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
            q_mask  = torch.arange(length).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(length, 1).transpose(0, 1)
            q_energy[~q_mask] = -np.inf
            q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
            q_context = torch.bmm(q_weights, q_value).squeeze(1)
            '''
            q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
            q_mask  = torch.arange(max_ctc_len).long().cuda().repeat(batch_size, 1) < torch.cuda.LongTensor(ctc_lengths).repeat(max_ctc_len, 1).transpose(0, 1)
            q_energy[~q_mask] = -np.inf
            q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
            q_context = torch.bmm(q_weights, q_value).squeeze(1)

            #context = torch.cat((q_context, i_context), dim=1)
            context = q_context.unsqueeze(1)
            decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2), decoder_hidden)
            decoder_outputs[:, step, :] = self.word_dist(self.out(torch.cat((decoder_output.squeeze(1), context.squeeze(1)), dim=1)))
            #decoder_outputs[:, step, :] = decoder_output.squeeze(1)
            if np.random.uniform() < sampling_rate and step < self.max_len - 2:
                decoder_input = ans_embed[:, step+1, :].unsqueeze(1)
            else:
                words = decoder_outputs[:, step, :].max(dim=1)[1]
                decoder_input = self.embed(words).unsqueeze(1)

        return decoder_outputs

    def generate(self, src_seqs, src_lengths, ctc_seqs, ctc_lengths, indices, max_len, beam_size, top_k):
        src_embed = self.embed(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden)

        cenc_out = src_last_hidden
        _batch_size = src_embed.size(0)
        # cenc_out: (batch_size, dim2)
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        decoder_input = self.embed(Variable(torch.zeros(_batch_size).long().cuda().fill_(self.dictionary['<start>']))).unsqueeze(1)
        length = src_hidden.size(1)
        max_ctc_len = max(ctc_lengths)
        context_embeddings = Variable(torch.cuda.FloatTensor(_batch_size, max_ctc_len, self.input_size))
        for x in range(_batch_size):
            context_embeddings[x, :ctc_lengths[x], :] = self.embed_context(ctc_seqs[x])

        a_key = F.tanh(self.a_key(decoder_hidden[0].squeeze(0)))

        q_key = F.tanh(self.q_key(context_embeddings))
        q_value = self.q_value(context_embeddings)
        q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
        #q_mask  = torch.arange(length).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(length, 1).transpose(0, 1)
        q_mask  = torch.arange(max_ctc_len).long().cuda().repeat(_batch_size, 1) < torch.cuda.LongTensor(ctc_lengths).repeat(max_ctc_len, 1).transpose(0, 1)
        q_energy[~q_mask] = -np.inf
        q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
        q_context = torch.bmm(q_weights, q_value).squeeze(1)

        #context = torch.cat((q_context, i_context), dim=1)
        context = q_context.unsqueeze(1)

        decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2), decoder_hidden)
        decoder_output = self.word_dist(self.out(torch.cat((decoder_output.squeeze(1), context.squeeze(1)), dim=1)))
        logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), beam_size, dim=1)
        #decoder_input = self.embedding(argtop[0])
        beam = Variable(torch.zeros(_batch_size, beam_size, max_len)).long().cuda()
        beam[:, :, 0] = argtop
        #beam_probs = logprobs[0].clone()
        beam_probs = logprobs.clone()
        #beam_eos = (argtop == self.eou)[0].data
        beam_eos = (argtop == self.eou).data
        hidden = (decoder_hidden[0].clone(), decoder_hidden[1].clone())
        decoder_hidden = (decoder_hidden[0].unsqueeze(2).expand(1, _batch_size, beam_size, self.hidden_size).contiguous().view(1, _batch_size * beam_size, -1),
                          decoder_hidden[1].unsqueeze(2).expand(1, _batch_size, beam_size, self.hidden_size).contiguous().view(1, _batch_size * beam_size, -1))
        decoder_input = self.embed(argtop.view(-1)).unsqueeze(1)
    
        for t in range(max_len-1):
            a_key = F.tanh(self.a_key(hidden[0].squeeze(0)))

            q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
            #q_mask  = torch.arange(length).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(length, 1).transpose(0, 1)
            q_mask  = torch.arange(max_ctc_len).long().cuda().repeat(_batch_size, 1) < torch.cuda.LongTensor(ctc_lengths).repeat(max_ctc_len, 1).transpose(0, 1)
            q_energy[~q_mask] = -np.inf
            q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
            q_context = torch.bmm(q_weights, q_value).squeeze(1)

            context = q_context.unsqueeze(1).expand(_batch_size, beam_size, self.hidden_size).contiguous().view(_batch_size * beam_size, 1, -1)
            decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2), decoder_hidden)
            decoder_output = self.word_dist(self.out(torch.cat((decoder_output.squeeze(1), context.squeeze(1)), dim=1)))

            logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), top_k, dim=1)
            #best_probs, best_args = (beam_probs.repeat(top_k, 1).transpose(0, 1) + logprobs).view(-1).topk(beam_size)
            best_probs, best_args = (beam_probs.view(-1).unsqueeze(1).expand(_batch_size * beam_size, top_k) + logprobs).view(_batch_size, beam_size, -1).view(_batch_size, -1).topk(beam_size)
            decoder_hidden = (decoder_hidden[0].view(1, _batch_size, beam_size, -1),
                              decoder_hidden[1].view(1, _batch_size, beam_size, -1))
            for x in range(_batch_size):
                last = (best_args / top_k)[x]
                curr = (best_args % top_k)[x]
                beam[x, :, :] = beam[x][last, :]
                beam_eos[x, :] = beam_eos[x][last.data]
                beam_probs[x, :] = beam_probs[x][last.data]
                beam[x, :, t+1] = argtop.view(_batch_size, beam_size, top_k)[x][last.data, curr.data] * Variable(~beam_eos[x]).long() + eos_filler * Variable(beam_eos[x]).long()
                mask = torch.cuda.ByteTensor(_batch_size, beam_size).fill_(0)
                mask[x] = ~beam_eos[x]
                beam_probs[mask] = (beam_probs[mask] * (t+1) + best_probs[mask]) / (t+2)
                decoder_hidden[0][:, x, :, :] = decoder_hidden[0][:, x, :, :][:, last, :]
                decoder_hidden[1][:, x, :, :] = decoder_hidden[1][:, x, :, :][:, last, :]
            beam_eos = beam_eos | (beam[:, :, t+1] == self.eou).data
            decoder_hidden = (decoder_hidden[0].view(1, _batch_size * beam_size, -1),
                              decoder_hidden[1].view(1, _batch_size * beam_size, -1))
            decoder_input = self.embed(beam[:, :, t+1].contiguous().view(-1)).unsqueeze(1)
        best, best_arg = beam_probs.max(1)
        generations = beam[torch.arange(_batch_size).long().cuda(), best_arg.data].data.cpu()
        return generations, best

    def loss(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, ctc_seqs, ctc_lengths, sampling_rate):
        decoder_outputs = self.forward(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, ctc_seqs, ctc_lengths, sampling_rate)
        loss = compute_perplexity(decoder_outputs, Variable(trg_seqs[:, 1:].cuda()), Variable(torch.cuda.LongTensor(trg_lengths)) - 1)

        return loss

    def evaluate(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, ctc_seqs, ctc_lengths):
        decoder_outputs = self.forward(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, ctc_seqs, ctc_lengths, 1)
        loss = compute_perplexity(decoder_outputs, Variable(trg_seqs[:, 1:].cuda()), Variable(torch.cuda.LongTensor(trg_lengths)) - 1)

        return loss


class LexicalizedGrammarDecoder(nn.Module):
    def __init__(self, lex_input_size, nt_input_size, rule_input_size, 
                 context_size, hidden_size, 
                 lex_vocab_size, nt_vocab_size, rule_vocab_size, 
                 lex_vectors, nt_vectors, rule_vectors,
                 lex_dictionary, nt_dictionary, rule_dictionary,
                 lex_level):
        super(LexicalizedGrammarDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.lex_input_size = lex_input_size
        self.nt_input_size = nt_input_size
        self.rule_input_size = rule_input_size
        self.lex_vocab_size = lex_vocab_size
        self.nt_vocab_size = nt_vocab_size
        self.rule_vocab_size = rule_vocab_size
        self.lex_level = lex_level
        self.lexicon = Embedding(lex_vocab_size, lex_input_size, lex_vectors, trainable=False)
        self.constituent = Embedding(nt_vocab_size, nt_input_size, nt_vectors, trainable=True)
        self.rule = Embedding(rule_vocab_size, rule_input_size, rule_vectors, trainable=True)
        self.depth = Embedding(25, 256, trainable=True)
        self.breadth = Embedding(40, 256, trainable=True)
        self.pos_embed_size = 0
        self.encoder = nn.LSTM(lex_input_size, hidden_size)
        self.context_size = context_size
        if self.lex_level == 0:
            self.lex_out = nn.Linear(hidden_size, lex_input_size)
            self.nt_out = nn.Linear(hidden_size, nt_input_size)
            #self.nt_dist.weight = self.constituent.weight
            self.leaf_decoder = nn.LSTM(lex_input_size, hidden_size, batch_first=True, dropout=0.0)
            self.decoder_input_size = nt_input_size + rule_input_size
            #self.decoder_input_size = lex_input_size * 6 + nt_input_size * 7

            self.leaf_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
            self.leaf_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
            self.leaf_cell_fc1 = nn.Linear(hidden_size, hidden_size)
            self.leaf_cell_fc2 = nn.Linear(hidden_size, hidden_size)

            #self.hidden_out_fc1 = nn.Linear(hidden_size * 3, hidden_size * 2)
            #self.hidden_out_fc2 = nn.Linear(hidden_size * 2, hidden_size * 1)
            self.decoder = nn.LSTM(self.decoder_input_size, hidden_size, batch_first=True, dropout=0.0)

            self.lex_hidden_out = nn.Linear(hidden_size * 2 + context_size, hidden_size)
            self.rule_hidden_out = nn.Linear(hidden_size * 2 + context_size, hidden_size)

            #self.rule_out = nn.Linear(hidden_size, rule_input_size)
            self.rule_dist = nn.Linear(hidden_size, rule_vocab_size)
            #self.rule_dist.weight = self.rule.weight
        elif self.lex_level == 1:
            self.lex_out = nn.Linear(hidden_size + nt_input_size, lex_input_size)
            self.nt_dist = nn.Linear(nt_input_size, nt_vocab_size)
            self.nt_out = nn.Linear(hidden_size, nt_input_size)
            self.nt_dist.weight = self.constituent.weight
            self.tree_input_size = lex_input_size + nt_input_size
            self.tree = nn.Linear((lex_input_size + nt_input_size) * 4, self.tree_input_size)
            self.decoder_input_size = lex_input_size * 4 + nt_input_size * 5 + self.tree_input_size
            self.decoder = nn.LSTM(self.decoder_input_size + context_size, hidden_size, batch_first=True, dropout=0.0)
        else:
            #self.lex_out = nn.Linear(hidden_size + lex_input_size, lex_input_size)
            self.lex_out = nn.Linear(hidden_size, lex_input_size)
            self.nt_dist = nn.Linear(nt_input_size, nt_vocab_size)
            self.nt_out = nn.Linear(hidden_size, nt_input_size)
            #self.nt_dist.weight = self.constituent.weight
            self.tree_input_size = lex_input_size * 2
            self.tree = nn.Linear(lex_input_size * 4 + nt_input_size * 0 + rule_input_size * 0, self.tree_input_size)
            self.leaf_decoder = nn.LSTM(lex_input_size, hidden_size, batch_first=True, dropout=0.0)
            self.lex_decoder = nn.LSTM(lex_input_size, hidden_size, batch_first=True, dropout=0.0)
            self.anc_decoder = nn.LSTM(lex_input_size, hidden_size, batch_first=True, dropout=0.0)
            self.anc_syn_decoder = nn.LSTM(nt_input_size + rule_input_size, hidden_size, batch_first=True, dropout=0.0)
            self.decoder_input_size = nt_input_size + rule_input_size + lex_input_size
            #self.decoder_input_size = lex_input_size * 6 + nt_input_size * 7

            self.leaf_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
            self.leaf_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
            self.leaf_cell_fc1 = nn.Linear(hidden_size, hidden_size)
            self.leaf_cell_fc2 = nn.Linear(hidden_size, hidden_size)

            self.lex_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
            self.lex_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
            self.lex_cell_fc1 = nn.Linear(hidden_size, hidden_size)
            self.lex_cell_fc2 = nn.Linear(hidden_size, hidden_size)

            self.anc_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
            self.anc_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
            self.anc_cell_fc1 = nn.Linear(hidden_size, hidden_size)
            self.anc_cell_fc2 = nn.Linear(hidden_size, hidden_size)

            self.anc_syn_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
            self.anc_syn_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
            self.anc_syn_cell_fc1 = nn.Linear(hidden_size, hidden_size)
            self.anc_syn_cell_fc2 = nn.Linear(hidden_size, hidden_size)

            self.rule_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
            self.rule_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
            self.rule_cell_fc1 = nn.Linear(hidden_size, hidden_size)
            self.rule_cell_fc2 = nn.Linear(hidden_size, hidden_size)

            #self.hidden_out_fc1 = nn.Linear(hidden_size * 3, hidden_size * 2)
            #self.hidden_out_fc2 = nn.Linear(hidden_size * 2, hidden_size * 1)
            self.decoder = nn.LSTM(self.decoder_input_size, hidden_size, batch_first=True, dropout=0.0)
            self.rule_decoder = nn.LSTM(self.rule_input_size, hidden_size, batch_first=True, dropout=0.0)
            if self.lex_level == 2:
                self.lex_hidden_out = nn.Linear(hidden_size * 3 + nt_input_size, hidden_size)
                self.rule_hidden_out = nn.Linear(hidden_size * 3 + nt_input_size, hidden_size)
                self.nt_hidden_out = nn.Linear(hidden_size * 3, hidden_size)
            else:
                self.decoder_output_size = hidden_size * 2 + context_size * 2
                self.lex_hidden_out = nn.Linear(self.decoder_output_size, lex_input_size)
                self.top_lex_hidden_out = nn.Linear(self.decoder_output_size, lex_input_size)
                self.rule_hidden_out = nn.Linear(hidden_size * 2 + nt_input_size, rule_input_size)
                self.nt_hidden_out = nn.Linear(hidden_size * 2, nt_vocab_size)

                self.rule_out = nn.Linear(hidden_size, rule_input_size)
                self.rule_dist = nn.Linear(rule_input_size, rule_vocab_size)
                self.rule_dist.weight = self.rule.weight
                self.nt_dist.weight = self.constituent.weight
        #self.rule_out = nn.Linear(hidden_size, rule_input_size)
        self.lex_dist = nn.Linear(lex_input_size, lex_vocab_size)
        self.hidden_fc1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_fc2 = nn.Linear(hidden_size, hidden_size)
        self.cell_fc1 = nn.Linear(hidden_size, hidden_size)
        self.cell_fc2 = nn.Linear(hidden_size, hidden_size)
        self.lex_dictionary = lex_dictionary
        self.nt_dictionary = nt_dictionary
        self.rule_dictionary = rule_dictionary
        self.eou = lex_dictionary['__eou__']
        self.lex_dist.weight = self.lexicon.weight

        if self.lex_level == 0:
            self.a_key = nn.Linear(hidden_size, 100)
            self.p_key = nn.Linear(hidden_size * 2, 100)
        else:
            self.a_key = nn.Linear(hidden_size * 2, 100)
            self.p_key = nn.Linear(hidden_size * 3, 100)
        self.q_key = nn.Linear(hidden_size, 100)
        self.psn_key = nn.Linear(hidden_size, 100)
        self.q_value = nn.Linear(hidden_size, context_size)
        self.psn_value = nn.Linear(hidden_size, context_size)

        self.init_forget_bias(self.encoder, 0)
        self.init_forget_bias(self.decoder, 0)
        self.init_forget_bias(self.leaf_decoder, 0)
        if self.lex_level >= 2:
            self.init_forget_bias(self.lex_decoder, 0)
            self.init_forget_bias(self.anc_decoder, 0)

    def init_rules(self):
        self.ROOT = self.nt_dictionary['ROOT']
        self.rules_by_id = defaultdict(str)
        self.nt_by_id = defaultdict(str)
        rules_by_nt = defaultdict(list)
        for k, v in self.rule_dictionary.items():
            self.rules_by_id[v] = k[6:]
            for nt in k.split()[1:]:
                rules_by_nt[nt].append(self.rule_dictionary[k])

        for k, v in self.nt_dictionary.items():
            self.nt_by_id[v] = k

        #self.rules_by_nt = {}
        self.rules_by_nt = torch.cuda.ByteTensor(self.nt_vocab_size, self.rule_vocab_size).fill_(False)
        for k, v in rules_by_nt.items():
            vv = torch.cuda.ByteTensor(self.rule_vocab_size).fill_(False)
            vv[v] = True
            self.rules_by_nt[self.nt_dictionary[k]] = vv

        self.preterminal = defaultdict(bool)
        for pt in Preterminal:
            self.preterminal[pt] = True

    def init_forget_bias(self, rnn, b):
        for names in rnn._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2 
                bias.data[start:end].fill_(b)

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        if self.lex_level >= 2:
            self.leaf_decoder.flatten_parameters()
            self.lex_decoder.flatten_parameters()
            self.anc_decoder.flatten_parameters()
            self.anc_syn_decoder.flatten_parameters()

    def hidden_transform(self, hidden, prefix):
        return eval('F.tanh(self.{}hidden_fc2(F.relu(self.{}hidden_fc1(hidden))))'.format(prefix, prefix))

    def cell_transform(self, cell, prefix):
        return eval('F.tanh(self.{}cell_fc2(F.relu(self.{}cell_fc1(cell))))'.format(prefix, prefix))

    def init_hidden(self, src_hidden, prefix=''):
        hidden = src_hidden[0]
        cell = src_hidden[1]
        hidden = self.hidden_transform(hidden, prefix)
        cell = self.cell_transform(cell, prefix)
        return (hidden, cell)

    def attention(self, decoder_hidden, src_hidden, src_max_len, src_lengths, src_last_hidden, psn_hidden, psn_max_len, psn_lengths):
        '''
        a_key = F.tanh(self.a_key(decoder_hidden[0].squeeze(0)))
        q_key = F.tanh(self.q_key(src_hidden))
        q_value = F.tanh(self.q_value(src_hidden))
        q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
        q_mask  = torch.arange(src_max_len).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(src_max_len, 1).transpose(0, 1)
        q_energy[~q_mask] = -np.inf
        q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
        q_context = torch.bmm(q_weights, q_value).squeeze(1)
        context = q_context.unsqueeze(1)
        '''
        a_key = F.tanh(self.a_key(decoder_hidden[0].squeeze(0)))
        p_key = F.tanh(self.p_key(torch.cat((decoder_hidden[0].squeeze(0), src_last_hidden[0].squeeze(0)), dim=1)))

        q_key = F.tanh(self.q_key(src_hidden))
        psn_key = F.tanh(self.psn_key(psn_hidden))
        q_value = F.tanh(self.q_value(src_hidden))
        psn_value = F.tanh(self.psn_value(psn_hidden))
        q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
        psn_energy = torch.bmm(psn_key, p_key.unsqueeze(2)).squeeze(2)
        q_mask = torch.arange(src_max_len).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(src_max_len, 1).transpose(0, 1)
        psn_mask = torch.arange(psn_max_len).long().cuda().repeat(psn_hidden.size(0), 1) < psn_lengths.cuda().repeat(psn_max_len, 1).transpose(0, 1)
        q_energy[~q_mask] = -np.inf
        psn_energy[~psn_mask] = -np.inf
        '''
        q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
        psn_weights = F.softmax(psn_energy, dim=1).unsqueeze(1)
        '''
        q_weights = F.sigmoid(q_energy).unsqueeze(1)
        psn_weights = F.sigmoid(psn_energy).unsqueeze(1)
        q_context = torch.bmm(q_weights, q_value)
        psn_context = torch.bmm(psn_weights, psn_value)

        #return context
        return torch.cat((q_context, psn_context), dim=2)

    def encode(self, tar_seqs, indices, name, init_hidden):
        lengths = [ind.max().item() for ind in indices]
        max_len = max(lengths)
        mask = [x.copy() for x in indices]
        _indices = [None for x in indices]
        for x in range(len(indices)):
            mask[x][1:] -= mask[x][:-1].copy()
            _indices[x] = np.zeros(tar_seqs[0].size(1), dtype=np.int64)
            _indices[x][1:len(indices[x])] = indices[x][:-1]
            _indices[x][0] = 0
        tar_lex_embed = self.lexicon(Variable(tar_seqs[1]).cuda())
        #tar_nt_embed = self.constituent(Variable(tar_seqs[0]).cuda())
        #_tar_embed = torch.cat((tar_lex_embed, tar_nt_embed), dim=2)
        _tar_embed = tar_lex_embed
        tar_embed = Variable(torch.zeros(_tar_embed.size(0), max_len, _tar_embed.size(2)).cuda())
        for x in range(tar_embed.size(0)):
            ind = torch.from_numpy(np.arange(len(mask[x]))[mask[x].astype(bool)]).long().cuda()
            tar_embed[x, :lengths[x], :] = _tar_embed[x][ind]
        t_lengths, perm_idx = torch.LongTensor(lengths).sort(0, descending=True)
        tar_embed = tar_embed[perm_idx.cuda()]
        packed_input = pack_padded_sequence(tar_embed, t_lengths.numpy(), batch_first=True)
        hidden, _ = eval('self.{}_decoder(packed_input, init_hidden)'.format(name))
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        hidden = hidden[perm_idx.sort()[1].cuda()]

        return hidden, _indices

    def encode_anc(self, anc_embed, anc_lengths, trg_lengths, batch_size, src_last_hidden, name):
        f_lengths, perm_idx = torch.LongTensor(anc_lengths).sort(0, descending=True)
        anc_embed = anc_embed[perm_idx.cuda()]
        nonzeros = f_lengths.nonzero().squeeze(1)
        zeros = (f_lengths == 0).nonzero().squeeze(1)
        packed_input = pack_padded_sequence(anc_embed[nonzeros.cuda()], f_lengths[nonzeros].numpy(), batch_first=True)
        anc_output, anc_last_hidden = eval('self.{}decoder(packed_input)'.format(name))
        anc_init_hidden = self.init_hidden(src_last_hidden, name) 
        _anc_hidden = Variable(torch.cuda.FloatTensor(len(anc_lengths), self.hidden_size).fill_(0))
        _anc_hidden[:nonzeros.size(0)] = anc_last_hidden[0].squeeze(0)
        _anc_hidden = _anc_hidden[perm_idx.sort()[1].cuda()]
        _anc_hidden[perm_idx[zeros].cuda()] = anc_init_hidden[0].squeeze(0)

        anc_hidden = Variable(torch.cuda.FloatTensor(batch_size, max(trg_lengths), self.hidden_size).fill_(0))
        start = 0
        for x in range(batch_size):
            anc_hidden[x, :trg_lengths[x], :] = _anc_hidden[start:start+trg_lengths[x]]
            start += trg_lengths[x]

        return anc_hidden, anc_init_hidden

    def forward(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, positions, ancestors, anc_lengths):
        batch_size = src_seqs.size(0)

        src_embed = self.lexicon(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 

        psn_lengths, perm_idx = torch.LongTensor(psn_lengths).sort(0, descending=True)
        psn_embed = self.lexicon(Variable(psn_seqs).cuda())
        psn_embed = psn_embed[perm_idx.cuda()]
        packed_input = pack_padded_sequence(psn_embed, psn_lengths.numpy(), batch_first=True)
        psn_output, psn_last_hidden = self.encoder(packed_input)
        psn_hidden, _ = pad_packed_sequence(psn_output, batch_first=True)
        psn_hidden = psn_hidden[perm_idx.sort()[1].cuda()]
        psn_lengths = psn_lengths[perm_idx.sort()[1]]

        #rule_decoder_hidden = self.init_hidden(src_last_hidden, 'rule_') 
        leaf_init_hidden = self.init_hidden(src_last_hidden, 'leaf_') 
        leaf_decoder_hidden, _leaf_indices = self.encode(trg_seqs, leaf_indices, 'leaf', leaf_init_hidden)
        leaf_decoder_hidden = torch.cat((leaf_init_hidden[0].transpose(0, 1), leaf_decoder_hidden), dim=1)
        if self.lex_level >= 2:
            '''
            _indices = [np.arange(len(lex_indices[x])) + 1 for x in range(len(lex_indices))]
            lex_init_hidden = self.init_hidden(src_last_hidden, 'lex_') 
            lex_decoder_hidden, _lex_indices = self.encode(trg_seqs, _indices, 'lex', lex_init_hidden)
            lex_decoder_hidden = torch.cat((lex_init_hidden[0].transpose(0, 1), lex_decoder_hidden), dim=1)
            '''
            pass

        ans_nt = self.constituent(Variable(trg_seqs[0]).cuda())

        ans_rule_embed = rule_seqs.clone()
        ans_rule_embed[:, 1:] = ans_rule_embed[:, :-1]
        ans_rule_embed[:, 0] = 0
        ans_rule_embed = self.rule(Variable(ans_rule_embed).cuda())
        ans_lex_embed = trg_seqs[1].clone()
        ans_lex_embed[:, 1:] = ans_lex_embed[:, :-1]
        ans_lex_embed[:, 0] = 0
        ans_lex_embed = self.lexicon(Variable(ans_lex_embed).cuda())

        '''
        anc_embed = self.lexicon(Variable(ancestors[0]).cuda())
        anc_hidden, anc_init_hidden = self.encode_anc(anc_embed, anc_lengths, trg_lengths, batch_size, src_last_hidden, 'anc_')

        anc_syn_embed = torch.cat((self.constituent(Variable(ancestors[1]).cuda()), self.rule(Variable(ancestors[2]).cuda())), dim=2)
        anc_syn_hidden, anc_syn_init_hidden = self.encode_anc(anc_syn_embed, anc_lengths, trg_lengths, batch_size, src_last_hidden, 'anc_syn_')
        '''

        ans_embed = torch.cat((ans_nt, ans_lex_embed, ans_rule_embed), dim=2)
        #tree_input = torch.cat((anc_lex1, anc_lex2), dim=2)
        
        trg_l = max(trg_lengths)
        batch_ind = torch.arange(batch_size).long().cuda()
        if self.lex_level == 0:
            decoder_outputs = Variable(torch.FloatTensor(batch_size, trg_l, self.hidden_size * 2 + self.context_size * 2).cuda())
            context = self.attention((leaf_init_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psc_hidden.size(1), psn_lengths) 
        else:
            decoder_outputs = Variable(torch.FloatTensor(batch_size, trg_l, self.decoder_output_size).cuda())
            #context = self.attention((torch.cat((leaf_init_hidden[0], anc_init_hidden[0]), dim=2), None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths) 
            context = self.attention((torch.cat((decoder_hidden[0], leaf_init_hidden[0]), dim=2), None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths) 
        #pos_embed = torch.cat((self.depth(Variable(positions[0]).cuda()), self.breadth(Variable(positions[1] / 2).cuda())), dim=2)
        for step in range(trg_l):
            #decoder_input = torch.cat((ans_embed[:, step, :].unsqueeze(1), tree_input[:, step, :].unsqueeze(1)), dim=2)
            decoder_input = ans_embed[:, step, :].unsqueeze(1)
            #rule_decoder_input = ans_rule_embed[:, step, :].unsqueeze(1)
            if self.lex_level >= 2:
            #if False:
                leaf_select = torch.cuda.LongTensor([x[step].item() for x in _leaf_indices])
                #lex_select = torch.cuda.LongTensor([x[step].item() for x in _lex_indices])
                #decoder_input = torch.cat((decoder_input, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1), lex_decoder_hidden[batch_ind, lex_select, :].unsqueeze(1)), dim=2)
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                #rule_decoder_output, rule_decoder_hidden = self.rule_decoder(rule_decoder_input, rule_decoder_hidden)
                #dec_cat_output = torch.cat((decoder_output, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1), lex_decoder_hidden[batch_ind, lex_select, :].unsqueeze(1)), dim=2)
                #dec_cat_output = torch.cat((decoder_output, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1), anc_hidden[:, step, :].unsqueeze(1), anc_syn_hidden[:, step, :].unsqueeze(1)), dim=2)
                dec_cat_output = torch.cat((decoder_output, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1)), dim=2)
                decoder_outputs[:, step, :self.hidden_size*2] = dec_cat_output.squeeze(1)
                decoder_outputs[:, step, -self.context_size*2:] = context.squeeze(1)
                #decoder_outputs[:, step, self.hidden_size*3+self.context_size*2:] = tree_input[:, step, :]
                #context = self.attention((dec_cat_output[:, :, self.hidden_size:-self.hidden_size].transpose(0, 1), None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths) 
                context = self.attention((dec_cat_output.transpose(0, 1)[:, :, :self.hidden_size*2], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths) 
            else:
                leaf_select = torch.cuda.LongTensor([x[step].item() for x in _leaf_indices])
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                dec_cat_output = torch.cat((decoder_output, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1)), dim=2)
                decoder_outputs[:, step, :self.hidden_size * 2] = dec_cat_output.squeeze(1)
                decoder_outputs[:, step, self.hidden_size * 2:] = context.squeeze(1)
                context = self.attention((dec_cat_output.transpose(0, 1), None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths) 

        return decoder_outputs, ans_embed

    def masked_loss(self, logits, target, lengths, mask, rule_select=None, is_word=False):
        batch_size = logits.size(0)
        max_len = lengths.data.max()
        l_mask  = torch.arange(max_len).long().cuda().expand(batch_size, max_len) < lengths.data.expand(max_len, batch_size).transpose(0, 1)
        if rule_select is not None:
            rule_select[~l_mask.expand(self.rule_vocab_size, batch_size, max_len).permute(1, 2, 0)] = True
            '''
            _logits = torch.zeros_like(logits)
            _logits[rule_select] = logits[rule_select]
            _logits[~rule_select] = -np.inf
            logits = _logits
            '''
            _logits = logits.clone()
            _logits[~rule_select] = -10e8
            logits = _logits
        '''
        logits_flat = logits.view(-1, logits.size(-1))
        log_probs_flat = F.log_softmax(logits_flat, dim=1)
        '''
        log_probs_flat = F.log_softmax(logits, dim=2).view(-1, logits.size(-1))
        target_flat = target.contiguous().view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*target.size())
        _mask = Variable(l_mask * mask)
        losses = losses * _mask.float()
        if is_word:
            loss = (losses[:, 1:].sum() / _mask[:, 1:].float().sum()) + (losses[:, 0]).sum() / losses.size(0)
        else:
            loss = losses.sum() / _mask.float().sum()
            #loss = (losses.sum(1) / _mask.float().sum(1)).sum() / losses.size(0)
        #loss = losses.sum() / batch_size
        return loss

    def loss(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, word_mask, rule_mask, positions, ancestors, anc_lengths):
        decoder_outputs, tree_input = self.forward(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, positions, ancestors, anc_lengths)
        batch_size, trg_len = trg_seqs[0].size(0), trg_seqs[0].size(1)
        if self.lex_level == 0:
            words = self.lex_dist(self.lex_out(F.tanh(self.lex_hidden_out(decoder_outputs))))
            rules = self.rule_dist(F.tanh(self.rule_hidden_out(decoder_outputs)))
        elif self.lex_level >= 2:
            tags = self.constituent(Variable(trg_seqs[2]).cuda())
            words = self.lex_dist(self.lex_hidden_out(decoder_outputs))
            #words[:, 0, :] = self.lex_dist(self.top_lex_hidden_out(decoder_outputs[:, 0, :]))
            words_embed = self.lexicon(Variable(trg_seqs[1]).cuda())
            #nts = self.nt_dist(self.nt_hidden_out(torch.cat((decoder_outputs[:, :, :self.hidden_size * 2], words_embed), dim=2)))
            nts = self.nt_hidden_out(decoder_outputs[:, :, :self.hidden_size * 2])
            #rules = self.rule_dist(self.rule_hidden_out(torch.cat((decoder_outputs[:, :, :self.hidden_size*2], tags, words_embed), dim=2)))
            rules = self.rule_dist(self.rule_hidden_out(torch.cat((decoder_outputs[:, :, :self.hidden_size*2], tags), dim=2)))

        #word_loss = self.masked_loss(words, Variable(trg_seqs[1]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), word_mask.cuda().byte(), is_word=True)
        word_loss = self.masked_loss(words, Variable(trg_seqs[1]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), word_mask.cuda().byte())
        if self.lex_level == 0:
            rule_loss = self.masked_loss(rules, Variable(rule_seqs).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), rule_mask.cuda().byte())
        else:
            rule_select = self.rules_by_nt[trg_seqs[2].view(-1).cuda()].view(batch_size, trg_len, -1)
            rule_select[:, :, 1] = True
            rule_loss = self.masked_loss(rules, Variable(rule_seqs).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), rule_mask.cuda().byte(), rule_select)
            #rule_loss = self.masked_loss(rules, Variable(rule_seqs).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), rule_mask.cuda().byte())
        if self.lex_level == 0:
            nt_loss = 0
        else:
            nt_loss = self.masked_loss(nts, Variable(trg_seqs[2]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), rule_mask.cuda().byte())
            #nt_loss = self.masked_loss(nts, Variable(trg_seqs[2]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), torch.ones_like(rule_mask).cuda().byte())
        return word_loss, nt_loss, rule_loss

    def generate(self, src_seqs, src_lengths, psn_seqs, psn_lengths, indices, max_len, beam_size, top_k, slots=None):
        self.syn_weight = 1

        src_embed = self.lexicon(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)

        psn_lengths, perm_idx = torch.LongTensor(psn_lengths).sort(0, descending=True)
        psn_embed = self.lexicon(Variable(psn_seqs).cuda())
        psn_embed = psn_embed[perm_idx.cuda()]
        packed_input = pack_padded_sequence(psn_embed, psn_lengths.numpy(), batch_first=True)
        psn_output, psn_last_hidden = self.encoder(packed_input)
        psn_hidden, _ = pad_packed_sequence(psn_output, batch_first=True)
        psn_hidden = psn_hidden[perm_idx.sort()[1].cuda()]
        psn_lengths = psn_lengths[perm_idx.sort()[1]]

        decoder_hidden = self.init_hidden(src_last_hidden) 
        leaf_decoder_hidden = self.init_hidden(src_last_hidden, 'leaf_') 
        if self.lex_level == 0:
            context = self.attention((leaf_decoder_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths) 
        elif self.lex_level >= 2:
            #lex_decoder_hidden = self.init_hidden(src_last_hidden, 'lex_')    
            anc_decoder_hidden = self.init_hidden(src_last_hidden, 'anc_')
            syn_decoder_hidden = self.init_hidden(src_last_hidden, 'anc_syn_')
            context = self.attention((torch.cat((decoder_hidden[0], leaf_decoder_hidden[0]), dim=2), None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths) 
            '''
            leaf_decoder_hidden = leaf_init_hidden[0].transpose(0, 1)
            lex_decoder_hidden = lex_init_hidden[0].transpose(0, 1)
            '''

        batch_size = src_embed.size(0)
        assert batch_size == 1
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        #context = self.attention(decoder_hidden, src_hidden, src_hidden.size(1), src_lengths) 
        if self.lex_level >= 2:
            decoder_input = Variable(torch.zeros(batch_size, 1, self.decoder_input_size)).cuda()
            decoder_input[:, :, :self.nt_input_size] = self.constituent(Variable(torch.cuda.LongTensor([self.ROOT]))).unsqueeze(1)
        else:
            decoder_input = Variable(torch.zeros(batch_size, 1, self.decoder_input_size)).cuda()
            decoder_input[:, :, :self.nt_input_size] = self.constituent(Variable(torch.cuda.LongTensor([self.ROOT]))).unsqueeze(1)
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        #rule_decoder_output, rule_decoder_hidden = self.rule_decoder(rule_decoder_input, rule_decoder_hidden)

        word_beam = torch.zeros(beam_size, max_len).long().cuda()
        rule_beam = torch.zeros(beam_size, max_len).long().cuda()
        nt_beam = torch.zeros(beam_size, max_len).long().cuda()
        word_count = torch.zeros(beam_size, top_k).long().cuda()
        rule_count = torch.zeros(beam_size, top_k).long().cuda()
        states_num = 1
        if self.lex_level == 0:
            decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0], context), dim=2)
            word_logits = self.lex_dist(self.lex_out(F.tanh(self.lex_hidden_out(decoder_output))))
            word_logprobs, word_argtop = torch.topk(F.log_softmax(word_logits.squeeze(1), dim=1), beam_size, dim=1)
            rule_logits = self.rule_dist(F.tanh(self.rule_hidden_out(decoder_output)))
            rule_logprobs, rule_argtop = torch.topk(F.log_softmax(rule_logits.squeeze(1), dim=1), beam_size, dim=1)
            rule_argtop = rule_argtop.squeeze(0)
            rule_beam[:, 0] = rule_argtop.data[rule_argtop.data % beam_size]
            rule_beam_probs = rule_logprobs.squeeze(0).data[rule_argtop.data % beam_size]
            word_beam_probs = torch.zeros_like(rule_beam_probs)
            rule_count.fill_(1)
            word_count.fill_(0)
        else:
            decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0], context), dim=2)
            #decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0], anc_decoder_hidden[0], syn_decoder_hidden[0], context), dim=2)
            self.nt_cand_num = 1
            self.rule_cand_num = 1
            self.lex_cand_num = beam_size
            if slots is not None:
                slots = torch.cuda.LongTensor(slots)
                states_num = 2 ** len(slots)
                slots_num = slots.size(0)
                word_count = torch.zeros(beam_size, states_num).long().cuda()
                rule_count = torch.zeros(beam_size, states_num).long().cuda()

                word_beam = torch.zeros(beam_size, states_num, max_len).long().cuda()
                rule_beam = torch.zeros(beam_size, states_num, max_len).long().cuda()
                nt_beam = torch.zeros(beam_size, states_num, max_len).long().cuda()
                word_beam_probs = torch.zeros(beam_size, states_num).float().fill_(-np.inf).cuda()
                rule_beam_probs = torch.zeros(beam_size, states_num).float().fill_(-np.inf).cuda()
                nt_beam_probs = torch.zeros(beam_size, states_num).float().fill_(-np.inf).cuda()
                word_count = torch.zeros(beam_size, states_num).long().cuda()
                rule_count = torch.zeros(beam_size, states_num).long().cuda()

                total_logprobs = torch.zeros(self.lex_cand_num + slots_num, self.nt_cand_num, self.rule_cand_num).cuda()

                word_logits = self.lex_dist(self.lex_out(F.tanh(self.lex_hidden_out(decoder_output))))
                word_logits = F.log_softmax(word_logits.squeeze(1), dim=1)
                slots_logprobs = word_logits[:, slots]
                word_logits[:, slots] = -np.inf
                word_logprobs, word_argtop = torch.topk(word_logits, self.lex_cand_num, dim=1)
                word_argtop = Variable(torch.cat((word_argtop.data, slots.unsqueeze(0)), dim=1))
                word_logprobs = Variable(torch.cat((word_logprobs.data, slots_logprobs.data), dim=1))

                total_logprobs += word_logprobs.data.expand(self.rule_cand_num, self.nt_cand_num, self.lex_cand_num + slots_num).transpose(0, 2)

                decoder_output = decoder_output.squeeze(0).expand(self.lex_cand_num + slots_num, self.decoder_output_size)
                word_embed = self.lexicon(word_argtop).squeeze(0)
                nt_logits = self.nt_dist(self.nt_out(F.tanh(self.nt_hidden_out(torch.cat((decoder_output[:, self.hidden_size], decoder_output[:, self.hidden_size * 3:self.hidden_size * 4], word_embed), dim=1)))))
                nt_logprobs, nt_argtop = torch.topk(F.log_softmax(nt_logits, dim=1), self.nt_cand_num, dim=1)
                total_logprobs += nt_logprobs.data.expand(self.rule_cand_num, self.lex_cand_num + slots_num, self.nt_cand_num).transpose(0, 1).transpose(1, 2)

                tag_embed = self.constituent(nt_argtop)
                word_embed = word_embed.expand(self.nt_cand_num, self.lex_cand_num + slots_num, self.lex_input_size).transpose(0, 1)
                decoder_output = decoder_output.expand(self.nt_cand_num, self.lex_cand_num + slots_num, self.decoder_output_size).transpose(0, 1)
                rule_logits = self.rule_dist(self.rule_out(F.tanh(self.rule_hidden_out(torch.cat((decoder_output[:, :, :self.hidden_size], decoder_output[:, :, self.hidden_size * 3:self.hidden_size * 4], tag_embed, word_embed), dim=2)))))
                rule_select = self.rules_by_nt[nt_argtop.data.view(-1)].view(self.lex_cand_num + slots_num, self.nt_cand_num, -1)
                rule_logits[~rule_select] = -np.inf
                rule_logprobs, rule_argtop = torch.topk(F.log_softmax(rule_logits, dim=2), self.rule_cand_num, dim=2)
                total_logprobs += rule_logprobs.data

                state_codes = np.zeros(self.lex_cand_num + slots_num, dtype=np.int32)
                for x in range(slots_num):
                    state_codes[self.lex_cand_num+x] = 2 ** x 
                null_ind = torch.cuda.ByteTensor(beam_size, states_num)
                for x in range(states_num):
                    if np.where(state_codes == x)[0].any():
                        #state_going = torch.from_numpy(np.where(state_codes == x)[0]).cuda()
                        state_going = torch.cuda.ByteTensor((state_codes == x).tolist()).expand(self.nt_cand_num, self.rule_cand_num, self.lex_cand_num + slots_num).permute(2, 0, 1)
                        state_going_num = np.sum(state_codes == x)
                        _total_logprobs = total_logprobs.clone()
                        _total_logprobs[~state_going] = -np.inf
                        logprobs, argtop = torch.topk(_total_logprobs.view(-1), beam_size, dim=0)
                        null_ind[:, x] = logprobs == -np.inf
                        argtop_ind = torch.cuda.LongTensor(beam_size, 3)
                        argtop_ind[:, 0] = argtop / (self.nt_cand_num * self.rule_cand_num)
                        argtop_ind[:, 1] = (argtop % (self.nt_cand_num * self.rule_cand_num)) / self.rule_cand_num
                        argtop_ind[:, 2] = (argtop % (self.nt_cand_num * self.rule_cand_num)) % self.rule_cand_num
                        word_beam[:, x, 0] = word_argtop.squeeze(0).data[argtop_ind[:, 0]]
                        word_beam_probs[:, x] = word_logprobs.squeeze(0).data[argtop_ind[:, 0]]
                        nt_beam[:, x, 0] = nt_argtop.data[argtop_ind[:, 0], argtop_ind[:, 1]]
                        nt_beam_probs[:, x] = nt_logprobs.data[argtop_ind[:, 0], argtop_ind[:, 1]]
                        rule_beam[:, x, 0] = rule_argtop.data[argtop_ind[:, 0], argtop_ind[:, 1], argtop_ind[:, 2]]
                        rule_beam_probs[:, x] = rule_logprobs.data[argtop_ind[:, 0], argtop_ind[:, 1], argtop_ind[:, 2]]
                word_beam_probs[null_ind] = -np.inf
                nt_beam_probs[null_ind] = -np.inf
                rule_beam_probs[null_ind] = -np.inf

                self.lex_cand_num = 20
                state_codes = np.zeros(beam_size * states_num, dtype=np.int32)
                trans_state_codes = np.zeros(self.lex_cand_num + slots_num, dtype=np.int32)
                for x in range(slots_num):
                    trans_state_codes[self.lex_cand_num+x] = 2 ** x
                for x in range(states_num):
                    state_codes[x*beam_size:(x+1)*beam_size] = x
            else:
                total_logprobs = torch.zeros(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num).cuda()

                #word_logits = self.lex_dist(self.lex_hidden_out(decoder_output))
                word_logits = self.lex_dist(self.top_lex_hidden_out(decoder_output))
                word_logprobs, word_argtop = torch.topk(F.log_softmax(word_logits.squeeze(1), dim=1), self.lex_cand_num, dim=1)
                total_logprobs += word_logprobs.data.expand(self.rule_cand_num, self.nt_cand_num, self.lex_cand_num).transpose(0, 2)

                decoder_output = decoder_output.squeeze(0).expand(self.lex_cand_num, self.decoder_output_size)
                word_embed = self.lexicon(word_argtop).squeeze(0)
                nt_logits = self.nt_dist(self.nt_hidden_out(torch.cat((decoder_output[:, :self.hidden_size * 2], word_embed), dim=1)))
                nt_logprobs, nt_argtop = torch.topk(F.log_softmax(nt_logits, dim=1), self.nt_cand_num, dim=1)
                total_logprobs += self.syn_weight * nt_logprobs.data.expand(self.rule_cand_num, self.lex_cand_num, self.nt_cand_num).transpose(0, 1).transpose(1, 2)

                tag_embed = self.constituent(nt_argtop)
                word_embed = word_embed.expand(self.nt_cand_num, self.lex_cand_num, self.lex_input_size).transpose(0, 1)
                decoder_output = decoder_output.expand(self.nt_cand_num, self.lex_cand_num, self.decoder_output_size).transpose(0, 1)
                rule_logits = self.rule_dist(self.rule_hidden_out(torch.cat((decoder_output[:, :, :self.hidden_size * 2], tag_embed, word_embed), dim=2)))
                rule_select = self.rules_by_nt[nt_argtop.data.view(-1)].view(self.lex_cand_num, self.nt_cand_num, -1)
                rule_logits[~rule_select] = -np.inf
                rule_logits = F.log_softmax(rule_logits, dim=2)
                rule_logprobs, rule_argtop = torch.topk(rule_logits, self.rule_cand_num, dim=2)
                total_logprobs += self.syn_weight * rule_logprobs.data

                logprobs, argtop = torch.topk(total_logprobs.view(-1), beam_size, dim=0)
                argtop_ind = torch.cuda.LongTensor(beam_size, 3)
                argtop_ind[:, 0] = argtop / (self.nt_cand_num * self.rule_cand_num)
                argtop_ind[:, 1] = (argtop % (self.nt_cand_num * self.rule_cand_num)) / self.rule_cand_num
                argtop_ind[:, 2] = (argtop % (self.nt_cand_num * self.rule_cand_num)) % self.rule_cand_num
                word_beam[:, 0] = word_argtop.squeeze(0).data[argtop_ind[:, 0]]
                word_beam_probs = word_logprobs.squeeze(0).data[argtop_ind[:, 0]]
                nt_beam[:, 0] = nt_argtop.data[argtop_ind[:, 0], argtop_ind[:, 1]]
                nt_beam_probs = nt_logprobs.data[argtop_ind[:, 0], argtop_ind[:, 1]]
                rule_beam[:, 0] = rule_argtop.data[argtop_ind[:, 0], argtop_ind[:, 1], argtop_ind[:, 2]]
                rule_beam_probs = rule_logprobs.data[argtop_ind[:, 0], argtop_ind[:, 1], argtop_ind[:, 2]]
            word_count.fill_(1)
            rule_count.fill_(1)
        #hidden = (decoder_hidden[0].clone(), decoder_hidden[1].clone())
        def expand_by_beam(decoder_hidden):
            return (decoder_hidden[0].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1),
                    decoder_hidden[1].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1))

        def expand_by_state_and_beam(decoder_hidden):
            return (decoder_hidden[0].expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1),
                    decoder_hidden[1].expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1))

        if slots is None:
            decoder_hidden = (decoder_hidden[0].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1),
                              decoder_hidden[1].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1))
            src_last_hidden = (src_last_hidden[0].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1),
                               src_last_hidden[1].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1))
            leaf_decoder_hidden = (leaf_decoder_hidden[0].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1),
                                   leaf_decoder_hidden[1].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1))
            if self.lex_level >= 2:
                anc_decoder_hidden = expand_by_beam(anc_decoder_hidden) 
                anc_decoder_input = self.lexicon(word_beam[:, 0]).unsqueeze(1)
                _, anc_decoder_hidden = self.anc_decoder(anc_decoder_input, anc_decoder_hidden)
                syn_decoder_hidden = expand_by_beam(syn_decoder_hidden) 
                syn_decoder_input = torch.cat((self.constituent(nt_beam[:, 0]), self.rule(rule_beam[:, 0])), dim=1).unsqueeze(1)
                _, syn_decoder_hidden = self.anc_syn_decoder(syn_decoder_input, syn_decoder_hidden)
            src_hidden = src_hidden.expand(beam_size, src_hidden.size(1), self.hidden_size)
            psn_hidden = psn_hidden.expand(beam_size, psn_hidden.size(1), self.hidden_size)
        else:
            decoder_hidden = (decoder_hidden[0].unsqueeze(2).expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1), 
                              decoder_hidden[1].unsqueeze(2).expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1)) 
            src_last_hidden = (src_last_hidden[0].unsqueeze(2).expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1), 
                               src_last_hidden[1].unsqueeze(2).expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1)) 
            leaf_decoder_hidden = (leaf_decoder_hidden[0].expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1),
                                   leaf_decoder_hidden[1].expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1))
            if self.lex_level >= 2:
                anc_decoder_hidden = expand_by_state_and_beam(anc_decoder_hidden)
                anc_decoder_input = self.lexicon(word_beam[:, :, 0]).transpose(0, 1).contiguous().view(-1, self.lex_input_size).unsqueeze(1)
                _, anc_decoder_hidden = self.anc_decoder(anc_decoder_input, anc_decoder_hidden)
                syn_decoder_hidden = expand_by_state_and_beam(syn_decoder_hidden)
                syn_decoder_input = torch.cat((self.constituent(nt_beam[:, :, 0]), self.rule(rule_beam[:, :, 0])), dim=2).transpose(0, 1).contiguous().view(states_num * beam_size, -1).unsqueeze(1)
                _, syn_decoder_hidden = self.anc_syn_decoder(syn_decoder_input, syn_decoder_hidden)
            #decoder_input = self.embed(beam[:, :, 0].contiguous().view(-1)).unsqueeze(1)
            src_hidden = src_hidden.expand(states_num * beam_size, src_hidden.size(1), self.hidden_size)
            psn_hidden = psn_hidden.expand(states_num * beam_size, psn_hidden.size(1), self.hidden_size)

        leaves = torch.cuda.LongTensor([[[0, 0], [0, 0], [0, 0]] for x in range(beam_size)])
        lexicons = torch.cuda.LongTensor([[[0, 0], [0, 0], [0, 0]] for x in range(beam_size)])
        tree_dict = {}
        node_ind = 0
        if slots is None:
            final = np.array([None for x in range(beam_size)])
            current = np.array([None for x in range(beam_size)])
            for y in range(beam_size):
                rule = self.rules_by_id[rule_beam[y, 0]]
                if self.lex_level == 0:
                    current[y] = Node('{}__{}__{}'.format('ROOT', word_beam[y, 0], rule))
                    current[y].id = node_ind
                    node_ind += 1
                else:
                    tag = self.nt_by_id[nt_beam[y, 0]]
                    current[y] = Node('{}__{}__{} [{}]'.format('ROOT', word_beam[y, 0], rule, rule.split().index(tag)))
                    current[y].id = node_ind
                    node_ind += 1
                tree_dict[current[y].id] = (
                                            (anc_decoder_hidden[0][:, y, :].clone(), anc_decoder_hidden[1][:, y, :].clone()),
                                            (syn_decoder_hidden[0][:, y, :].clone(), syn_decoder_hidden[1][:, y, :].clone())
                                           )
        else:
            final = np.array([[None for x in range(beam_size)] for y in range(states_num)])
            current = np.array([[None for x in range(beam_size)] for y in range(states_num)])
            for x in range(states_num):
                for y in range(beam_size):
                    rule = self.rules_by_id[rule_beam[y, x, 0]]
                    if rule == '':
                        current[x, y] = Node('')
                        continue
                    if self.lex_level == 0:
                        current[x, y] = Node('{}__{}__{}'.format('ROOT', word_beam[y, x, 0], rule))
                        current[x, y].id = node_ind
                        node_ind += 1
                    else:
                        tag = self.nt_by_id[nt_beam[y, x, 0]]
                        current[x, y] = Node('{}__{}__{} [{}]'.format('ROOT', word_beam[y, x, 0], rule, rule.split().index(tag)))
                        current[x, y].id = node_ind
                        node_ind += 1
                    tree_dict[current[x, y].id] = (
                                                   (anc_decoder_hidden[0][:, x*beam_size+y, :].clone(), anc_decoder_hidden[1][:, x*beam_size+y, :].clone()),
                                                   (syn_decoder_hidden[0][:, x*beam_size+y, :].clone(), syn_decoder_hidden[1][:, x*beam_size+y, :].clone())
                                                  )

        def inheritance(rule):
            return literal_eval(rule[rule.find('['):rule.find(']')+1])

        for t in range(max_len-1):
            if slots is None:
                self.nt_cand_num = 2
                self.rule_cand_num = 2
                self.lex_cand_num = 20
            else:
                self.nt_cand_num = 2
                self.rule_cand_num = 2
                self.lex_cand_num = 20
            if slots is None:
                ans_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
                ans_par_rule = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
                ans_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
                word_mask = torch.cuda.ByteTensor(beam_size, top_k)
                word_mask.fill_(True)
                rule_mask = torch.cuda.ByteTensor(beam_size, top_k)
                rule_mask.fill_(True)
                curr_nt = np.array(['' for x in range(beam_size)], dtype=object)

                for x in range(beam_size):
                    if current[x] is None:
                        rule_mask[x] = False
                        word_mask[x] = False
                    else:
                        par_nt, par_lex, rule = current[x].name.split('__')
                        curr_nt[x] = rule.split()[len(current[x].children)]
                        if self.lex_level == 0:
                            if not self.preterminal[curr_nt[x]]:
                                rule_count[x] += 1
                                word_mask[x] = False
                            else:
                                word_count[x] += 1
                                rule_mask[x] = False
                        else:
                            if not self.preterminal[curr_nt[x]]:
                                rule_count[x] += 1
                            else:
                                rule_mask[x] = False
                            if len(current[x].children) not in inheritance(rule):
                                word_count[x] += 1
                            else:
                                word_mask[x] = False
                        ans_nt[x] = self.nt_dictionary[curr_nt[x]]
                        ans_par_rule[x] = self.rule_dictionary['RULE: {}'.format(rule[:rule.find('[')-1])]
                        ans_lex[x] = int(par_lex)
            else:
                ans_nt = Variable(torch.cuda.LongTensor(beam_size, states_num)).fill_(0)
                ans_par_rule = Variable(torch.cuda.LongTensor(beam_size, states_num)).fill_(0)
                anc_lex1 = Variable(torch.cuda.LongTensor(beam_size, states_num)).fill_(0)
                anc_lex2 = Variable(torch.cuda.LongTensor(beam_size, states_num)).fill_(0)
                word_mask = torch.cuda.ByteTensor(beam_size, states_num, self.lex_cand_num + slots_num)
                word_mask.fill_(True)
                rule_mask = torch.cuda.ByteTensor(beam_size, states_num, self.lex_cand_num + slots_num)
                rule_mask.fill_(True)
                curr_nt = np.array([['' for x in range(beam_size)] for y in range(states_num)], dtype=object)

                for x in range(beam_size):
                    for y in range(states_num):
                        if current[y, x] is None or current[y, x].name == '':
                            rule_mask[x, y] = False
                            word_mask[x, y] = False
                        else:
                            par_nt, par_lex, rule = current[y, x].name.split('__')
                            curr_nt[y, x] = rule.split()[len(current[y, x].children)]
                            if curr_nt[y, x][0] == '[':
                                curr_nt[y, x] = 'S'
                            if self.lex_level == 0:
                                if not self.preterminal[curr_nt[y, x]]:
                                    rule_count[x, y] += 1
                                    word_mask[x, y] = False
                                else:
                                    word_count[x, y] += 1
                                    rule_mask[x, y] = False
                            else:
                                if not self.preterminal[curr_nt[y, x]]:
                                    rule_count[x, y] += 1
                                else:
                                    rule_mask[x, y] = False
                                if len(current[y, x].children) not in inheritance(rule):
                                    word_count[x, y] += 1
                                else:
                                    word_mask[x, y] = False
                            ans_nt[x, y] = self.nt_dictionary[curr_nt[y, x]]
                            if rule[0] != ' ':
                                ans_par_rule[x, y] = self.rule_dictionary['RULE: {}'.format(rule[:rule.find('[')-1])]
                            else:
                                ans_par_rule[x, y] = self.rule_dictionary['RULE: S']
                            '''
                            ancestors = current[y, x].ancestors
                            if len(ancestors) >= 2:
                                anc_lex1[x, y] = int(ancestors[0].name.split('__')[1])
                            if len(ancestors) >= 3:
                                anc_lex2[x, y] = int(ancestors[1].name.split('__')[1])
                            '''
            ans_nt = self.constituent(ans_nt)
            ans_par_rule = self.rule(ans_par_rule)
            ans_lex = self.lexicon(ans_lex)
            '''
            anc_lex1 = self.lexicon(anc_lex1)
            anc_lex2 = self.lexicon(anc_lex2)
            '''
            if slots is None:
                #tree_input = torch.cat((anc_lex1, anc_lex2), dim=1).unsqueeze(1)
                ans_embed = torch.cat((ans_nt, ans_lex, ans_par_rule), dim=1)
                decoder_input = ans_embed.unsqueeze(1)
            else:
                #tree_input = torch.cat((anc_lex1, anc_lex2), dim=2).transpose(0, 1).contiguous().view(-1, self.tree_input_size).unsqueeze(1)
                ans_embed = torch.cat((ans_nt, ans_par_rule), dim=2)
                decoder_input = ans_embed.transpose(0, 1).contiguous().view(-1, self.decoder_input_size).unsqueeze(1)
               
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            if self.lex_level == 0:
                context = self.attention((leaf_decoder_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths) 
            elif self.lex_level >= 2:
                #decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0].transpose(0, 1), anc_decoder_hidden[0].transpose(0, 1), syn_decoder_hidden[0].transpose(0, 1)), dim=2)
                decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0].transpose(0, 1)), dim=2)
                context = self.attention((decoder_output.transpose(0, 1)[:, :, :self.hidden_size*2], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths) 
                decoder_output = torch.cat((decoder_output, context), dim=2)
            dup_mask = ~(word_mask | rule_mask)

            if self.lex_level == 0:
                decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0].transpose(0, 1), context), dim=2)
                word_logits = self.lex_dist(self.lex_out(F.tanh(self.lex_hidden_out((decoder_output)))))
                word_logprobs, word_argtop = torch.topk(F.log_softmax(word_logits.squeeze(1), dim=1), top_k, dim=1)
                word_beam_logprobs = ((word_beam_probs).expand(top_k, beam_size).transpose(0, 1) + word_logprobs.data * word_mask.float()) 

                rule_logits = self.rule_dist(F.tanh(self.rule_hidden_out(decoder_output)))
                dup_mask[:, 0] = 0
                word_beam_logprobs[dup_mask] = -np.inf
                rule_logprobs, rule_argtop = torch.topk(F.log_softmax(rule_logits.squeeze(1), dim=1), top_k, dim=1)
                rule_beam_logprobs = ((rule_beam_probs).expand(top_k, beam_size).transpose(0, 1) + rule_logprobs.data * rule_mask.float())
                rule_beam_logprobs[dup_mask] = -np.inf
                total_logprobs = word_beam_logprobs / word_count.float() + rule_beam_logprobs / rule_count.float()
                #total_logprobs = (word_beam_logprobs + rule_beam_logprobs) / (t + 1)
                #total_logprobs = word_beam_logprobs / np.sqrt(t + 1) + rule_beam_logprobs / (t + 1)
                best_probs, best_args = total_logprobs.view(-1).topk(beam_size)
            else:
                if slots is None:
                    total_logprobs = torch.zeros(beam_size, self.lex_cand_num, self.nt_cand_num, self.rule_cand_num).cuda()
                    decoder_output = decoder_output.squeeze(1)
                    word_mask = word_mask[:, 0].expand(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num, beam_size).permute(3, 0, 1, 2).float()
                    _word_count = word_count[:, 0].expand(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num, beam_size).permute(3, 0, 1, 2).float()
                    rule_mask = rule_mask[:, 0].expand(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num, beam_size).permute(3, 0, 1, 2).float()
                    _rule_count = rule_count[:, 0].expand(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num, beam_size).permute(3, 0, 1, 2).float()

                    word_logits = self.lex_dist(self.lex_hidden_out(decoder_output))
                    word_logprobs, word_argtop = torch.topk(F.log_softmax(word_logits, dim=1), self.lex_cand_num, dim=1)
                    word_beam_logprobs = word_logprobs.data.expand(self.nt_cand_num, self.rule_cand_num, beam_size, self.lex_cand_num).permute(2, 3, 0, 1)

                    dup_mask = torch.ByteTensor(beam_size, self.lex_cand_num, self.nt_cand_num, self.rule_cand_num).fill_(False)
                    for x in range(beam_size):
                        if rule_mask[x, 0, 0, 0] == 0:
                            if word_mask[x, 0, 0, 0] == 0:
                                dup_mask[x, :, :, :] = True
                                dup_mask[x, 0, 0, 0] = False
                                if current[x] is not None:
                                    inherit = current[x].name.split('__')[1]
                                    word_argtop[x] = int(inherit)
                            else:
                                dup_mask[x, :, :, :] = True
                                dup_mask[x, :, 0, 0] = False
                        elif word_mask[x, 0, 0, 0] == 0:
                            dup_mask[x, :, :, :] = True
                            dup_mask[x, 0, :, :] = False
                            inherit = current[x].name.split('__')[1]
                            word_argtop[x] = int(inherit)
                    dup_mask = dup_mask.cuda()

                    word_embed = self.lexicon(word_argtop)
                    decoder_output = decoder_output.expand(self.lex_cand_num, beam_size, self.decoder_output_size).transpose(0, 1)
                    nt_logits = self.nt_dist(self.nt_hidden_out(torch.cat((decoder_output[:, :, :self.hidden_size * 2], word_embed), dim=2)))
                    nt_logits = F.log_softmax(nt_logits, dim=2)
                    nt_logits[:, :, 0] = -np.inf
                    nt_logits[:, :, 1] = -np.inf
                    nt_logprobs, nt_argtop = torch.topk(nt_logits, self.nt_cand_num, dim=2)
                    nt_beam_logprobs = nt_logprobs.data.expand(self.rule_cand_num, beam_size, self.lex_cand_num, self.nt_cand_num).permute(1, 2, 3, 0)

                    tag_embed = self.constituent(nt_argtop.view(beam_size * self.lex_cand_num, -1)).view(beam_size, self.lex_cand_num, self.nt_cand_num, -1)
                    decoder_output = decoder_output.expand(self.nt_cand_num, beam_size, self.lex_cand_num, self.decoder_output_size).permute(1, 2, 0, 3)
                    word_embed = word_embed.expand(self.nt_cand_num, beam_size, self.lex_cand_num, self.lex_input_size).permute(1, 2, 0, 3)
                    rule_logits = self.rule_dist(self.rule_hidden_out(torch.cat((decoder_output[:, :, :, :self.hidden_size * 2], tag_embed, word_embed), dim=3)))
                    rule_select = self.rules_by_nt[nt_argtop.data.view(-1)].view(beam_size, self.lex_cand_num, self.nt_cand_num, -1)
                    rule_logits[~rule_select] = -np.inf
                    rule_logits = F.log_softmax(rule_logits, dim=3)
                    rule_logprobs, rule_argtop = torch.topk(rule_logits, self.rule_cand_num, dim=3)
                    rule_beam_logprobs = rule_logprobs.data

                    word_beam_logprobs = word_beam_logprobs * word_mask + word_beam_probs.expand(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num, beam_size).permute(3, 0, 1, 2)
                    nt_beam_logprobs = nt_beam_logprobs * rule_mask + nt_beam_probs.expand(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num, beam_size).permute(3, 0, 1, 2)
                    rule_beam_logprobs = rule_beam_logprobs * rule_mask + rule_beam_probs.expand(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num, beam_size).permute(3, 0, 1, 2)

                    total_logprobs = word_beam_logprobs / _word_count + self.syn_weight * nt_beam_logprobs / _rule_count + self.syn_weight * rule_beam_logprobs / _rule_count
                    #total_logprobs = word_beam_logprobs / _word_count + top_word_beam_probs.expand(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num, beam_size).permute(3, 0, 1, 2) + self.syn_weight * nt_beam_logprobs / _rule_count + self.syn_weight * rule_beam_logprobs / _rule_count
                    total_logprobs[dup_mask] = -np.inf

                    logprobs, argtop = torch.topk(total_logprobs.view(-1), beam_size, dim=0)
                    argtop_ind = torch.cuda.LongTensor(beam_size, 4)
                    argtop_ind[:, 0] = argtop / (self.lex_cand_num * self.nt_cand_num * self.rule_cand_num)
                    argtop_ind[:, 1] = (argtop % (self.lex_cand_num * self.nt_cand_num * self.rule_cand_num)) / (self.nt_cand_num * self.rule_cand_num)
                    argtop_ind[:, 2] = ((argtop % (self.lex_cand_num * self.nt_cand_num * self.rule_cand_num)) % (self.nt_cand_num * self.rule_cand_num)) / self.rule_cand_num
                    argtop_ind[:, 3] = ((argtop % (self.lex_cand_num * self.nt_cand_num * self.rule_cand_num)) % (self.nt_cand_num * self.rule_cand_num)) % self.rule_cand_num
                else:
                    new_word_beam = word_beam.clone()
                    new_word_beam_probs = word_beam_probs.clone()
                    new_rule_beam = rule_beam.clone()
                    new_rule_beam_probs = rule_beam_probs.clone()
                    new_nt_beam = nt_beam.clone()
                    new_nt_beam_probs = nt_beam_probs.clone()

                    total_logprobs = torch.zeros(beam_size * states_num, self.lex_cand_num, self.nt_cand_num, self.rule_cand_num).cuda()
                    decoder_output = decoder_output.squeeze(1)
                    word_mask = word_mask.expand(self.nt_cand_num, self.rule_cand_num, beam_size, states_num, self.lex_cand_num + slots_num).permute(2, 3, 4, 0, 1).float()
                    _word_count = word_count.expand(self.lex_cand_num + slots_num, self.nt_cand_num, self.rule_cand_num, beam_size, states_num).permute(3, 4, 0, 1, 2).float()
                    rule_mask = rule_mask.expand(self.nt_cand_num, self.rule_cand_num, beam_size, states_num, self.lex_cand_num + slots_num).permute(2, 3, 4, 0, 1).float()
                    _rule_count = rule_count.expand(self.lex_cand_num + slots_num, self.nt_cand_num, self.rule_cand_num, beam_size, states_num).permute(3, 4, 0, 1, 2).float()

                    word_logits = self.lex_dist(self.lex_out(F.tanh(self.lex_hidden_out(decoder_output))))
                    word_logits = F.log_softmax(word_logits.squeeze(1), dim=1)
                    slots_logprobs = word_logits[:, slots]
                    word_logits[:, slots] = -np.inf
                    word_logprobs, word_argtop = torch.topk(word_logits, self.lex_cand_num, dim=1)
                    word_argtop = Variable(torch.cat((word_argtop.data, slots.expand(beam_size * states_num, slots_num)), dim=1))
                    word_logprobs = Variable(torch.cat((word_logprobs.data, slots_logprobs.data), dim=1))
                    word_beam_logprobs = word_logprobs.data.expand(self.nt_cand_num, self.rule_cand_num, beam_size * states_num, self.lex_cand_num + slots_num).permute(2, 3, 0, 1)
                    word_beam_logprobs = word_beam_logprobs.contiguous().view(states_num, beam_size, self.lex_cand_num + slots_num, self.nt_cand_num, self.rule_cand_num).permute(1, 0, 2, 3, 4)

                    dup_mask = torch.ByteTensor(states_num, beam_size, self.lex_cand_num + slots_num, self.nt_cand_num, self.rule_cand_num).fill_(False)
                    transition = np.bitwise_or(np.repeat(state_codes, self.lex_cand_num + slots_num), np.tile(trans_state_codes, states_num * beam_size)).reshape(states_num, beam_size, self.lex_cand_num + slots_num).swapaxes(0, 1)
                    for y in range(states_num):
                        for x in range(beam_size):
                            if rule_mask[x, y, 0, 0, 0] == 0:
                                if word_mask[x, y, 0, 0, 0] == 0:
                                    dup_mask[y, x, :, :, :] = True
                                    dup_mask[y, x, 0, 0, 0] = False
                                    if current[y, x] is not None and current[y, x].name != '':
                                        inherit = current[y, x].name.split('__')[1]
                                        word_argtop[y*beam_size+x] = int(inherit)
                                    transition[x, y, :] = y
                                else:
                                    dup_mask[y, x, :, :, :] = True
                                    dup_mask[y, x, :, 0, 0] = False
                            elif word_mask[x, y, 0, 0, 0] == 0:
                                dup_mask[y, x, :, :, :] = True
                                dup_mask[y, x, 0, :, :] = False
                                inherit = current[y, x].name.split('__')[1]
                                word_argtop[y*beam_size+x] = int(inherit)
                                transition[x, y, :] = y
                    dup_mask = dup_mask.cuda()

                    word_embed = self.lexicon(word_argtop)
                    decoder_output = decoder_output.expand(self.lex_cand_num + slots_num, beam_size * states_num, self.decoder_output_size).transpose(0, 1)
                    nt_logits = self.nt_dist(self.nt_out(F.tanh(self.nt_hidden_out(torch.cat((decoder_output[:, :, :self.hidden_size * 4], word_embed), dim=2)))))
                    nt_logits = F.log_softmax(nt_logits, dim=2)
                    nt_logits[:, :, 0] = -np.inf
                    nt_logits[:, :, 1] = -np.inf
                    nt_logprobs, nt_argtop = torch.topk(nt_logits, self.nt_cand_num, dim=2)
                    nt_beam_logprobs = nt_logprobs.data.expand(self.rule_cand_num, beam_size * states_num, self.lex_cand_num + slots_num, self.nt_cand_num).permute(1, 2, 3, 0)
                    nt_beam_logprobs = nt_beam_logprobs.contiguous().view(states_num, beam_size, self.lex_cand_num + slots_num, self.nt_cand_num, self.rule_cand_num).permute(1, 0, 2, 3, 4)

                    tag_embed = self.constituent(nt_argtop.view(beam_size * states_num * (self.lex_cand_num + slots_num), -1)).view(beam_size * states_num, self.lex_cand_num + slots_num, self.nt_cand_num, -1)
                    decoder_output = decoder_output.expand(self.nt_cand_num, beam_size * states_num, self.lex_cand_num + slots_num, self.decoder_output_size).permute(1, 2, 0, 3)
                    word_embed = word_embed.expand(self.nt_cand_num, beam_size * states_num, self.lex_cand_num + slots_num, self.lex_input_size).permute(1, 2, 0, 3)
                    rule_logits = self.rule_dist(self.rule_out(F.tanh(self.rule_hidden_out(torch.cat((decoder_output[:, :, :, :self.hidden_size * 4], tag_embed, word_embed), dim=3)))))
                    rule_logits = F.log_softmax(rule_logits, dim=3)
                    rule_select = self.rules_by_nt[nt_argtop.data.view(-1)].view(beam_size * states_num, self.lex_cand_num + slots_num, self.nt_cand_num, -1)
                    rule_logits[~rule_select] = -np.inf
                    rule_logits = F.log_softmax(rule_logits, dim=3)
                    rule_logprobs, rule_argtop = torch.topk(rule_logits, self.rule_cand_num, dim=3)
                    rule_beam_logprobs = rule_logprobs.data
                    rule_beam_logprobs = rule_beam_logprobs.contiguous().view(states_num, beam_size, self.lex_cand_num + slots_num, self.nt_cand_num, self.rule_cand_num).permute(1, 0, 2, 3, 4)

                    word_beam_logprobs = word_beam_logprobs * word_mask + word_beam_probs.expand(self.lex_cand_num + slots_num, self.nt_cand_num, self.rule_cand_num, beam_size, states_num).permute(3, 4, 0, 1, 2)
                    nt_beam_logprobs = nt_beam_logprobs * rule_mask + nt_beam_probs.expand(self.lex_cand_num + slots_num, self.nt_cand_num, self.rule_cand_num, beam_size, states_num).permute(3, 4, 0, 1, 2)
                    rule_beam_logprobs = rule_beam_logprobs * rule_mask + rule_beam_probs.expand(self.lex_cand_num + slots_num, self.nt_cand_num, self.rule_cand_num, beam_size, states_num).permute(3, 4, 0, 1, 2)

                    total_logprobs = word_beam_logprobs / _word_count + nt_beam_logprobs / _rule_count + rule_beam_logprobs / _rule_count
                    total_logprobs[dup_mask] = -np.inf
                    total_logprobs[total_logprobs != total_logprobs] = -np.inf

                    transition = torch.cuda.ByteTensor(transition) 
                    argtop_ind = torch.cuda.LongTensor(beam_size, states_num, 5)
                    all_logprobs = torch.zeros(beam_size, states_num).cuda()
                    for x in range(states_num):
                        state_going = (transition == x).expand(self.nt_cand_num, self.rule_cand_num, beam_size, states_num, self.lex_cand_num + slots_num).permute(2, 3, 4, 0, 1)
                        _total_logprobs = total_logprobs.clone()
                        _total_logprobs[~state_going] = -np.inf
                        logprobs, argtop = torch.topk(_total_logprobs.view(-1), beam_size, dim=0)
                        all_logprobs[:, x] = logprobs
                        argtop_ind[:, x, 0] = argtop / (states_num * (self.lex_cand_num + slots_num) * self.nt_cand_num * self.rule_cand_num)
                        argtop_ind[:, x, 1] = (argtop % (states_num * (self.lex_cand_num + slots_num) * self.nt_cand_num * self.rule_cand_num)) / ((self.lex_cand_num + slots_num) * self.nt_cand_num * self.rule_cand_num)
                        argtop_ind[:, x, 2] = (argtop % ((self.lex_cand_num + slots_num) * self.nt_cand_num * self.rule_cand_num)) / (self.nt_cand_num * self.rule_cand_num)
                        argtop_ind[:, x, 3] = (argtop % (self.nt_cand_num * self.rule_cand_num)) / self.rule_cand_num
                        argtop_ind[:, x, 4] = argtop % self.rule_cand_num

            def reshape_hidden(decoder_hidden, dim2):
                return (decoder_hidden[0].view(1, dim2, beam_size, -1),
                        decoder_hidden[1].view(1, dim2, beam_size, -1))
            if slots is None:
                decoder_hidden = (decoder_hidden[0].view(1, batch_size, beam_size, -1), decoder_hidden[1].view(1, batch_size, beam_size, -1))
                leaf_decoder_hidden = (leaf_decoder_hidden[0].view(1, batch_size, beam_size, -1),
                                       leaf_decoder_hidden[1].view(1, batch_size, beam_size, -1))
                if self.lex_level >= 2:
                    anc_decoder_hidden = reshape_hidden(anc_decoder_hidden, batch_size)
                    syn_decoder_hidden = reshape_hidden(syn_decoder_hidden, batch_size)
            else:
                decoder_hidden = (decoder_hidden[0].view(1, states_num, beam_size, -1), decoder_hidden[1].view(1, states_num, beam_size, -1))
                leaf_decoder_hidden = (leaf_decoder_hidden[0].view(1, states_num, beam_size, -1),
                                       leaf_decoder_hidden[1].view(1, states_num, beam_size, -1))
                if self.lex_level >= 2:
                    anc_decoder_hidden = reshape_hidden(anc_decoder_hidden, states_num)
                    syn_decoder_hidden = reshape_hidden(syn_decoder_hidden, states_num)

            if self.lex_level != 3:
                last = (best_args / top_k)
                curr = (best_args % top_k)
                if self.lex_level == 0:
                    current = current[last.tolist()]
                    leaves = leaves[last.tolist()]
                    lexicons = lexicons[last.tolist()]
                    curr_nt = curr_nt[last.tolist()]
                    final = final[last.tolist()]
                    word_beam = word_beam[last]
                    rule_beam = rule_beam[last]
                    word_count = word_count[last]
                    rule_count = rule_count[last]
                    word_beam_probs = word_beam_logprobs[last, curr]
                    rule_beam_probs = rule_beam_logprobs[last, curr]
                    word_beam[:, t+1] = word_argtop[last, curr].data
                    rule_beam[:, t+1] = rule_argtop[last, curr].data
                else:
                    current = current[(last / self.nt_cand_num).tolist()]
                    leaves = leaves[(last / self.nt_cand_num).tolist()]
                    lexicons = lexicons[(last / self.nt_cand_num).tolist()]
                    curr_nt = curr_nt[(last / self.nt_cand_num).tolist()]
                    final = final[(last / self.nt_cand_num).tolist()]
                    word_beam = word_beam[(last / self.nt_cand_num)]
                    rule_beam = rule_beam[(last / self.nt_cand_num)]
                    word_count = word_count[(last / self.nt_cand_num)]
                    rule_count = rule_count[(last / self.nt_cand_num)]
                    word_beam_probs = word_beam_logprobs[last, curr]
                    rule_beam_probs = rule_beam_logprobs[last, curr]
                    word_beam[:, t+1] = word_argtop[last, curr].data
                    rule_beam[:, t+1] = rule_argtop[last, curr].data
                if self.lex_level > 0:
                    nt_beam = nt_beam[(last / self.nt_cand_num)]
                    nt_beam[:, t+1] = nt_argtop[last / self.nt_cand_num, last % self.nt_cand_num].data
                    last /= self.nt_cand_num
            else:
                if slots is None:
                    last = argtop_ind[:, 0]
                    current = current[last.tolist()]
                    leaves = leaves[last.tolist()]
                    lexicons = lexicons[last.tolist()]
                    curr_nt = curr_nt[last.tolist()]
                    final = final[last.tolist()]
                    word_beam = word_beam[last]
                    nt_beam = nt_beam[last]
                    rule_beam = rule_beam[last]
                    word_count = word_count[last]
                    rule_count = rule_count[last]
                    word_beam[:, t+1] = word_argtop.data[last, argtop_ind[:, 1]]
                    nt_beam[:, t+1] = nt_argtop.data[last, argtop_ind[:, 1], argtop_ind[:, 2]]
                    rule_beam[:, t+1] = rule_argtop.data[last, argtop_ind[:, 1], argtop_ind[:, 2], argtop_ind[:, 3]]
                    word_beam_probs = word_beam_logprobs[last, argtop_ind[:, 1], argtop_ind[:, 2], argtop_ind[:, 3]]
                    word_beam_probs = word_beam_logprobs[last, argtop_ind[:, 1], argtop_ind[:, 2], argtop_ind[:, 3]]
                    nt_beam_probs = nt_beam_logprobs[last, argtop_ind[:, 1], argtop_ind[:, 2], argtop_ind[:, 3]]
                    rule_beam_probs = rule_beam_logprobs[last, argtop_ind[:, 1], argtop_ind[:, 2], argtop_ind[:, 3]]
                else:
                    new_curr_nt = np.array([['' for x in range(beam_size)] for y in range(states_num)], dtype=object)
                    new_final = np.array([[None for x in range(beam_size)] for y in range(states_num)])
                    new_current = np.array([[None for x in range(beam_size)] for y in range(states_num)])
                    word_argtop = word_argtop.contiguous().view(states_num, beam_size, self.lex_cand_num + slots_num).transpose(0, 1)
                    nt_argtop = nt_argtop.contiguous().view(states_num, beam_size, self.lex_cand_num + slots_num, self.nt_cand_num).transpose(0, 1)
                    rule_argtop = rule_argtop.contiguous().view(states_num, beam_size, self.lex_cand_num + slots_num, self.nt_cand_num, self.rule_cand_num).transpose(0, 1)
                    for x in range(states_num):
                        last = argtop_ind[:, x, :2]
                        new_current[x, :] = current[last[:, 1].tolist(), last[:, 0].tolist()]
                        new_final[x, :] = final[last[:, 1].tolist(), last[:, 0].tolist()]
                        new_curr_nt[x, :] = curr_nt[last[:, 1].tolist(), last[:, 0].tolist()]
                        new_word_beam[:, x, :] = word_beam[last[:, 0].tolist(), last[:, 1].tolist(), :]
                        new_word_beam[:, x, t+1] = word_argtop.data[argtop_ind[:, x, 0], argtop_ind[:, x, 1], argtop_ind[:, x, 2]]
                        word_beam_probs[:, x] = word_beam_logprobs[argtop_ind[:, x, 0], argtop_ind[:, x, 1], argtop_ind[:, x, 2], argtop_ind[:, x, 3], argtop_ind[:, x, 4]]
                        new_nt_beam[:, x, :] = nt_beam[last[:, 0].tolist(), last[:, 1].tolist(), :]
                        new_nt_beam[:, x, t+1] = nt_argtop.data[argtop_ind[:, x, 0], argtop_ind[:, x, 1], argtop_ind[:, x, 2], argtop_ind[:, x, 3]]
                        nt_beam_probs[:, x] = nt_beam_logprobs[argtop_ind[:, x, 0], argtop_ind[:, x, 1], argtop_ind[:, x, 2], argtop_ind[:, x, 3], argtop_ind[:, x, 4]]
                        new_rule_beam[:, x, :] = rule_beam[last[:, 0].tolist(), last[:, 1].tolist(), :]
                        new_rule_beam[:, x, t+1] = rule_argtop.data[argtop_ind[:, x, 0], argtop_ind[:, x, 1], argtop_ind[:, x, 2], argtop_ind[:, x, 3], argtop_ind[:, x, 4]]
                        rule_beam_probs[:, x] = rule_beam_logprobs[argtop_ind[:, x, 0], argtop_ind[:, x, 1], argtop_ind[:, x, 2], argtop_ind[:, x, 3], argtop_ind[:, x, 4]]
                    word_beam = new_word_beam
                    nt_beam = new_nt_beam
                    rule_beam = new_rule_beam
                    null_ind = all_logprobs == -np.inf
                    word_beam_probs[null_ind] = -np.inf
                    nt_beam_probs[null_ind] = -np.inf
                    rule_beam_probs[null_ind] = -np.inf
                    current = new_current
                    final = new_final
                    curr_nt = new_curr_nt

            def merge_hidden(decoder_hidden, dim2):
                return (decoder_hidden[0].view(1, dim2 * beam_size, -1),
                        decoder_hidden[1].view(1, dim2 * beam_size, -1))

            if slots is None:
                decoder_hidden[0][:, 0, :, :] = decoder_hidden[0][:, 0, :, :][:, last, :]
                decoder_hidden[1][:, 0, :, :] = decoder_hidden[1][:, 0, :, :][:, last, :]
                decoder_hidden = (decoder_hidden[0].view(1, batch_size * beam_size, -1), decoder_hidden[1].view(1, batch_size * beam_size, -1))

                leaf_decoder_hidden[0][:, 0, :, :] = leaf_decoder_hidden[0][:, 0, :, :][:, last, :]
                leaf_decoder_hidden[1][:, 0, :, :] = leaf_decoder_hidden[1][:, 0, :, :][:, last, :]
                leaf_decoder_hidden = (leaf_decoder_hidden[0].view(1, batch_size * beam_size, -1),
                                       leaf_decoder_hidden[1].view(1, batch_size * beam_size, -1))
                if self.lex_level >= 2:
                    anc_decoder_hidden[0][:, 0, :, :] = anc_decoder_hidden[0][:, 0, :, :][:, last, :]
                    anc_decoder_hidden[1][:, 0, :, :] = anc_decoder_hidden[1][:, 0, :, :][:, last, :]
                    syn_decoder_hidden[0][:, 0, :, :] = syn_decoder_hidden[0][:, 0, :, :][:, last, :]
                    syn_decoder_hidden[1][:, 0, :, :] = syn_decoder_hidden[1][:, 0, :, :][:, last, :]
                    
                    anc_decoder_hidden = merge_hidden(anc_decoder_hidden, batch_size)
                    syn_decoder_hidden = merge_hidden(syn_decoder_hidden, batch_size)
            else:
                new_decoder_hidden = (decoder_hidden[0].data.clone(), decoder_hidden[1].data.clone())
                new_leaf_decoder_hidden = (leaf_decoder_hidden[0].data.clone(), leaf_decoder_hidden[1].data.clone())
                if self.lex_level >= 2:
                    new_anc_decoder_hidden = (anc_decoder_hidden[0].data.clone(), anc_decoder_hidden[1].data.clone())
                    new_syn_decoder_hidden = (syn_decoder_hidden[0].data.clone(), syn_decoder_hidden[1].data.clone())
                for x in range(states_num):
                    last = argtop_ind[:, x, :2]
                    new_decoder_hidden[0][:, x, :, :] = decoder_hidden[0][:, last[:, 1], last[:, 0], :].data
                    new_decoder_hidden[1][:, x, :, :] = decoder_hidden[1][:, last[:, 1], last[:, 0], :].data

                    new_leaf_decoder_hidden[0][:, x, :, :] = leaf_decoder_hidden[0][:, last[:, 1], last[:, 0], :].data
                    new_leaf_decoder_hidden[1][:, x, :, :] = leaf_decoder_hidden[1][:, last[:, 1], last[:, 0], :].data

                    if self.lex_level >= 2:
                        new_anc_decoder_hidden[0][:, x, :, :] = anc_decoder_hidden[0][:, last[:, 1], last[:, 0], :].data
                        new_anc_decoder_hidden[1][:, x, :, :] = anc_decoder_hidden[1][:, last[:, 1], last[:, 0], :].data

                        new_syn_decoder_hidden[0][:, x, :, :] = syn_decoder_hidden[0][:, last[:, 1], last[:, 0], :].data
                        new_syn_decoder_hidden[1][:, x, :, :] = syn_decoder_hidden[1][:, last[:, 1], last[:, 0], :].data

                decoder_hidden = (Variable(new_decoder_hidden[0].view(1, states_num * beam_size, -1)), Variable(new_decoder_hidden[1].view(1, states_num * beam_size, -1)))
                leaf_decoder_hidden = (Variable(new_leaf_decoder_hidden[0].view(1, states_num * beam_size, -1)), Variable(new_leaf_decoder_hidden[1].view(1, states_num * beam_size, -1)))
                if self.lex_level >= 2:
                    anc_decoder_hidden = (Variable(new_anc_decoder_hidden[0].view(1, states_num * beam_size, -1)), Variable(new_anc_decoder_hidden[1].view(1, states_num * beam_size, -1)))
                    syn_decoder_hidden = (Variable(new_syn_decoder_hidden[0].view(1, states_num * beam_size, -1)), Variable(new_syn_decoder_hidden[1].view(1, states_num * beam_size, -1)))

            if slots is None:
                if self.lex_level == 0:
                    for x in range(beam_size):
                        current[x] = deepcopy(current[x])
                        if current[x] is None:
                            continue
                        if self.preterminal[curr_nt[x]]:
                            word = word_beam[x, t+1]
                            ch = Node('{}__{}__ '.format(curr_nt[x], word), parent=current[x])

                            lex_decoder_input = self.lexicon(Variable(torch.cuda.LongTensor([int(word)]))).unsqueeze(1)
                            _, _leaf_decoder_hidden = self.leaf_decoder(lex_decoder_input, (leaf_decoder_hidden[0][:, x, :].unsqueeze(1), leaf_decoder_hidden[1][:, x, :].unsqueeze(1)))
                            leaf_decoder_hidden[0][:, x, :] = _leaf_decoder_hidden[0][0]
                            leaf_decoder_hidden[1][:, x, :] = _leaf_decoder_hidden[1][0]

                            while True:
                                if current[x] is None:
                                    break
                                if current[x].parent is None:
                                    final[x] = deepcopy(current[x])
                                _, _, rule = current[x].name.split('__')
                                num_children = len(rule.split())
                                if num_children > len(current[x].children):
                                    break
                                else:
                                    current[x] = current[x].parent
                        else:
                            rule = self.rules_by_id[rule_beam[x, t+1]]
                            word = 0
                            current[x] = Node('{}__{}__{}'.format(curr_nt[x], word, rule), parent=current[x])
                else:
                    anc_select = []
                    leaf_select = []
                    anc_word = []
                    anc_nt = []
                    anc_rule = []
                    leaf_word = []
                    for x in range(beam_size):
                        current[x] = deepcopy(current[x])
                        #tree_dict[x] = deepcopy(tree_dict[x])
                        if current[x] is None:
                            continue
                        #nt, word, rule = current[x].name.split('__')
                        if self.preterminal[curr_nt[x]]:
                            _, word, rule = current[x].name.split('__')
                            if len(current[x].children) not in inheritance(rule):
                                word = word_beam[x, t+1]
                            ch = Node('{}__{}__ []'.format(curr_nt[x], word), parent=current[x])

                            while True:
                                if current[x] is None:
                                    break
                                if current[x].parent is None:
                                    final[x] = deepcopy(current[x])
                                _, _, rule = current[x].name.split('__')
                                num_children = len(rule[:rule.find('[')].split())
                                if num_children > len(current[x].children):
                                    break
                                else:
                                    current[x] = current[x].parent

                            leaf_select.append(x)
                            leaf_word.append(int(word))
                            if current[x] is not None:
                                new_hidden = tree_dict[current[x].id]
                                anc_decoder_hidden[0][:, x, :].data = new_hidden[0][0].data.clone()
                                anc_decoder_hidden[1][:, x, :].data = new_hidden[0][1].data.clone()
                                syn_decoder_hidden[0][:, x, :].data = new_hidden[1][0].data.clone()
                                syn_decoder_hidden[1][:, x, :].data = new_hidden[1][1].data.clone()
                        else:
                            nt, word, par_rule = current[x].name.split('__')
                            rule = self.rules_by_id[rule_beam[x, t+1]]
                            if len(current[x].children) not in inheritance(par_rule):
                                word = word_beam[x, t+1]

                            anc_select.append(x)
                            anc_word.append(int(word))
                            anc_nt.append(self.nt_dictionary[nt])
                            anc_rule.append(self.rule_dictionary['RULE: {}'.format(par_rule[:par_rule.find('[')-1])])

                            tag = self.nt_by_id[nt_beam[x, t+1]]
                            current[x] = Node('{}__{}__{} [{}]'.format(curr_nt[x], word, rule, rule.split().index(tag)), parent=current[x])
                            current[x].id = node_ind
                            node_ind += 1
                    if anc_select:
                        anc_decoder_input = self.lexicon(Variable(torch.cuda.LongTensor(anc_word))).unsqueeze(1)
                        _, _anc_decoder_hidden = self.anc_decoder(anc_decoder_input, (anc_decoder_hidden[0][:, anc_select, :], anc_decoder_hidden[1][:, anc_select, :]))
                        anc_decoder_hidden[0][:, anc_select, :] = _anc_decoder_hidden[0][0].clone()
                        anc_decoder_hidden[1][:, anc_select, :] = _anc_decoder_hidden[1][0].clone()

                        syn_decoder_input = torch.cat((self.constituent(Variable(torch.cuda.LongTensor(anc_nt))), self.rule(Variable(torch.cuda.LongTensor(anc_rule)))), dim=1).unsqueeze(1)
                        _, _syn_decoder_hidden = self.anc_syn_decoder(syn_decoder_input, (syn_decoder_hidden[0][:, anc_select, :], syn_decoder_hidden[1][:, anc_select, :]))
                        syn_decoder_hidden[0][:, anc_select, :] = _syn_decoder_hidden[0][0].clone()
                        syn_decoder_hidden[1][:, anc_select, :] = _syn_decoder_hidden[1][0].clone()
                    if leaf_select:
                        leaf_decoder_input = self.lexicon(Variable(torch.cuda.LongTensor(leaf_word))).unsqueeze(1)
                        _, _leaf_decoder_hidden = self.leaf_decoder(leaf_decoder_input, (leaf_decoder_hidden[0][:, leaf_select, :], leaf_decoder_hidden[1][:, leaf_select, :]))
                        leaf_decoder_hidden[0][:, leaf_select, :] = _leaf_decoder_hidden[0][0]
                        leaf_decoder_hidden[1][:, leaf_select, :] = _leaf_decoder_hidden[1][0]
                    for x in range(beam_size):
                        if current[x] is not None:
                            tree_dict[current[x].id] = (
                                                        (anc_decoder_hidden[0][:, x, :].clone(), anc_decoder_hidden[1][:, x, :].clone()),
                                                        (syn_decoder_hidden[0][:, x, :].clone(), syn_decoder_hidden[1][:, x, :].clone())
                                                       )
            else:
                if self.lex_level == 0:
                    for x in range(beam_size):
                        current[x] = deepcopy(current[x])
                        if current[x] is None:
                            continue
                        if self.preterminal[curr_nt[x]]:
                            word = word_beam[x, t+1]
                            ch = Node('{}__{}__ '.format(curr_nt[x], word), parent=current[x])

                            lex_decoder_input = self.lexicon(Variable(torch.cuda.LongTensor([int(word)]))).unsqueeze(1)
                            _, _leaf_decoder_hidden = self.leaf_decoder(lex_decoder_input, (leaf_decoder_hidden[0][:, x, :].unsqueeze(1), leaf_decoder_hidden[1][:, x, :].unsqueeze(1)))
                            leaf_decoder_hidden[0][:, x, :] = _leaf_decoder_hidden[0][0]
                            leaf_decoder_hidden[1][:, x, :] = _leaf_decoder_hidden[1][0]

                            while True:
                                if current[x] is None:
                                    break
                                if current[x].parent is None:
                                    final[x] = deepcopy(current[x])
                                _, _, rule = current[x].name.split('__')
                                num_children = len(rule.split())
                                if num_children > len(current[x].children):
                                    break
                                else:
                                    current[x] = current[x].parent
                        else:
                            rule = self.rules_by_id[rule_beam[x, t+1]]
                            word = 0
                            current[x] = Node('{}__{}__{}'.format(curr_nt[x], word, rule), parent=current[x])
                else:
                    anc_select = []
                    leaf_select = []
                    anc_word = []
                    anc_nt = []
                    anc_rule = []
                    leaf_word = []
                    for x in range(beam_size):
                        for y in range(states_num):
                            current[y, x] = deepcopy(current[y, x])
                            #tree_dict[x] = deepcopy(tree_dict[x])
                            if current[y, x] is None:
                                continue
                            #nt, word, rule = current[x].name.split('__')
                            if self.preterminal[curr_nt[y, x]]:
                                _, word, rule = current[y, x].name.split('__')
                                if len(current[y, x].children) not in inheritance(rule):
                                    word = word_beam[x, y, t+1]
                                ch = Node('{}__{}__ []'.format(curr_nt[y, x], word), parent=current[y, x])

                                leaf_select.append(y * beam_size + x)
                                leaf_word.append(int(word))

                                while True:
                                    if current[y, x] is None:
                                        break
                                    if current[y, x].parent is None:
                                        final[y, x] = deepcopy(current[y, x])
                                    _, _, rule = current[y, x].name.split('__')
                                    num_children = len(rule[:rule.find('[')].split())
                                    if num_children > len(current[y, x].children):
                                        break
                                    else:
                                        current[y, x] = current[y, x].parent
                                if current[y, x] is not None:
                                    new_hidden = tree_dict[current[y, x].id]
                                    anc_decoder_hidden[0][:, y*beam_size+x, :].data = new_hidden[0][0].data.clone()
                                    anc_decoder_hidden[1][:, y*beam_size+x, :].data = new_hidden[0][1].data.clone()
                                    syn_decoder_hidden[0][:, y*beam_size+x, :].data = new_hidden[1][0].data.clone()
                                    syn_decoder_hidden[1][:, y*beam_size+x, :].data = new_hidden[1][1].data.clone()
                            else:
                                nt, word, par_rule = current[y, x].name.split('__')
                                rule = self.rules_by_id[rule_beam[x, y, t+1]]
                                if len(current[y, x].children) not in inheritance(par_rule):
                                    word = word_beam[x, y, t+1]

                                anc_select.append(y * beam_size + x)
                                anc_word.append(int(word))
                                anc_nt.append(self.nt_dictionary[nt])
                                anc_rule.append(self.rule_dictionary['RULE: {}'.format(par_rule[:par_rule.find('[')-1])])

                                tag = self.nt_by_id[nt_beam[x, y, t+1]]
                                if tag in rule:
                                    current[y, x] = Node('{}__{}__{} [{}]'.format(curr_nt[y, x], word, rule, rule.split().index(tag)), parent=current[y, x])
                                else:
                                    current[y, x] = Node('{}__{}__{} [{}]'.format(curr_nt[y, x], word, rule, 0), parent=current[y, x])
                                current[y, x].id = node_ind
                                node_ind += 1
                    if anc_select:
                        anc_decoder_input = self.lexicon(Variable(torch.cuda.LongTensor(anc_word))).unsqueeze(1)
                        _, _anc_decoder_hidden = self.anc_decoder(anc_decoder_input, (anc_decoder_hidden[0][:, anc_select, :], anc_decoder_hidden[1][:, anc_select, :]))
                        anc_decoder_hidden[0][:, anc_select, :] = _anc_decoder_hidden[0][0].clone()
                        anc_decoder_hidden[1][:, anc_select, :] = _anc_decoder_hidden[1][0].clone()

                        syn_decoder_input = torch.cat((self.constituent(Variable(torch.cuda.LongTensor(anc_nt))), self.rule(Variable(torch.cuda.LongTensor(anc_rule)))), dim=1).unsqueeze(1)
                        _, _syn_decoder_hidden = self.anc_syn_decoder(syn_decoder_input, (syn_decoder_hidden[0][:, anc_select, :], syn_decoder_hidden[1][:, anc_select, :]))
                        syn_decoder_hidden[0][:, anc_select, :] = _syn_decoder_hidden[0][0].clone()
                        syn_decoder_hidden[1][:, anc_select, :] = _syn_decoder_hidden[1][0].clone()
                    if leaf_select:
                        leaf_decoder_input = self.lexicon(Variable(torch.cuda.LongTensor(leaf_word))).unsqueeze(1)
                        _, _leaf_decoder_hidden = self.leaf_decoder(leaf_decoder_input, (leaf_decoder_hidden[0][:, leaf_select, :], leaf_decoder_hidden[1][:, leaf_select, :]))
                        leaf_decoder_hidden[0][:, leaf_select, :] = _leaf_decoder_hidden[0][0]
                        leaf_decoder_hidden[1][:, leaf_select, :] = _leaf_decoder_hidden[1][0]
                    for x in range(beam_size):
                        for y in range(states_num):
                            if current[y, x] is not None:
                                tree_dict[current[y, x].id] = (
                                                               (anc_decoder_hidden[0][:, y*beam_size+x, :].clone(), anc_decoder_hidden[1][:, y*beam_size+x, :].clone()),
                                                               (syn_decoder_hidden[0][:, y*beam_size+x, :].clone(), syn_decoder_hidden[1][:, y*beam_size+x, :].clone())
                                                              )
            if (current == None).all():
                break

        '''
        best, best_arg = beam_probs.max(1)
        generations = beam[torch.arange(_batch_size).long().cuda(), best_arg.data].data.cpu()
        return generations, best
        '''
        if slots is None:
            if final[0] is None:
                return current[0]
            else:
                return final[0]
        else:
            if final[-1][0] is None:
                return current[-1][0]
            else:
                return final[-1][0]


class LexicalizedGrammarLM(nn.Module):
    def __init__(self, lex_input_size, nt_input_size, rule_input_size, 
                 context_size, hidden_size, 
                 lex_vocab_size, nt_vocab_size, rule_vocab_size, 
                 lex_vectors, nt_vectors, rule_vectors,
                 lex_dictionary, nt_dictionary, rule_dictionary,
                 lex_level):
        super(LexicalizedGrammarLM, self).__init__()
        self.hidden_size = hidden_size
        self.lex_input_size = lex_input_size
        self.nt_input_size = nt_input_size
        self.rule_input_size = rule_input_size
        self.lex_vocab_size = lex_vocab_size
        self.nt_vocab_size = nt_vocab_size
        self.rule_vocab_size = rule_vocab_size
        self.lex_level = lex_level
        self.lexicon = Embedding(lex_vocab_size, lex_input_size, lex_vectors, trainable=False)
        self.constituent = Embedding(nt_vocab_size, nt_input_size, nt_vectors, trainable=True)
        self.rule = Embedding(rule_vocab_size, rule_input_size, rule_vectors, trainable=True)
        self.depth = Embedding(25, 256, trainable=True)
        self.breadth = Embedding(40, 256, trainable=True)
        self.pos_embed_size = 0
        self.encoder = nn.LSTM(lex_input_size, hidden_size)
        self.context_size = context_size
        self.dropout = 0.2
        if self.lex_level == 0:
            self.lex_out = nn.Linear(hidden_size, lex_input_size)
            self.nt_out = nn.Linear(hidden_size, nt_input_size)
            #self.nt_dist.weight = self.constituent.weight
            self.leaf_decoder = nn.LSTM(lex_input_size, hidden_size, batch_first=True, dropout=0.0)
            self.decoder_input_size = nt_input_size + rule_input_size
            #self.decoder_input_size = lex_input_size * 6 + nt_input_size * 7

            self.leaf_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
            self.leaf_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
            self.leaf_cell_fc1 = nn.Linear(hidden_size, hidden_size)
            self.leaf_cell_fc2 = nn.Linear(hidden_size, hidden_size)

            #self.hidden_out_fc1 = nn.Linear(hidden_size * 3, hidden_size * 2)
            #self.hidden_out_fc2 = nn.Linear(hidden_size * 2, hidden_size * 1)
            self.decoder = nn.LSTM(self.decoder_input_size, hidden_size, batch_first=True, dropout=self.dropout)

            self.lex_hidden_out = nn.Linear(hidden_size * 2 + context_size, hidden_size)
            self.rule_hidden_out = nn.Linear(hidden_size * 2 + context_size, hidden_size)

            #self.rule_out = nn.Linear(hidden_size, rule_input_size)
            self.rule_dist = nn.Linear(hidden_size, rule_vocab_size)
            #self.rule_dist.weight = self.rule.weight
        elif self.lex_level == 1:
            self.lex_out = nn.Linear(hidden_size + nt_input_size, lex_input_size)
            self.nt_dist = nn.Linear(nt_input_size, nt_vocab_size)
            self.nt_out = nn.Linear(hidden_size, nt_input_size)
            self.nt_dist.weight = self.constituent.weight
            self.tree_input_size = lex_input_size + nt_input_size
            self.tree = nn.Linear((lex_input_size + nt_input_size) * 4, self.tree_input_size)
            self.decoder_input_size = lex_input_size * 4 + nt_input_size * 5 + self.tree_input_size
            self.decoder = nn.LSTM(self.decoder_input_size + context_size, hidden_size, batch_first=True, dropout=0.0)
        else:
            #self.lex_out = nn.Linear(hidden_size + lex_input_size, lex_input_size)
            self.lex_out = nn.Linear(hidden_size, lex_input_size)
            self.nt_dist = nn.Linear(nt_input_size, nt_vocab_size)
            self.nt_out = nn.Linear(hidden_size, nt_input_size)
            #self.nt_dist.weight = self.constituent.weight
            self.tree_input_size = lex_input_size * 2
            self.tree = nn.Linear(lex_input_size * 4 + nt_input_size * 0 + rule_input_size * 0, self.tree_input_size)
            self.leaf_decoder = nn.LSTM(lex_input_size, hidden_size, batch_first=True, dropout=self.dropout)
            self.lex_decoder = nn.LSTM(lex_input_size, hidden_size, batch_first=True, dropout=self.dropout)
            self.anc_decoder = nn.LSTM(lex_input_size, hidden_size, batch_first=True, dropout=self.dropout)
            self.anc_syn_decoder = nn.LSTM(nt_input_size + rule_input_size, hidden_size, batch_first=True, dropout=self.dropout)
            self.decoder_input_size = nt_input_size + rule_input_size
            #self.decoder_input_size = lex_input_size * 6 + nt_input_size * 7

            self.leaf_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
            self.leaf_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
            self.leaf_cell_fc1 = nn.Linear(hidden_size, hidden_size)
            self.leaf_cell_fc2 = nn.Linear(hidden_size, hidden_size)

            self.lex_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
            self.lex_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
            self.lex_cell_fc1 = nn.Linear(hidden_size, hidden_size)
            self.lex_cell_fc2 = nn.Linear(hidden_size, hidden_size)

            self.anc_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
            self.anc_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
            self.anc_cell_fc1 = nn.Linear(hidden_size, hidden_size)
            self.anc_cell_fc2 = nn.Linear(hidden_size, hidden_size)

            self.anc_syn_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
            self.anc_syn_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
            self.anc_syn_cell_fc1 = nn.Linear(hidden_size, hidden_size)
            self.anc_syn_cell_fc2 = nn.Linear(hidden_size, hidden_size)

            self.rule_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
            self.rule_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
            self.rule_cell_fc1 = nn.Linear(hidden_size, hidden_size)
            self.rule_cell_fc2 = nn.Linear(hidden_size, hidden_size)

            #self.hidden_out_fc1 = nn.Linear(hidden_size * 3, hidden_size * 2)
            #self.hidden_out_fc2 = nn.Linear(hidden_size * 2, hidden_size * 1)
            self.decoder = nn.LSTM(self.decoder_input_size, hidden_size, batch_first=True, dropout=self.dropout)
            self.rule_decoder = nn.LSTM(self.rule_input_size, hidden_size, batch_first=True, dropout=self.dropout)
            if self.lex_level == 2:
                self.lex_hidden_out = nn.Linear(hidden_size * 3 + nt_input_size, hidden_size)
                self.rule_hidden_out = nn.Linear(hidden_size * 3 + nt_input_size, hidden_size)
                self.nt_hidden_out = nn.Linear(hidden_size * 3, hidden_size)
            else:
                self.decoder_output_size = hidden_size * 4
                self.lex_hidden_out = nn.Linear(self.decoder_output_size, hidden_size)
                self.rule_hidden_out = nn.Linear(self.decoder_output_size + nt_input_size + lex_input_size, hidden_size)
                self.nt_hidden_out = nn.Linear(self.decoder_output_size + lex_input_size, hidden_size)

                self.rule_out = nn.Linear(hidden_size, rule_input_size)
                self.rule_dist = nn.Linear(rule_input_size, rule_vocab_size)
                #self.rule_dist.weight = self.rule.weight
        #self.rule_out = nn.Linear(hidden_size, rule_input_size)
        self.lex_dist = nn.Linear(lex_input_size, lex_vocab_size)
        self.hidden_fc1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_fc2 = nn.Linear(hidden_size, hidden_size)
        self.cell_fc1 = nn.Linear(hidden_size, hidden_size)
        self.cell_fc2 = nn.Linear(hidden_size, hidden_size)
        self.lex_dictionary = lex_dictionary
        self.nt_dictionary = nt_dictionary
        self.rule_dictionary = rule_dictionary
        self.eou = lex_dictionary['__eou__']
        self.lex_dist.weight = self.lexicon.weight

        if self.lex_level == 0:
            self.a_key = nn.Linear(hidden_size, 200)
            self.p_key = nn.Linear(hidden_size * 2, 200)
        else:
            self.a_key = nn.Linear(hidden_size * 4, 300)
            self.p_key = nn.Linear(hidden_size * 5, 300)
        self.q_key = nn.Linear(hidden_size, 300)
        self.psn_key = nn.Linear(hidden_size, 300)
        self.q_value = nn.Linear(hidden_size, context_size)
        self.psn_value = nn.Linear(hidden_size, context_size)

        self.init_forget_bias(self.encoder, 0)
        self.init_forget_bias(self.decoder, 0)
        self.init_forget_bias(self.leaf_decoder, 0)
        if self.lex_level >= 2:
            self.init_forget_bias(self.lex_decoder, 0)
            self.init_forget_bias(self.anc_decoder, 0)

    def init_rules(self):
        self.ROOT = self.nt_dictionary['ROOT']
        self.rules_by_id = defaultdict(str)
        self.nt_by_id = defaultdict(str)
        rules_by_nt = defaultdict(list)
        for k, v in self.rule_dictionary.items():
            self.rules_by_id[v] = k[6:]
            for nt in k.split()[1:]:
                rules_by_nt[nt].append(self.rule_dictionary[k])

        for k, v in self.nt_dictionary.items():
            self.nt_by_id[v] = k

        #self.rules_by_nt = {}
        self.rules_by_nt = torch.cuda.ByteTensor(self.nt_vocab_size, self.rule_vocab_size).fill_(False)
        for k, v in rules_by_nt.items():
            vv = torch.cuda.ByteTensor(self.rule_vocab_size).fill_(False)
            vv[v] = True
            self.rules_by_nt[self.nt_dictionary[k]] = vv

        self.preterminal = defaultdict(bool)
        for pt in Preterminal:
            self.preterminal[pt] = True

    def init_forget_bias(self, rnn, b):
        for names in rnn._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2 
                bias.data[start:end].fill_(b)

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        if self.lex_level >= 2:
            self.leaf_decoder.flatten_parameters()
            self.lex_decoder.flatten_parameters()
            self.anc_decoder.flatten_parameters()
            self.anc_syn_decoder.flatten_parameters()

    def hidden_transform(self, hidden, prefix):
        return eval('F.tanh(self.{}hidden_fc2(F.relu(self.{}hidden_fc1(hidden))))'.format(prefix, prefix))

    def cell_transform(self, cell, prefix):
        return eval('F.tanh(self.{}cell_fc2(F.relu(self.{}cell_fc1(cell))))'.format(prefix, prefix))

    def init_hidden(self, src_hidden, prefix=''):
        hidden = src_hidden[0]
        cell = src_hidden[1]
        hidden = self.hidden_transform(hidden, prefix)
        cell = self.cell_transform(cell, prefix)
        return (hidden, cell)

    def attention(self, decoder_hidden, src_hidden, src_max_len, src_lengths, src_last_hidden, psn_hidden, psn_max_len, psn_lengths):
        '''
        a_key = F.tanh(self.a_key(decoder_hidden[0].squeeze(0)))
        q_key = F.tanh(self.q_key(src_hidden))
        q_value = F.tanh(self.q_value(src_hidden))
        q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
        q_mask  = torch.arange(src_max_len).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(src_max_len, 1).transpose(0, 1)
        q_energy[~q_mask] = -np.inf
        q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
        q_context = torch.bmm(q_weights, q_value).squeeze(1)
        context = q_context.unsqueeze(1)
        '''
        a_key = F.tanh(self.a_key(decoder_hidden[0].squeeze(0)))
        p_key = F.tanh(self.p_key(torch.cat((decoder_hidden[0].squeeze(0), src_last_hidden[0].squeeze(0)), dim=1)))

        q_key = F.tanh(self.q_key(src_hidden))
        psn_key = F.tanh(self.psn_key(psn_hidden))
        q_value = F.tanh(self.q_value(src_hidden))
        psn_value = F.tanh(self.psn_value(psn_hidden))
        q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
        psn_energy = torch.bmm(psn_key, p_key.unsqueeze(2)).squeeze(2)
        q_mask  = torch.arange(src_max_len).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(src_max_len, 1).transpose(0, 1)
        psn_mask  = torch.arange(psn_max_len).long().cuda().repeat(psn_hidden.size(0), 1) < psn_lengths.cuda().repeat(psn_max_len, 1).transpose(0, 1)
        q_energy[~q_mask] = -np.inf
        psn_energy[~psn_mask] = -np.inf
        '''
        q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
        psn_weights = F.softmax(psn_energy, dim=1).unsqueeze(1)
        '''
        q_weights = F.sigmoid(q_energy).unsqueeze(1)
        psn_weights = F.sigmoid(psn_energy).unsqueeze(1)
        q_context = torch.bmm(q_weights, q_value)
        psn_context = torch.bmm(psn_weights, psn_value)

        #return context
        return torch.cat((q_context, psn_context), dim=2)

    def encode(self, tar_seqs, indices, name, init_hidden):
        lengths = [ind.max().item() for ind in indices]
        max_len = max(lengths)
        mask = [x.copy() for x in indices]
        _indices = [None for x in indices]
        for x in range(len(indices)):
            mask[x][1:] -= mask[x][:-1].copy()
            _indices[x] = np.zeros(tar_seqs[0].size(1), dtype=np.int64)
            _indices[x][1:len(indices[x])] = indices[x][:-1]
            _indices[x][0] = 0
        tar_lex_embed = self.lexicon(Variable(tar_seqs[1]).cuda())
        #tar_nt_embed = self.constituent(Variable(tar_seqs[0]).cuda())
        #_tar_embed = torch.cat((tar_lex_embed, tar_nt_embed), dim=2)
        _tar_embed = tar_lex_embed
        tar_embed = Variable(torch.zeros(_tar_embed.size(0), max_len, _tar_embed.size(2)).cuda())
        for x in range(tar_embed.size(0)):
            ind = torch.from_numpy(np.arange(len(mask[x]))[mask[x].astype(bool)]).long().cuda()
            tar_embed[x, :lengths[x], :] = _tar_embed[x][ind]
        t_lengths, perm_idx = torch.LongTensor(lengths).sort(0, descending=True)
        tar_embed = tar_embed[perm_idx.cuda()]
        packed_input = pack_padded_sequence(tar_embed, t_lengths.numpy(), batch_first=True)
        hidden, _ = eval('self.{}_decoder(packed_input, init_hidden)'.format(name))
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        hidden = hidden[perm_idx.sort()[1].cuda()]

        return hidden, _indices

    def encode_anc(self, anc_embed, anc_lengths, trg_lengths, batch_size, name):
        f_lengths, perm_idx = torch.LongTensor(anc_lengths).sort(0, descending=True)
        anc_embed = anc_embed[perm_idx.cuda()]
        '''
        nonzeros = f_lengths.nonzero().squeeze(1)
        zeros = (f_lengths == 0).nonzero().squeeze(1)
        '''
        #packed_input = pack_padded_sequence(anc_embed[nonzeros.cuda()], f_lengths[nonzeros].numpy(), batch_first=True)
        packed_input = pack_padded_sequence(anc_embed, f_lengths.numpy(), batch_first=True)
        #anc_output, anc_last_hidden = eval('self.{}decoder(packed_input)'.format(name))
        anc_output, _ = eval('self.{}decoder(packed_input)'.format(name))
        '''
        #anc_init_hidden = self.init_hidden(src_last_hidden, name) 
        anc_init_hidden = (Variable(torch.zeros(1, batch_size, self.hidden_size).cuda()), Variable(torch.zeros(1, batch_size, self.hidden_size).cuda()))
        _anc_hidden = Variable(torch.cuda.FloatTensor(len(anc_lengths), self.hidden_size).fill_(0))
        #_anc_hidden[:nonzeros.size(0)] = anc_last_hidden[0].squeeze(0)
        _anc_hidden = _anc_hidden[perm_idx.sort()[1].cuda()]
        #_anc_hidden[perm_idx[zeros].cuda()] = anc_init_hidden[0].squeeze(0)

        anc_hidden = Variable(torch.cuda.FloatTensor(batch_size, max(trg_lengths), self.hidden_size).fill_(0))
        start = 0
        for x in range(batch_size):
            anc_hidden[x, :trg_lengths[x], :] = _anc_hidden[start:start+trg_lengths[x]]
            start += trg_lengths[x]

        return anc_hidden, anc_init_hidden
        '''
        return pad_packed_sequence(anc_output)[0].transpose(0, 1)[perm_idx.sort()[1].cuda()]

    def forward(self, indices, trg_seqs, trg_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, positions, ancestors, anc_lengths):
        batch_size = trg_seqs[0].size(0)

        #rule_decoder_hidden = self.init_hidden(src_last_hidden, 'rule_') 
        #leaf_init_hidden = self.init_hidden(src_last_hidden, 'leaf_') 
        decoder_hidden = (Variable(torch.zeros(1, batch_size, self.hidden_size).cuda()), Variable(torch.zeros(1, batch_size, self.hidden_size).cuda()))
        leaf_init_hidden = (Variable(torch.zeros(1, batch_size, self.hidden_size).cuda()), Variable(torch.zeros(1, batch_size, self.hidden_size).cuda()))
        leaf_decoder_hidden, _leaf_indices = self.encode(trg_seqs, leaf_indices, 'leaf', leaf_init_hidden)
        leaf_decoder_hidden = torch.cat((leaf_init_hidden[0].transpose(0, 1), leaf_decoder_hidden), dim=1)

        ans_nt = self.constituent(Variable(trg_seqs[0]).cuda())

        ans_rule_embed = rule_seqs.clone()
        ans_rule_embed[:, 1:] = ans_rule_embed[:, :-1]
        ans_rule_embed[:, 0] = 0
        ans_rule_embed = self.rule(Variable(ans_rule_embed).cuda())

        anc_embed = self.lexicon(Variable(ancestors[0]).cuda())
        anc_hidden = self.encode_anc(anc_embed, anc_lengths, trg_lengths, batch_size, 'anc_')

        anc_syn_embed = torch.cat((self.constituent(Variable(ancestors[1]).cuda()), self.rule(Variable(ancestors[2]).cuda())), dim=2)
        anc_syn_hidden = self.encode_anc(anc_syn_embed, anc_lengths, trg_lengths, batch_size, 'anc_syn_')

        ans_embed = torch.cat((ans_nt, ans_rule_embed), dim=2)
        
        trg_l = max(trg_lengths)
        batch_ind = torch.arange(batch_size).long().cuda()
        decoder_outputs = Variable(torch.FloatTensor(batch_size, trg_l, self.decoder_output_size).cuda())

        depths = torch.max(positions[0] - 1, torch.zeros_like(positions[0])).cuda()
        leaf_ind = positions[1]
        leaf_num = [x[-1].item() for x in leaf_indices]
        leaf_cumsum = np.zeros(batch_size, dtype=np.int32)
        leaf_cumsum[1:] = np.cumsum(leaf_num)[:-1]
        for step in range(trg_l):
            decoder_input = ans_embed[:, step, :].unsqueeze(1)
            leaf_select = torch.cuda.LongTensor([x[step].item() for x in _leaf_indices])
            anc_leaf_select = (leaf_cumsum + leaf_ind[:, step]).cuda()
            #lex_select = torch.cuda.LongTensor([x[step].item() for x in _lex_indices])
            #decoder_input = torch.cat((decoder_input, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1), lex_decoder_hidden[batch_ind, lex_select, :].unsqueeze(1)), dim=2)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            if step:
                decoder_outputs[:, step, :self.hidden_size*4] = torch.cat((decoder_output, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1), 
                                            anc_hidden[anc_leaf_select, depths[:, step], :].unsqueeze(1), 
                                            anc_syn_hidden[anc_leaf_select, depths[:, step], :].unsqueeze(1)), dim=2).squeeze(1)
            else:
                decoder_outputs[:, step, :self.hidden_size*4] = torch.cat((decoder_output, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1), Variable(torch.zeros(batch_size, 1, self.hidden_size * 2).cuda())), dim=2).squeeze(1)

        return decoder_outputs, ans_embed

    def masked_loss(self, logits, target, lengths, mask, rule_select=None, nce=False):
        batch_size = logits.size(0)
        max_len = lengths.data.max()
        l_mask  = torch.arange(max_len).long().cuda().expand(batch_size, max_len) < lengths.data.expand(max_len, batch_size).transpose(0, 1)
        if rule_select is not None:
            rule_select[~l_mask.expand(self.rule_vocab_size, batch_size, max_len).permute(1, 2, 0)] = True
            '''
            _logits = torch.zeros_like(logits)
            _logits[rule_select] = logits[rule_select]
            _logits[~rule_select] = -np.inf
            logits = _logits
            '''
            _logits = logits.clone()
            _logits[~rule_select] = -10e8
            logits = _logits
        '''
        logits_flat = logits.view(-1, logits.size(-1))
        log_probs_flat = F.log_softmax(logits_flat, dim=1)
        '''
        log_probs_flat = F.log_softmax(logits, dim=2).view(-1, logits.size(-1))
        target_flat = target.contiguous().view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        if nce:
            neg_log_probs_flat = torch.log(1 + 10e-8 - F.softmax(logits, dim=2)).view(-1, logits.size(-1))
            losses_flat += -neg_log_probs_flat.sum(1).unsqueeze(1) + torch.gather(neg_log_probs_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*target.size())
        _mask = Variable(l_mask * mask)
        losses = losses * _mask.float()
        loss = losses.sum() / _mask.float().sum()
        return loss

    def loss(self, indices, trg_seqs, trg_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, word_mask, rule_mask, positions, ancestors, anc_lengths):
        decoder_outputs, tree_input = self.forward(indices, trg_seqs, trg_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, positions, ancestors, anc_lengths)
        batch_size, trg_len = trg_seqs[0].size(0), trg_seqs[0].size(1)
        if self.lex_level == 0:
            words = self.lex_dist(self.lex_out(F.tanh(self.lex_hidden_out(decoder_outputs))))
            rules = self.rule_dist(F.tanh(self.rule_hidden_out(decoder_outputs)))
        elif self.lex_level >= 2:
            tags = self.constituent(Variable(trg_seqs[2]).cuda())
            words = self.lex_dist(self.lex_out(F.tanh(self.lex_hidden_out(decoder_outputs))))
            words_embed = self.lexicon(Variable(trg_seqs[1]).cuda())
            nts = self.nt_dist(self.nt_out(F.tanh(self.nt_hidden_out(torch.cat((decoder_outputs, words_embed), dim=2)))))
            rules = self.rule_dist(self.rule_out(F.tanh(self.rule_hidden_out(torch.cat((decoder_outputs, tags, words_embed), dim=2)))))

        word_loss = self.masked_loss(words, Variable(trg_seqs[1]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), word_mask.cuda().byte())
        if self.lex_level == 0:
            rule_loss = self.masked_loss(rules, Variable(rule_seqs).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), rule_mask.cuda().byte())
        else:
            rule_select = self.rules_by_nt[trg_seqs[2].view(-1).cuda()].view(batch_size, trg_len, -1)
            rule_select[:, :, 1] = True
            rule_loss = self.masked_loss(rules, Variable(rule_seqs).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), rule_mask.cuda().byte(), rule_select)
        if self.lex_level == 0:
            nt_loss = 0
        else:
            nt_loss = self.masked_loss(nts, Variable(trg_seqs[2]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), rule_mask.cuda().byte())
            #nt_loss = self.masked_loss(nts, Variable(trg_seqs[2]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), torch.ones_like(rule_mask).cuda().byte())
        return word_loss, nt_loss, rule_loss


class DummyLexicalizedGrammarDecoder(nn.Module):
    def __init__(self, lex_input_size, nt_input_size, rule_input_size, 
                 context_size, hidden_size, 
                 lex_vocab_size, nt_vocab_size, rule_vocab_size, 
                 lex_vectors, nt_vectors, rule_vectors,
                 lex_dictionary, nt_dictionary, rule_dictionary,
                 dropout, lex_level, data):
        super(DummyLexicalizedGrammarDecoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.lex_input_size = lex_input_size
        self.nt_input_size = nt_input_size
        self.rule_input_size = rule_input_size
        self.lex_vocab_size = lex_vocab_size
        self.nt_vocab_size = nt_vocab_size
        self.rule_vocab_size = rule_vocab_size
        self.lex_level = lex_level
        self.lexicon = Embedding(lex_vocab_size, lex_input_size, lex_vectors, trainable=True)
        self.constituent = Embedding(nt_vocab_size, nt_input_size, nt_vectors, trainable=True)
        self.rule = Embedding(rule_vocab_size, rule_input_size, rule_vectors, trainable=True)
        self.depth_embed_size = 300
        self.depth = Embedding(10, self.depth_embed_size, trainable=True)
        self.eod = Embedding(1, 300, trainable=True)
        self.breadth = Embedding(40, 256, trainable=True)
        self.pos_embed_size = 0
        self.encoder = nn.LSTM(lex_input_size, hidden_size)
        self.lm_encoder = nn.LSTM(lex_input_size, hidden_size)
        self.eou = lex_dictionary['__eou__']
        if data in ['persona', 'movie']:
            self.syn_encoder = nn.LSTM(lex_input_size, hidden_size)
        else:
            self.syn_encoder = nn.LSTM(nt_input_size + rule_input_size, hidden_size)
        self.context_size = context_size
        self.data = data
        
        #self.lex_out = nn.Linear(hidden_size + lex_input_size, lex_input_size)
        self.lex_out = nn.Linear(hidden_size * 2, lex_input_size)
        self.nt_dist = nn.Linear(nt_input_size, nt_vocab_size)
        #self.nt_dist.weight = self.constituent.weight
        self.tree_input_size = lex_input_size * 2
        self.tree = nn.Linear(lex_input_size * 4 + nt_input_size * 0 + rule_input_size * 0, self.tree_input_size)
        self.leaf_decoder = nn.LSTM(lex_input_size, hidden_size, batch_first=True, dropout=0.0)
        self.lm_decoder = nn.LSTM(lex_input_size, hidden_size, batch_first=True, dropout=0.0)
        self.lex_decoder = nn.LSTM(lex_input_size, hidden_size, batch_first=True, dropout=0.0)
        self.anc_decoder = nn.LSTM(lex_input_size, hidden_size, batch_first=True, dropout=0.0)
        self.anc_syn_decoder = nn.LSTM(nt_input_size + rule_input_size, hidden_size, batch_first=True, dropout=0.0)
        self.leaf_syn_decoder = nn.LSTM(nt_input_size, hidden_size, batch_first=True, dropout=0.0)
        self.decoder_input_size = nt_input_size + rule_input_size + self.depth_embed_size
        #self.decoder_input_size = lex_input_size * 6 + nt_input_size * 7

        self.leaf_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
        self.leaf_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
        self.leaf_cell_fc1 = nn.Linear(hidden_size, hidden_size)
        self.leaf_cell_fc2 = nn.Linear(hidden_size, hidden_size)

        self.leaf_syn_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
        self.leaf_syn_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
        self.leaf_syn_cell_fc1 = nn.Linear(hidden_size, hidden_size)
        self.leaf_syn_cell_fc2 = nn.Linear(hidden_size, hidden_size)

        self.lex_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
        self.lex_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
        self.lex_cell_fc1 = nn.Linear(hidden_size, hidden_size)
        self.lex_cell_fc2 = nn.Linear(hidden_size, hidden_size)

        self.anc_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
        self.anc_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
        self.anc_cell_fc1 = nn.Linear(hidden_size, hidden_size)
        self.anc_cell_fc2 = nn.Linear(hidden_size, hidden_size)

        self.anc_syn_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
        self.anc_syn_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
        self.anc_syn_cell_fc1 = nn.Linear(hidden_size, hidden_size)
        self.anc_syn_cell_fc2 = nn.Linear(hidden_size, hidden_size)

        self.rule_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
        self.rule_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
        self.rule_cell_fc1 = nn.Linear(hidden_size, hidden_size)
        self.rule_cell_fc2 = nn.Linear(hidden_size, hidden_size)

        self.lm_hidden_fc1 = nn.Linear(hidden_size, hidden_size)
        self.lm_hidden_fc2 = nn.Linear(hidden_size, hidden_size)
        self.lm_cell_fc1 = nn.Linear(hidden_size, hidden_size)
        self.lm_cell_fc2 = nn.Linear(hidden_size, hidden_size)

        #self.hidden_out_fc1 = nn.Linear(hidden_size * 3, hidden_size * 2)
        #self.hidden_out_fc2 = nn.Linear(hidden_size * 2, hidden_size * 1)
        self.decoder = nn.LSTM(self.decoder_input_size, hidden_size, batch_first=True, dropout=0.0)
        self.rule_decoder = nn.LSTM(self.rule_input_size, hidden_size, batch_first=True, dropout=0.0)
        
        if self.lex_level == 0:
            self.decoder_output_size = hidden_size * 2 + context_size * 1
        else:
            self.decoder_output_size = hidden_size * 3 + context_size * 1
        self.top_lex_hidden_out = nn.Linear(self.decoder_output_size, lex_input_size)
        self.leaf_weight = nn.Linear(hidden_size * 2, 2)
        if self.data in ['persona', 'movie']:
            self.merge_out = nn.Linear(hidden_size * 2, hidden_size)
            self.lex_hidden_out = nn.Linear(hidden_size * 2 + context_size * 1, lex_input_size)
            self.lex_hidden_out_0 = nn.Linear(hidden_size + context_size * 1, lex_input_size)
            self.lex_hidden_out_1 = nn.Linear(hidden_size + context_size * 1, lex_input_size)
            self.lex_hidden_out_2 = nn.Linear(hidden_size + context_size * 1, lex_input_size)
            self.rule_hidden_out = nn.Linear(hidden_size * 1, rule_input_size)
            self.nt_hidden_out = nn.Linear(hidden_size + context_size, nt_input_size)
        else:
            self.lex_hidden_out = nn.Linear(hidden_size * 1 + context_size * 1, lex_input_size)
            self.lex_hidden_out_2 = nn.Linear(hidden_size + context_size, lex_input_size)
            self.rule_hidden_out = nn.Linear(hidden_size * 2 + context_size * 1, rule_input_size)
        #self.nt_hidden_out = nn.Linear(hidden_size * 4 + lex_input_size, hidden_size)

        self.rule_out = nn.Linear(hidden_size * 2, rule_input_size)
        self.rule_dist = nn.Linear(rule_input_size, rule_vocab_size)
        self.rule_dist.weight = self.rule.weight
        self.nt_dist.weight = self.constituent.weight
        #self.rule_out = nn.Linear(hidden_size, rule_input_size)
        self.lex_dist = nn.Linear(lex_input_size, lex_vocab_size)
        self.lex_dist.weight = self.lexicon.weight
        self.hidden_fc1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_fc2 = nn.Linear(hidden_size, hidden_size)
        self.cell_fc1 = nn.Linear(hidden_size, hidden_size)
        self.cell_fc2 = nn.Linear(hidden_size, hidden_size)
        self.lex_dictionary = lex_dictionary
        self.nt_dictionary = nt_dictionary
        self.rule_dictionary = rule_dictionary
        self.eou = lex_dictionary['__eou__']
        self.start = lex_dictionary['<start>']
        #self.lex_dist.weight = self.lexicon.weight

        self.key_size = 200
        if self.data in ['persona', 'movie']:
            self.merge_for_key = nn.Linear(hidden_size * 2, hidden_size)
            self.lm_a_key = nn.Linear(hidden_size * 1, self.key_size)
            self.leaf_a_key = nn.Linear(hidden_size * 1, self.key_size)
            self.anc_a_key = nn.Linear(hidden_size * 1, self.key_size)
            self.lex_a_key = nn.Linear(hidden_size * 1, self.key_size)
            self.lm_p_key = nn.Linear(hidden_size * 2, self.key_size)
            self.leaf_p_key = nn.Linear(hidden_size * 2, self.key_size)
            self.anc_p_key = nn.Linear(hidden_size * 2, self.key_size)
            self.lex_p_key = nn.Linear(hidden_size * 2, self.key_size)
            self.lm_q_key = nn.Linear(hidden_size, self.key_size)
            self.leaf_q_key = nn.Linear(hidden_size, self.key_size)
            self.anc_q_key = nn.Linear(hidden_size, self.key_size)
            self.lex_q_key = nn.Linear(hidden_size, self.key_size)
            self.psn_key = nn.Linear(hidden_size, self.key_size)
            self.lm_energy = nn.Linear(self.key_size, 1)
            self.leaf_energy = nn.Linear(self.key_size, 1)
            self.lm_q_value = nn.Linear(hidden_size, context_size)
            self.leaf_q_value = nn.Linear(hidden_size, context_size)
            self.anc_q_value = nn.Linear(hidden_size, context_size)
            self.lex_q_value = nn.Linear(hidden_size, context_size)
            self.lm_psn_value = nn.Linear(hidden_size, context_size)
            self.leaf_psn_value = nn.Linear(hidden_size, context_size)
            self.anc_psn_value = nn.Linear(hidden_size, context_size)
            self.lex_psn_value = nn.Linear(hidden_size, context_size)
        elif self.data == 'microsoft':
            self.a_key = nn.Linear(hidden_size * 2, 100)
            self.aa_key = nn.Linear(hidden_size * 1, 100)
            self.b_key = nn.Linear(hidden_size * 1, 100)
            self.qa_key = nn.Linear(hidden_size * 1, 100)
            self.qa_value = nn.Linear(hidden_size, context_size)
            self.qb_key = nn.Linear(hidden_size * 1, 100)
            self.qb_value = nn.Linear(hidden_size, context_size)

        self.init_forget_bias(self.encoder, 0)
        self.init_forget_bias(self.decoder, 0)
        self.init_forget_bias(self.leaf_decoder, 0)
        self.init_forget_bias(self.lm_decoder, 0)
        if self.lex_level > 0:
            self.init_forget_bias(self.lex_decoder, 0)
            self.init_forget_bias(self.anc_decoder, 0)

    def init_rules(self, id2word):
        if self.nt_vocab_size > 3:
            self.ROOT = self.nt_dictionary['ROOT']
        else:
            self.ROOT = self.nt_dictionary['NT']
        self.rules_by_id = defaultdict(str)
        self.nt_by_id = defaultdict(str)
        rules_by_nt = defaultdict(list)
        self.id2word = id2word

        if self.lex_level > 0:
            self.preterminal = defaultdict(bool)
            if self.nt_vocab_size > 3:
                for pt in Preterminal:
                    self.preterminal[pt] = True
            else:
                self.preterminal['PT'] = True

        non_ending_rules = [1, 2]
        ending_rules = []
        for k, v in self.rule_dictionary.items():
            self.rules_by_id[v] = k[6:]
            if v not in [1, 2]:
                for nt in k.split()[1:]:
                    rules_by_nt[nt].append(self.rule_dictionary[k])
                #if self.lex_level == 3:
                if False:
                    if any(not self.preterminal[x] for x in k.split()[1:]):
                        non_ending_rules.append(v)
                    else:
                        ending_rules.append(v)
                else:
                    if len(k.split()) == 2 and k[-5:] != '_ROOT':
                        ending_rules.append(v)
                    else:
                        non_ending_rules.append(v)
        self.non_ending_rules = torch.cuda.LongTensor(non_ending_rules)
        self.ending_rules = ending_rules

        for k, v in self.nt_dictionary.items():
            self.nt_by_id[v] = k

        #self.rules_by_nt = {}
        self.rules_by_nt = torch.cuda.ByteTensor(self.nt_vocab_size, self.rule_vocab_size).fill_(False)
        for k, v in rules_by_nt.items():
            if k[-5:] != '_ROOT':
                vv = torch.cuda.ByteTensor(self.rule_vocab_size).fill_(False)
                vv[v] = True
                self.rules_by_nt[self.nt_dictionary[k]] = vv

        if self.lex_level > 0:
            self.root_rules = torch.cuda.ByteTensor(self.rule_vocab_size).fill_(False)
            for k, v in self.rule_dictionary.items():
                if k[-5:] == '_ROOT':
                    self.root_rules[v] = True

    def init_bin_rules(self, id2word):
        self.ROOT = self.nt_dictionary['NT']
        self.rules_by_id = defaultdict(str)
        self.nt_by_id = defaultdict(str)
        rules_by_nt = defaultdict(list)
        self.id2word = id2word

        if self.lex_level != 2:
            self.preterminal = defaultdict(bool)
            for pt in ['PT']:
                self.preterminal[pt] = True

        non_ending_rules = [1, 2]
        ending_rules = []
        for k, v in self.rule_dictionary.items():
            self.rules_by_id[v] = k[6:]
            if v not in [1, 2]:
                for nt in k.split()[1:]:
                    rules_by_nt[nt].append(self.rule_dictionary[k])
                if self.lex_level == 3:
                    if any(not self.preterminal[x] for x in k.split()[1:]):
                        non_ending_rules.append(v)
                    else:
                        ending_rules.append(v)
        self.non_ending_rules = torch.cuda.LongTensor(non_ending_rules)
        self.ending_rules = torch.cuda.LongTensor(ending_rules)

        for k, v in self.nt_dictionary.items():
            self.nt_by_id[v] = k

        #self.rules_by_nt = {}
        self.rules_by_nt = torch.cuda.ByteTensor(self.nt_vocab_size, self.rule_vocab_size).fill_(False)
        for k, v in rules_by_nt.items():
            vv = torch.cuda.ByteTensor(self.rule_vocab_size).fill_(False)
            vv[v] = True
            self.rules_by_nt[self.nt_dictionary[k]] = vv

    def check_grammar(self, nt_word, nt_rule, word_rule, word_word):
        self.nt_word = torch.cuda.ByteTensor(nt_word.shape[0], nt_word.shape[1]).fill_(True)
        self.nt_rule = torch.cuda.ByteTensor(self.nt_vocab_size, self.rule_vocab_size).fill_(False)
        self.word_rule = torch.cuda.ByteTensor(self.lex_vocab_size, self.rule_vocab_size).fill_(False)
        self.nt_dictionary[''] = 0
        #function_words = [self.nt_dictionary[x] for x in ['CC', 'DT', 'EX', 'IN', 'MD', 'TO', 'WDT', 'WP', 'WP$', 'WRB', 'PRP', 'PRP$']]
        for x in range(nt_word.shape[1]):
            if nt_word[:, x].sum() > 0:
                prob = nt_word[:, x] / nt_word[:, x].sum()
                if self.data == 'persona':
                    self.nt_word[:, x] = torch.cuda.ByteTensor((prob > 0.2).tolist())
                else:
                    self.nt_word[:, x] = torch.cuda.ByteTensor((prob > 0.1).tolist())

        for x in range(self.rule_vocab_size):
            '''
            if nt_rule[:, x].sum() > 0:
                prob = nt_rule[:, x] / nt_rule[:, x].sum()
                if self.data == 'persona':
                    if nt_rule[:, x].sum() >= 20000:
                        self.nt_rule[:, x] = torch.cuda.ByteTensor((prob > 0.15).tolist())
                    else:
                        self.nt_rule[:, x] = torch.cuda.ByteTensor((prob != 0).tolist())
                else:
                    self.nt_rule[:, x] = torch.cuda.ByteTensor((prob != 0).tolist())
            '''
            contains = [nt for nt in range(1, self.nt_vocab_size) if self.nt_by_id[nt] in self.rules_by_id[x].split() and self.rules_by_id[x][-5:] != '_ROOT']
            if contains:
                self.nt_rule[contains, [x]] = True

        #if self.lex_level == 3:
        if False:
            self.rule_by_last_nt = torch.cuda.ByteTensor(self.nt_vocab_size, self.rule_vocab_size).fill_(False)
            for k, v in self.rule_dictionary.items():
                if k not in ['<UNK>', 'RULE: EOD']:
                    nt = self.nt_dictionary[k.split()[-1]]
                    self.rule_by_last_nt[nt, v] = True

            self.rule_by_word = torch.cuda.ByteTensor(self.lex_vocab_size, self.rule_vocab_size).fill_(False)
            for k, v in self.lex_dictionary.items():
                nt = self.nt_word[:, v].cpu().numpy()
                ind = self.rule_by_last_nt[np.where(nt == 1)[0].tolist()].sum(0)
                self.rule_by_word[v] = ind != 0

        if self.lex_level > 0:
            self.root_nt_word = torch.cuda.ByteTensor(self.rule_vocab_size, self.lex_vocab_size).fill_(False)
            for x in range(1, self.lex_vocab_size):
                for y in range(1, self.nt_vocab_size):
                    if self.nt_word[y, x] == True:
                        nt = self.nt_by_id[y]
                        rule = 'RULE: {}_ROOT'.format(nt)
                        if rule in self.rule_dictionary:
                            self.root_nt_word[self.rule_dictionary[rule], x] = True
            self.word_word = (torch.from_numpy(word_word) > 0).cuda()

    def init_forget_bias(self, rnn, b):
        for names in rnn._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2 
                bias.data[start:end].fill_(b)

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.syn_encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        self.leaf_decoder.flatten_parameters()
        if self.lex_level > 0:
            self.lex_decoder.flatten_parameters()
            self.anc_decoder.flatten_parameters()
            self.anc_syn_decoder.flatten_parameters()
            self.lm_decoder.flatten_parameters()

    def hidden_transform(self, hidden, prefix):
        return eval('self.{}hidden_fc2(F.relu(self.{}hidden_fc1(hidden)))'.format(prefix, prefix))

    def cell_transform(self, cell, prefix):
        return eval('self.{}cell_fc2(F.relu(self.{}cell_fc1(cell)))'.format(prefix, prefix))

    def init_hidden(self, src_hidden, prefix=''):
        hidden = src_hidden[0]
        cell = src_hidden[1]
        hidden = self.hidden_transform(hidden, prefix)
        cell = self.cell_transform(cell, prefix)
        return (hidden, cell)

    def attention2(self, decoder_hidden, src_hidden, src_max_len, src_lengths, src_last_hidden, name):
        l = 'a' if name == 'lex' else 'b'
        if decoder_hidden[0].size(2) == self.hidden_size * 2 or name == 'syn':
            a_key = eval('F.tanh(self.{}_key(decoder_hidden[0].squeeze(0)))'.format(l))
        else:
            a_key = F.tanh(self.aa_key(decoder_hidden[0].squeeze(0)))
        q_key = eval('F.tanh(self.q{}_key(src_hidden))'.format(l))
        q_value = eval('F.tanh(self.q{}_value(src_hidden))'.format(l))
        q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
        q_mask  = torch.arange(src_max_len).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(src_max_len, 1).transpose(0, 1)
        q_energy[~q_mask] = -np.inf
        q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
        context = torch.bmm(q_weights, q_value)
        return context

    def attention(self, decoder_hidden, src_hidden, src_max_len, src_lengths, src_last_hidden, name):
        if decoder_hidden[0].size(2) == self.hidden_size * 2:
            hidden = F.tanh(self.merge_for_key(decoder_hidden[0]))
        else:
            hidden = decoder_hidden[0]
        a_key = eval('self.{}a_key(hidden.squeeze(0))'.format(name))
        #p_key = eval('self.{}p_key(torch.cat((hidden.squeeze(0), src_last_hidden[0].squeeze(0)), dim=1))'.format(name))
        q_key = eval('self.{}q_key(src_hidden)'.format(name))
        #psn_key = self.psn_key(psn_hidden)
        #q_value = eval('self.{}q_value(src_hidden)'.format(name))
        q_value = src_hidden
        #psn_value = eval('self.{}psn_value(psn_hidden)'.format(name))
        q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
        #q_energy = eval('self.{}energy(F.tanh(q_key + a_key.unsqueeze(1))).squeeze(2)'.format(name))
        #psn_energy = torch.bmm(psn_key, p_key.unsqueeze(2)).squeeze(2)
        q_mask = torch.arange(src_max_len).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(src_max_len, 1).transpose(0, 1)
        #psn_mask = torch.arange(psn_max_len).long().cuda().repeat(psn_hidden.size(0), 1) < psn_lengths.cuda().repeat(psn_max_len, 1).transpose(0, 1)
        q_energy[~q_mask] = -np.inf
        #psn_energy[~psn_mask] = -np.inf
        q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
        '''
        psn_weights = F.softmax(psn_energy, dim=1).unsqueeze(1)
        q_weights = F.sigmoid(q_energy).unsqueeze(1)
        psn_weights = F.sigmoid(psn_energy).unsqueeze(1)
        '''
        q_context = torch.bmm(q_weights, q_value)
        #psn_context = torch.bmm(psn_weights, psn_value)

        return q_context
        #return torch.cat((q_context, psn_context), dim=2)

    def encode(self, tar_seqs, indices, name, init_hidden):
        batch_size = tar_seqs[0].size(0)
        lengths = [ind.max().item() for ind in indices]
        max_len = max(lengths)
        mask = [x.copy() for x in indices]
        _indices = [None for x in indices]
        for x in range(len(indices)):
            mask[x][1:] -= mask[x][:-1].copy()
            _indices[x] = np.zeros(tar_seqs[0].size(1), dtype=np.int64)
            _indices[x][1:len(indices[x])] = indices[x][:-1]
            _indices[x][0] = 0
        if name == 'leaf_syn':
            tar_lex_embed = self.constituent(Variable(tar_seqs[0]).cuda())
        else:
            tar_lex_embed = self.lexicon(Variable(tar_seqs[1]).cuda())
        #tar_nt_embed = self.constituent(Variable(tar_seqs[0]).cuda())
        #_tar_embed = torch.cat((tar_lex_embed, tar_nt_embed), dim=2)
        _tar_embed = tar_lex_embed
        tar_embed = Variable(torch.zeros(_tar_embed.size(0), max_len, _tar_embed.size(2)).cuda())
        for x in range(tar_embed.size(0)):
            ind = torch.from_numpy(np.arange(len(mask[x]))[mask[x].astype(bool)]).long().cuda()
            tar_embed[x, :lengths[x], :] = _tar_embed[x][ind]
        tar_embed = self.drop(tar_embed)
        t_lengths, perm_idx = torch.LongTensor(lengths).sort(0, descending=True)
        tar_embed = tar_embed[perm_idx.cuda()]
        packed_input = pack_padded_sequence(tar_embed, t_lengths.numpy(), batch_first=True)
        hidden, _ = eval('self.{}_decoder(packed_input, init_hidden)'.format(name))
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        hidden = hidden[perm_idx.sort()[1].cuda()]

        return hidden, _indices

    def leaf_encode(self, tar_seqs, indices, name, init_hidden):
        batch_size = tar_seqs[0].size(0)
        lengths = [ind.max().item() + 1 for ind in indices]
        max_len = max(lengths)
        mask = [x.copy() for x in indices]
        _indices = [None for x in indices]
        for x in range(len(indices)):
            mask[x][1:] -= mask[x][:-1].copy()
            mask[x][0] = 1
            _indices[x] = np.zeros(tar_seqs[0].size(1) + 1, dtype=np.int64)
            _indices[x][1:len(indices[x])] = indices[x][:-1]
            _indices[x][0] = 0
            _indices[x][len(indices[x])] = indices[x][-1]
        tar_lex_embed = self.lm_lexicon(Variable(torch.cat((torch.LongTensor(batch_size).fill_(self.start).unsqueeze(1), tar_seqs[1]), dim=1).cuda()))
        #tar_lex_embed = self.lm_lexicon(Variable(tar_seqs[1]).cuda())
        #tar_nt_embed = self.constituent(Variable(tar_seqs[0]).cuda())
        #_tar_embed = torch.cat((tar_lex_embed, tar_nt_embed), dim=2)
        _tar_embed = tar_lex_embed
        tar_embed = Variable(torch.zeros(_tar_embed.size(0), max_len, _tar_embed.size(2)).cuda())
        for x in range(tar_embed.size(0)):
            ind = torch.from_numpy(np.arange(len(mask[x]))[mask[x].astype(bool)]).long().cuda()
            tar_embed[x, :lengths[x], :] = _tar_embed[x][ind]
        t_lengths, perm_idx = torch.LongTensor(lengths).sort(0, descending=True)
        tar_embed = tar_embed[perm_idx.cuda()]
        tar_embed = self.drop(tar_embed)
        packed_input = pack_padded_sequence(tar_embed, t_lengths.numpy(), batch_first=True)
        hidden, _ = eval('self.{}_decoder(packed_input, init_hidden)'.format(name))
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        hidden = hidden[perm_idx.sort()[1].cuda()]

        return hidden, _indices

    def encode_anc(self, anc_embed, anc_lengths, trg_lengths, batch_size, src_last_hidden, name):
        _src_last_hidden = Variable(torch.cuda.FloatTensor(len(anc_lengths), self.hidden_size))
        _src_last_cell = Variable(torch.cuda.FloatTensor(len(anc_lengths), self.hidden_size))
        start = 0
        for x in range(batch_size):
            _src_last_hidden[start:start+trg_lengths[x], :] = src_last_hidden[0][:, x, :].expand(trg_lengths[x], self.hidden_size)
            _src_last_cell[start:start+trg_lengths[x], :] = src_last_hidden[1][:, x, :].expand(trg_lengths[x], self.hidden_size)
            start += trg_lengths[x]

        f_lengths, perm_idx = torch.LongTensor(anc_lengths).sort(0, descending=True)
        anc_embed = anc_embed[perm_idx.cuda()]
        anc_embed = self.drop(anc_embed)
        #nonzeros = f_lengths.nonzero().squeeze(1)
        #zeros = (f_lengths == 0).nonzero().squeeze(1)
        #packed_input = pack_padded_sequence(anc_embed[nonzeros.cuda()], f_lengths[nonzeros].numpy(), batch_first=True)
        packed_input = pack_padded_sequence(anc_embed, f_lengths.numpy(), batch_first=True)
        anc_init_hidden = self.init_hidden((_src_last_hidden[perm_idx.cuda()].unsqueeze(0), _src_last_cell[perm_idx.cuda()].unsqueeze(0)), name) 
        anc_output, anc_last_hidden = eval('self.{}decoder(packed_input, anc_init_hidden)'.format(name))
        '''
        _anc_hidden = Variable(torch.cuda.FloatTensor(len(anc_lengths), self.hidden_size).fill_(0))
        _anc_hidden[:nonzeros.size(0)] = anc_last_hidden[0].squeeze(0)
        _anc_hidden = _anc_hidden[perm_idx.sort()[1].cuda()]
        _anc_hidden[perm_idx[zeros].cuda()] = anc_init_hidden[0].squeeze(0)
        '''
        _anc_hidden = anc_last_hidden[0].squeeze(0)
        _anc_hidden = _anc_hidden[perm_idx.sort()[1].cuda()]

        anc_hidden = Variable(torch.cuda.FloatTensor(batch_size, max(trg_lengths), self.hidden_size).fill_(0))
        start = 0
        for x in range(batch_size):
            anc_hidden[x, :trg_lengths[x], :] = _anc_hidden[start:start+trg_lengths[x]]
            start += trg_lengths[x]

        return anc_hidden, self.init_hidden(src_last_hidden, name)

    def forward(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, rule_seqs, lex_seqs, leaf_indices, lex_indices, positions, ancestors, anc_lengths):
        batch_size = src_seqs.size(0)

        src_embed = self.lexicon(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        #decoder_hidden = self.init_hidden(src_last_hidden) 
        decoder_hidden = self.init_hidden(src_last_hidden) 

        '''
        psn_lengths, perm_idx = torch.LongTensor(psn_lengths).sort(0, descending=True)
        if self.data == 'persona':
            psn_embed = self.lexicon(Variable(psn_seqs).cuda())
        else:
            psn_embed = torch.cat((self.constituent(Variable(psn_seqs[0]).cuda()), self.rule(Variable(psn_seqs[1]).cuda())), dim=2)
        psn_embed = psn_embed[perm_idx.cuda()]
        packed_input = pack_padded_sequence(psn_embed, psn_lengths.numpy(), batch_first=True)
        if self.data == 'persona':
            psn_output, psn_last_hidden = self.encoder(packed_input)
        else:
            psn_output, psn_last_hidden = self.syn_encoder(packed_input)
        psn_hidden, _ = pad_packed_sequence(psn_output, batch_first=True)
        psn_hidden = psn_hidden[perm_idx.sort()[1].cuda()]
        psn_lengths = psn_lengths[perm_idx.sort()[1]]
        '''

        leaf_init_hidden = self.init_hidden(src_last_hidden, 'leaf_') 
        leaf_decoder_hidden, _leaf_indices = self.encode(trg_seqs, leaf_indices, 'leaf', leaf_init_hidden)
        leaf_decoder_hidden = torch.cat((leaf_init_hidden[0].transpose(0, 1), leaf_decoder_hidden), dim=1)

        lm_decoder_hidden = self.init_hidden(src_last_hidden, 'lm_') 
        #lm_decoder_hidden, lm_indices = self.leaf_encode(trg_seqs, leaf_indices, 'lm', lm_init_hidden)
        #lm_decoder_hidden = torch.cat((lm_init_hidden[0].transpose(0, 1), lm_decoder_hidden), dim=1)

        '''
        leaf_syn_init_hidden = self.init_hidden(src_last_hidden, 'leaf_syn_') 
        leaf_syn_decoder_hidden, leaf_syn_indices = self.encode(trg_seqs, leaf_indices, 'leaf_syn', leaf_syn_init_hidden)
        leaf_syn_decoder_hidden = torch.cat((leaf_syn_init_hidden[0].transpose(0, 1), leaf_syn_decoder_hidden), dim=1)
        '''

        '''
        #_indices = [np.arange(len(lex_indices[x])) + 1 for x in range(len(lex_indices))]
        lex_init_hidden = self.init_hidden(src_last_hidden, 'lex_') 
        lex_decoder_hidden, _lex_indices = self.encode(trg_seqs, lex_indices, 'lex', lex_init_hidden)
        lex_decoder_hidden = torch.cat((lex_init_hidden[0].transpose(0, 1), lex_decoder_hidden), dim=1)
        '''

        ans_nt = self.constituent(Variable(trg_seqs[0]).cuda())

        '''
        ans_rule_embed = rule_seqs.clone()
        ans_rule_embed[:, 1:] = ans_rule_embed[:, :-1]
        ans_rule_embed[:, 0] = 0
        ans_rule_embed = self.rule(Variable(ans_rule_embed).cuda())
        '''
        ans_rule_embed = self.rule(Variable(rule_seqs).cuda())

        anc_lex = torch.cat((torch.LongTensor([self.start for x in range(ancestors[0].size(0))]).unsqueeze(1), ancestors[0]), dim=1)
        anc_lengths = [l + 1 for l in anc_lengths]
        anc_embed = self.lexicon(Variable(anc_lex).cuda())
        anc_hidden, anc_init_hidden = self.encode_anc(anc_embed, anc_lengths, trg_lengths, batch_size, src_last_hidden, 'anc_')

        anc_syn_embed = torch.cat((self.constituent(Variable(ancestors[1]).cuda()), self.rule(Variable(ancestors[2]).cuda())), dim=2)
        #anc_syn_hidden, anc_syn_init_hidden = self.encode_anc(anc_syn_embed, anc_lengths, trg_lengths, batch_size, src_last_hidden, 'anc_syn_')

        depth = positions[0] + 1
        depth[depth > 9] = 9
        depth_embed = self.depth(Variable(depth).cuda())
        ans_embed = torch.cat((self.drop(ans_nt), self.drop(ans_rule_embed), self.drop(depth_embed)), dim=2)
        
        trg_l = max(trg_lengths)
        leaf_l = max(psn_lengths)
        batch_ind = torch.arange(batch_size).long().cuda()
        decoder_outputs = Variable(torch.FloatTensor(batch_size, trg_l, self.decoder_output_size).cuda())
        lm_decoder_outputs = Variable(torch.FloatTensor(batch_size, leaf_l, self.hidden_size + self.context_size * 1).cuda())
        if self.lex_level == 0:
            context = self.attention((leaf_init_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'leaf_')
        else:
            #context = self.attention((torch.cat((leaf_init_hidden[0], anc_init_hidden[0]), dim=2), None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths) 
            if self.data in ['persona', 'movie']:
                '''
                leaf_context = self.attention((leaf_init_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths, 'leaf_') 
                anc_context = self.attention((anc_init_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths, 'anc_') 
                #lex_context = self.attention((lex_init_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths, 'lex_') 
                '''
                #lm_context = self.attention((lm_init_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'lm_')
                #context = torch.cat((leaf_context, anc_context), dim=2)
                context = self.drop(self.attention((torch.cat((leaf_init_hidden[0], anc_init_hidden[0]), dim=2), None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'leaf_'))
            else:
                lm_decoder_outputs = Variable(torch.FloatTensor(batch_size, lm_decoder_hidden.size(1), self.hidden_size + self.context_size * 1).cuda())
                context = self.attention2((torch.cat((leaf_init_hidden[0], anc_init_hidden[0]), dim=2), None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'lex') 
                #syn_context = self.attention2((torch.cat((decoder_hidden[0], anc_syn_init_hidden[0]), dim=2), None), psn_hidden, psn_hidden.size(1), psn_lengths, psn_last_hidden, 'syn')
                syn_context = self.attention2((decoder_hidden[0], None), psn_hidden, psn_hidden.size(1), psn_lengths.tolist(), psn_last_hidden, 'syn')
                lm_context = self.attention2((lm_init_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'lex') 
        #pos_embed = torch.cat((self.depth(Variable(positions[0]).cuda()), self.breadth(Variable(positions[1] / 2).cuda())), dim=2)
        '''
        for step in range(lm_decoder_hidden.size(1)):
            lm_decoder_outputs[:, step, :self.hidden_size] = lm_decoder_hidden[:, step, :]
            if self.data == 'persona':
                lm_decoder_outputs[:, step, -self.context_size*1:] = lm_context.squeeze(1)
                lm_context = self.attention((lm_decoder_hidden[:, [step], :].transpose(0, 1), None), lm_src_hidden, lm_src_hidden.size(1), src_lengths, lm_src_last_hidden, 'lm_') 
            else:
                lm_decoder_outputs[:, step, -self.context_size:] = lm_context.squeeze(1)
                lm_context = self.attention2((lm_decoder_hidden[:, [step], :].transpose(0, 1), None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'lex') 
        '''
        if self.lex_level > 0:
            lm_decoder_input = self.lexicon(Variable(torch.cuda.LongTensor(batch_size).fill_(self.start))).unsqueeze(1)
            for step in range(leaf_l):
                lm_context = self.attention(lm_decoder_hidden, src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'lm_') 
                lm_decoder_output, lm_decoder_hidden = self.lm_decoder(lm_decoder_input, lm_decoder_hidden)
                lm_decoder_outputs[:, step, :self.hidden_size] = lm_decoder_output.squeeze(1)
                lm_decoder_outputs[:, step, -self.context_size*1:] = lm_context.squeeze(1)
                lm_decoder_input = self.lexicon(Variable(psn_seqs[:, step].cuda())).unsqueeze(1)
            
        for step in range(trg_l):
            #decoder_input = torch.cat((ans_embed[:, step, :].unsqueeze(1), tree_input[:, step, :].unsqueeze(1)), dim=2)
            decoder_input = ans_embed[:, step, :].unsqueeze(1)
            #rule_decoder_input = ans_rule_embed[:, step, :].unsqueeze(1)
            if self.lex_level > 0:
            #if False:
                leaf_select = torch.cuda.LongTensor([x[step].item() for x in _leaf_indices])
                #lex_select = torch.cuda.LongTensor([x[step].item() for x in _lex_indices])
                #decoder_input = torch.cat((decoder_input, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1), lex_decoder_hidden[batch_ind, lex_select, :].unsqueeze(1)), dim=2)
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                #rule_decoder_output, rule_decoder_hidden = self.rule_decoder(rule_decoder_input, rule_decoder_hidden)
                #dec_cat_output = torch.cat((decoder_output, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1), lex_decoder_hidden[batch_ind, lex_select, :].unsqueeze(1)), dim=2)
                #dec_cat_output = torch.cat((decoder_output, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1), anc_hidden[:, step, :].unsqueeze(1), anc_syn_hidden[:, step, :].unsqueeze(1)), dim=2)
                dec_cat_output = torch.cat((self.drop(decoder_output), self.drop(leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1)), self.drop(anc_hidden[:, step, :].unsqueeze(1))), dim=2)
                #dec_cat_output = torch.cat((decoder_output, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1), lex_decoder_hidden[batch_ind, lex_select, :].unsqueeze(1)), dim=2)
                #dec_cat_output = torch.cat((decoder_output, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1), anc_hidden[:, step, :].unsqueeze(1)), dim=2)
                decoder_outputs[:, step, :self.hidden_size*3] = dec_cat_output.squeeze(1)
                if self.data in ['persona', 'movie']:
                    decoder_outputs[:, step, -self.context_size*1:] = context.squeeze(1)
                else:
                    decoder_outputs[:, step, -self.context_size*2:-self.context_size] = context.squeeze(1)
                    decoder_outputs[:, step, -self.context_size:] = syn_context.squeeze(1)
                #decoder_outputs[:, step, self.hidden_size*3+self.context_size*2:] = tree_input[:, step, :]
                #context = self.attention((dec_cat_output[:, :, self.hidden_size:-self.hidden_size].transpose(0, 1), None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths) 
                if self.data in ['persona', 'movie']:
                    '''
                    leaf_context = self.attention((dec_cat_output.transpose(0, 1)[:, :, self.hidden_size:self.hidden_size*2], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths, 'leaf_') 
                    anc_context = self.attention((dec_cat_output.transpose(0, 1)[:, :, self.hidden_size*2:self.hidden_size*3], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths, 'anc_') 
                    #lex_context = self.attention((dec_cat_output.transpose(0, 1)[:, :, self.hidden_size*2:self.hidden_size*3], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths, 'lex_') 
                    context = torch.cat((leaf_context, anc_context), dim=2)
                    '''
                    context = self.drop(self.attention((dec_cat_output.transpose(0, 1)[:, :, self.hidden_size:self.hidden_size*3], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'leaf_'))
                else:
                    context = self.attention2((dec_cat_output.transpose(0, 1)[:, :, self.hidden_size:self.hidden_size*3], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'lex') 
                    '''
                    syn_context = self.attention2((torch.cat((dec_cat_output.transpose(0, 1)[:, :, :self.hidden_size], 
                                                              dec_cat_output.transpose(0, 1)[:, :, self.hidden_size*3:self.hidden_size*4]), 
                                                             dim=2), None
                                                  ), psn_hidden, psn_hidden.size(1), psn_lengths, psn_last_hidden, 'syn') 
                    '''
                    syn_context = self.attention2((dec_cat_output.transpose(0, 1)[:, :, :self.hidden_size], None), psn_hidden, psn_hidden.size(1), psn_lengths.tolist(), psn_last_hidden, 'syn')
            else:
                leaf_select = torch.cuda.LongTensor([x[step].item() for x in _leaf_indices])
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                dec_cat_output = torch.cat((decoder_output, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1)), dim=2)
                decoder_outputs[:, step, :self.hidden_size * 2] = dec_cat_output.squeeze(1)
                decoder_outputs[:, step, self.hidden_size * 2:] = context.squeeze(1)
                context = self.attention((dec_cat_output.transpose(0, 1)[:, :, self.hidden_size:self.hidden_size*2], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'leaf_') 

        return decoder_outputs, lm_decoder_outputs

    def masked_loss(self, logits, target, lengths, mask, rule_select=None, nce=False):
        batch_size = logits.size(0)
        max_len = lengths.data.max()
        #max_len = logits.size(1)
        l_mask  = torch.arange(max_len).long().cuda().expand(batch_size, max_len) < lengths.data.expand(max_len, batch_size).transpose(0, 1)
        if rule_select is not None:
            rule_select[~l_mask.expand(self.rule_vocab_size, batch_size, max_len).permute(1, 2, 0)] = True
            '''
            _logits = torch.zeros_like(logits)
            _logits[rule_select] = logits[rule_select]
            _logits[~rule_select] = -np.inf
            logits = _logits
            '''
            _logits = logits.clone()
            _logits[~rule_select] = -10e8
            logits = _logits
        '''
        logits_flat = logits.view(-1, logits.size(-1))
        log_probs_flat = F.log_softmax(logits_flat, dim=1)
        '''
        log_probs_flat = F.log_softmax(logits, dim=2).view(-1, logits.size(-1))
        target_flat = target.contiguous().view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        if nce:
            neg_log_probs_flat = torch.log(1 + 10e-8 - F.softmax(logits, dim=2)).view(-1, logits.size(-1))
            losses_flat += -neg_log_probs_flat.sum(1).unsqueeze(1) + torch.gather(neg_log_probs_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*target.size())
        _mask = Variable(l_mask * mask)
        losses = losses * _mask.float()
        loss = losses.sum() / _mask.float().sum()
        return loss

    def loss(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, rule_seqs, lex_seqs, leaf_indices, lex_indices, word_mask, rule_mask, noun_mask, positions, ancestors, anc_lengths):
        decoder_outputs, lm_decoder_hidden = self.forward(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, rule_seqs, lex_seqs, leaf_indices, lex_indices, positions, ancestors, anc_lengths)
        batch_size, trg_len = trg_seqs[0].size(0), trg_seqs[0].size(1)

        tags = self.constituent(Variable(trg_seqs[2]).cuda())
        #words = self.lex_dist(self.lex_out(F.tanh(self.lex_hidden_out(decoder_outputs))))
        if self.data in ['persona', 'movie']:
            if self.lex_level == 0:
                word_logits = self.lex_dist(F.tanh(self.lex_hidden_out(decoder_outputs[:, :, self.hidden_size:])))
            else:
                word_logits = self.lex_dist(F.tanh(self.lex_hidden_out(torch.cat((decoder_outputs[:, :, self.hidden_size:self.hidden_size*3], decoder_outputs[:, :, -self.context_size*1:]), dim=2))))
                #word_logits = self.lex_dist(F.tanh(self.lex_hidden_out(torch.cat((F.tanh(self.merge_out(decoder_outputs[:, :, self.hidden_size:self.hidden_size*3])), decoder_outputs[:, :, -self.context_size*1:]), dim=2))))
                '''
                words_1 = self.lex_dist(F.tanh(self.lex_hidden_out_0(torch.cat((decoder_outputs[:, :, self.hidden_size:self.hidden_size*2], decoder_outputs[:, :, -self.context_size*1:]), dim=2))))
                words_2 = self.lex_dist(F.tanh(self.lex_hidden_out_1(torch.cat((decoder_outputs[:, :, self.hidden_size*2:self.hidden_size*3], decoder_outputs[:, :, -self.context_size*1:]), dim=2))))
                leaf_weight = F.softmax(self.leaf_weight(decoder_outputs[:, :, self.hidden_size:self.hidden_size*3]), dim=2)
                word_logits = torch.log(F.softmax(words_1, dim=2) * leaf_weight[:, :, [0]] + F.softmax(words_2, dim=2) * leaf_weight[:, :, [1]] + 10e-12)
                '''
        else:
            word_logits = self.lex_dist(F.tanh(self.lex_hidden_out(torch.cat((decoder_outputs[:, :, self.hidden_size:self.hidden_size*3], decoder_outputs[:, :, -self.context_size*2:-self.context_size]), dim=2))))
        words = torch.cuda.LongTensor(batch_size, lm_decoder_hidden.size(1)).fill_(0)
        nts = torch.cuda.LongTensor(batch_size, lm_decoder_hidden.size(1)).fill_(0)
        leaf_mask = torch.cuda.ByteTensor(batch_size, lm_decoder_hidden.size(1)).fill_(False)
        leaf_lengths = torch.zeros(batch_size).long().cuda()
        for x in range(batch_size):
            ind = leaf_indices[x].copy()
            l_num = ind[-1].item()
            leaf_lengths[x] = l_num + 1
            leaf_mask[x, :l_num + 1] = True
            ind[1:] -= ind[:-1]
            words[x, :l_num] = trg_seqs[1][x][np.where(ind==1)[0].tolist()]
            words[x, l_num] = self.eou
            nts[x, :l_num] = trg_seqs[0][x][np.where(ind==1)[0].tolist()]
        words_embed = self.lexicon(Variable(trg_seqs[1]).cuda())
        #nts = self.nt_dist(self.nt_out(F.tanh(self.nt_hidden_out(torch.cat((decoder_outputs[:, :, :self.hidden_size * 4], words_embed), dim=2)))))
        #rules = self.rule_dist(self.rule_out(F.tanh(self.rule_hidden_out(torch.cat((decoder_outputs[:, :, :self.hidden_size*4], words_embed), dim=2)))))
        #rules = self.rule_dist(self.rule_hidden_out(torch.cat((decoder_outputs[:, :, :self.hidden_size*4], words_embed), dim=2)))
        #rules = self.rule_dist(self.rule_hidden_out(decoder_outputs[:, :, :self.hidden_size*4]))
        if self.data in ['persona', 'movie']:
            #rules = self.rule_dist(self.rule_hidden_out(torch.cat((decoder_outputs[:, :, self.hidden_size*3:self.hidden_size*4], decoder_outputs[:, :, :self.hidden_size*2]), dim=2)))
            if self.lex_level == 0:
                rules = self.rule_dist(F.tanh(self.rule_hidden_out(decoder_outputs[:, :, :self.hidden_size*2])))
            else:
                #rules = self.rule_dist(F.tanh(self.rule_hidden_out(torch.cat((decoder_outputs[:, :, :self.hidden_size*1], F.tanh(self.merge_out(decoder_outputs[:, :, self.hidden_size:self.hidden_size*3]))), dim=2))))
                rules = self.rule_dist(F.tanh(self.rule_hidden_out(decoder_outputs[:, :, :self.hidden_size*1])))
        else:
            #rules = self.rule_dist(self.rule_hidden_out(torch.cat((decoder_outputs[:, :, self.hidden_size*3:self.hidden_size*4], decoder_outputs[:, :, :self.hidden_size*2], decoder_outputs[:, :, -self.context_size:]), dim=2)))
            rules = self.rule_dist(F.tanh(self.rule_hidden_out(torch.cat((decoder_outputs[:, :, :self.hidden_size*2], decoder_outputs[:, :, -self.context_size:]), dim=2))))

        if self.lex_level == 0:
            word_loss = self.masked_loss(word_logits, Variable(trg_seqs[1]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), word_mask.cuda().byte())
            lm_loss = torch.zeros_like(word_loss)
        else:
            word_loss = self.masked_loss(word_logits, Variable(trg_seqs[1]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), word_mask.cuda().byte())
            lm_loss = compute_perplexity(self.lex_dist(F.tanh(self.lex_hidden_out_2(lm_decoder_hidden))), Variable(psn_seqs.cuda()), Variable(torch.cuda.LongTensor(psn_lengths)))
        noun_loss = self.masked_loss(word_logits, Variable(trg_seqs[1]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), noun_mask.cuda().byte())
        #lm_loss = self.masked_loss(self.lm_lex_dist(F.tanh(self.lex_hidden_out_2(lm_decoder_hidden))), Variable(words), Variable(leaf_lengths), leaf_mask)

        rule_loss = self.masked_loss(rules, Variable(trg_seqs[3]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), rule_mask.cuda().byte())
        #nt_logits = self.nt_dist(F.tanh(self.nt_hidden_out(lm_decoder_hidden)))
        #nt_loss = self.masked_loss(nt_logits, Variable(nts), Variable(leaf_lengths), leaf_mask)
        #nt_loss = self.masked_loss(nts, Variable(trg_seqs[2]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), torch.ones_like(rule_mask).cuda().byte())
        return word_loss, lm_loss, rule_loss, noun_loss, noun_mask.sum()
        #return word_loss, nt_loss, rule_loss

    def generate(self, src_seqs, src_lengths, psn_seqs, psn_lengths, indices, max_len, beam_size, top_k, slots=None):
        self.syn_weight = 1
        self.lex_weight = 1
        self.lm_weight = .0
        self.rerank = False

        src_embed = self.lexicon(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)

        psn_lengths, perm_idx = torch.LongTensor(psn_lengths).sort(0, descending=True)
        if self.data in ['persona', 'movie']:
            psn_embed = self.lexicon(Variable(psn_seqs).cuda())
        else:
            psn_embed = torch.cat((self.constituent(Variable(psn_seqs[0]).cuda()), self.rule(Variable(psn_seqs[1]).cuda())), dim=2)
        psn_embed = psn_embed[perm_idx.cuda()]
        packed_input = pack_padded_sequence(psn_embed, psn_lengths.numpy(), batch_first=True)
        if self.data in ['persona', 'movie']:
            psn_output, psn_last_hidden = self.encoder(packed_input)
        else:
            psn_output, psn_last_hidden = self.syn_encoder(packed_input)
        psn_hidden, _ = pad_packed_sequence(psn_output, batch_first=True)
        psn_hidden = psn_hidden[perm_idx.sort()[1].cuda()]
        psn_lengths = psn_lengths[perm_idx.sort()[1]]

        decoder_hidden = self.init_hidden(src_last_hidden) 
        leaf_decoder_hidden = self.init_hidden(src_last_hidden, 'leaf_') 
        if self.lex_level == 0:
            context = self.attention((leaf_decoder_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'leaf_')
        else:
            lex_decoder_hidden = self.init_hidden(src_last_hidden, 'lex_')    
            anc_decoder_hidden = self.init_hidden(src_last_hidden, 'anc_')
            syn_decoder_hidden = self.init_hidden(src_last_hidden, 'anc_syn_')
            lm_decoder_hidden = self.init_hidden(src_last_hidden, 'lm_')
            if self.data in ['persona', 'movie']:
                '''
                leaf_context = self.attention((leaf_decoder_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths, 'leaf_') 
                anc_context = self.attention((anc_decoder_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths, 'anc_') 
                #lex_context = self.attention((lex_decoder_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths, 'lex_') 
                context = torch.cat((leaf_context, anc_context), dim=2)
                '''
                context = self.attention((torch.cat((leaf_decoder_hidden[0], anc_decoder_hidden[0]), dim=2), None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'leaf_') 
            else:
                context = self.attention2((torch.cat((leaf_decoder_hidden[0], anc_decoder_hidden[0]), dim=2), None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'lex')
                #syn_context = self.attention2((torch.cat((decoder_hidden[0], syn_decoder_hidden[0]), dim=2), None), psn_hidden, psn_hidden.size(1), psn_lengths, psn_last_hidden, 'syn')
                syn_context = self.attention2((decoder_hidden[0], None), psn_hidden, psn_hidden.size(1), psn_lengths.tolist(), psn_last_hidden, 'syn')

        batch_size = src_embed.size(0)
        assert batch_size == 1
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        #context = self.attention(decoder_hidden, src_hidden, src_hidden.size(1), src_lengths) 
        decoder_input = Variable(torch.zeros(batch_size, 1, self.decoder_input_size)).cuda()
        decoder_input[:, :, :self.nt_input_size] = self.constituent(Variable(torch.cuda.LongTensor([self.ROOT]))).unsqueeze(1)
        decoder_input[:, :, -300:] = self.depth(Variable(torch.cuda.LongTensor([1]))).unsqueeze(1)
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        anc_decoder_input = self.lexicon(Variable(torch.cuda.LongTensor([self.start]))).unsqueeze(1)
        anc_decoder_output, anc_decoder_hidden = self.anc_decoder(anc_decoder_input, anc_decoder_hidden)
        #rule_decoder_output, rule_decoder_hidden = self.rule_decoder(rule_decoder_input, rule_decoder_hidden)

        word_beam = torch.zeros(beam_size, max_len).long().cuda()
        rule_beam = torch.zeros(beam_size, max_len).long().cuda()
        nt_beam = torch.zeros(beam_size, max_len).long().cuda()
        word_count = torch.zeros(beam_size, top_k).long().cuda()
        rule_count = torch.zeros(beam_size, top_k).long().cuda()
        states_num = 1
        self.lex_cand_num = beam_size
        self.rule_cand_num = 3
        if self.lex_level == 0:
            decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0], context), dim=2)
            word_logits = self.lex_dist(F.tanh(self.lex_hidden_out(decoder_output[:, :, self.hidden_size:])))
            word_logprobs, word_argtop = torch.topk(F.log_softmax(word_logits.squeeze(1), dim=1), beam_size, dim=1)
            rule_logits = self.rule_dist(F.tanh(self.rule_hidden_out(decoder_output[:, :, :self.hidden_size*2])))
            rule_logprobs, rule_argtop = torch.topk(F.log_softmax(rule_logits.squeeze(1), dim=1), beam_size, dim=1)
            rule_argtop = rule_argtop.squeeze(0)
            rule_beam[:, 0] = rule_argtop.data[rule_argtop.data % beam_size]
            rule_beam_probs = rule_logprobs.squeeze(0).data[rule_argtop.data % beam_size]
            word_beam_probs = torch.zeros_like(rule_beam_probs)
            rule_count.fill_(1)
            word_count.fill_(0)
        else:
            #decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0], anc_decoder_hidden[0], context), dim=2)
            if self.data in ['persona', 'movie']:
                decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0], anc_decoder_hidden[0], context), dim=2)
            else:
                decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0], anc_decoder_hidden[0], syn_decoder_hidden[0], context, syn_context), dim=2)
            
            total_logprobs = torch.zeros(self.lex_cand_num, self.rule_cand_num).cuda()

            if self.data in ['persona', 'movie']:
                #word_logits = self.lex_dist(self.lex_hidden_out(decoder_output))
                #word_logits = self.lex_dist(F.tanh(self.lex_hidden_out(torch.cat((F.tanh(self.merge_out(decoder_output[:, :, self.hidden_size:self.hidden_size*3])), decoder_output[:, :, -self.context_size*1:]), dim=2))))
                word_logits = self.lex_dist(F.tanh(self.lex_hidden_out(torch.cat((decoder_output[:, :, self.hidden_size:self.hidden_size*3], decoder_output[:, :, -self.context_size*1:]), dim=2))))
            else:
                word_logits = self.lex_dist(F.tanh(self.lex_hidden_out(torch.cat((decoder_output[:, :, self.hidden_size:self.hidden_size*3], decoder_output[:, :, -self.context_size*2:-self.context_size]), dim=2))))
            word_logits = F.log_softmax(word_logits.squeeze(1), dim=1)
            word_logprobs, word_argtop = torch.topk(word_logits, self.lex_cand_num, dim=1)
            total_logprobs += word_logprobs.data.expand(self.rule_cand_num, self.lex_cand_num).transpose(0, 1)

            word_embed = self.lexicon(word_argtop).squeeze(0)
            word_embed = word_embed.expand(self.lex_cand_num, self.lex_input_size)
            decoder_output = decoder_output.squeeze(0).expand(self.lex_cand_num, self.decoder_output_size)
            if self.data in ['persona', 'movie']:
                rule_logits = self.rule_dist(F.tanh(self.rule_hidden_out(decoder_output[:, :self.hidden_size * 1])))
                #rule_logits = self.rule_dist(F.tanh(self.rule_hidden_out(torch.cat((decoder_output[:, :self.hidden_size*1], F.tanh(self.merge_out(decoder_output[:, self.hidden_size:self.hidden_size*3]))), dim=1))))
            else:
                #rule_logits = self.rule_dist(self.rule_hidden_out(torch.cat((decoder_output[:, self.hidden_size*3:self.hidden_size*4], decoder_output[:, :self.hidden_size*2], decoder_output[:, -self.context_size:]), dim=1)))
                rule_logits = self.rule_dist(F.tanh(self.rule_hidden_out(torch.cat((decoder_output[:, :self.hidden_size*2], decoder_output[:, -self.context_size:]), dim=1))))
            rule_logits = F.log_softmax(rule_logits, dim=1)
            rule_logits[:, 0] = -np.inf
            rule_logits[:, 1] = -np.inf
            rule_logits[:, 2] = -np.inf
            if self.lex_level > 0:
                root_rules = self.root_rules.expand(self.lex_cand_num, self.rule_vocab_size)
                rule_logits[~root_rules] = -np.inf
                rule_select_by_word = self.root_nt_word[:, word_argtop.squeeze(0).data].transpose(0, 1)
                rule_logits[~rule_select_by_word] = -np.inf
            rule_logprobs, rule_argtop = torch.topk(rule_logits, self.rule_cand_num, dim=1)
            total_logprobs += self.syn_weight * rule_logprobs.data

            logprobs, argtop = torch.topk(total_logprobs.view(-1), beam_size, dim=0)
            argtop_ind = torch.cuda.LongTensor(beam_size, 2)
            argtop_ind[:, 0] = argtop / self.rule_cand_num
            argtop_ind[:, 1] = argtop % self.rule_cand_num
            word_beam[:, 0] = word_argtop.squeeze(0).data[argtop_ind[:, 0]]
            word_beam_probs = word_logprobs.squeeze(0).data[argtop_ind[:, 0]]
            leaf_beam_probs = torch.zeros_like(word_beam_probs).fill_(0)
            rule_beam[:, 0] = rule_argtop.data[argtop_ind[:, 0], argtop_ind[:, 1]]
            rule_beam_probs = rule_logprobs.data[argtop_ind[:, 0], argtop_ind[:, 1]]
            first_word_probs = word_beam_probs.clone()

            word_count.fill_(1)
            rule_count.fill_(1)
        self.lex_cand_num = 15
        #hidden = (decoder_hidden[0].clone(), decoder_hidden[1].clone())
        def expand_by_beam(decoder_hidden):
            return (decoder_hidden[0].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1),
                    decoder_hidden[1].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1))

        def expand_by_state_and_beam(decoder_hidden):
            return (decoder_hidden[0].expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1),
                    decoder_hidden[1].expand(1, states_num, beam_size, self.hidden_size).contiguous().view(1, states_num * beam_size, -1))

        decoder_hidden = (decoder_hidden[0].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1),
                          decoder_hidden[1].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1))
        src_last_hidden = (src_last_hidden[0].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1),
                           src_last_hidden[1].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1))
        leaf_decoder_hidden = (leaf_decoder_hidden[0].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1),
                               leaf_decoder_hidden[1].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1))
        if self.lex_level > 0:
            anc_decoder_hidden = expand_by_beam(anc_decoder_hidden) 
            anc_decoder_input = self.lexicon(word_beam[:, 0]).unsqueeze(1)
            _, anc_decoder_hidden = self.anc_decoder(anc_decoder_input, anc_decoder_hidden)
            syn_decoder_hidden = expand_by_beam(syn_decoder_hidden) 
            syn_decoder_input = torch.cat((self.constituent(nt_beam[:, 0]), self.rule(rule_beam[:, 0])), dim=1).unsqueeze(1)
            _, syn_decoder_hidden = self.anc_syn_decoder(syn_decoder_input, syn_decoder_hidden)
            if self.rerank:
                lm_decoder_hidden = (lm_decoder_hidden[0].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1),
                                     lm_decoder_hidden[1].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1))
            lex_decoder_hidden = (lex_decoder_hidden[0].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1),
                                  lex_decoder_hidden[1].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1))
        src_hidden = src_hidden.expand(beam_size, src_hidden.size(1), self.hidden_size)
        psn_hidden = psn_hidden.expand(beam_size, psn_hidden.size(1), self.hidden_size)

        leaves = torch.cuda.LongTensor([[[0, 0], [0, 0], [0, 0]] for x in range(beam_size)])
        num_leaves = np.zeros(beam_size)
        lexicons = torch.cuda.LongTensor([[[0, 0], [0, 0], [0, 0]] for x in range(beam_size)])
        tree_dict = {}
        node_ind = 0

        #final = np.array([None for x in range(beam_size)])
        final = None
        final_score = -np.inf
        finished = 0
        current = np.array([None for x in range(beam_size)])
        for y in range(beam_size):
            rule = self.rules_by_id[rule_beam[y, 0]]
            if self.lex_level == 0:
                current[y] = Node('{}__{}__{}'.format('ROOT', word_beam[y, 0], rule))
                current[y].id = node_ind
                node_ind += 1
            else:
                try:
                    tag = rule.split()[-1]
                except:
                    pdb.set_trace()
                root = 'ROOT' if self.nt_vocab_size > 3 else 'NT'
                current[y] = Node('{}__{}__{} [{}]'.format(root, word_beam[y, 0], rule, len(rule.split())-1))
                current[y].id = node_ind
                node_ind += 1
            if self.lex_level > 0:
                tree_dict[current[y].id] = (
                                            (anc_decoder_hidden[0][:, y, :].clone(), anc_decoder_hidden[1][:, y, :].clone()),
                                            (syn_decoder_hidden[0][:, y, :].clone(), syn_decoder_hidden[1][:, y, :].clone())
                                           )
        
        def inheritance(rule):
            return literal_eval(rule[rule.find('['):rule.find(']')+1])

        def is_preterminal(x, t):
            '''
            if self.lex_level == 3:
                return self.preterminal[curr_nt[x]]
            elif self.lex_level in [0, 2]:
                return rule_beam[x, t] in self.ending_rules
            else:
                raise NotImplementedError
            '''
            return rule_beam[x, t] in self.ending_rules

        def rindex(lst, item):
            return len(lst) - lst[::-1].index(item) - 1

        for t in range(max_len-1):
            ans_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_par_rule = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            anc_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            depth = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            word_mask = torch.cuda.ByteTensor(beam_size, top_k)
            word_mask.fill_(True)
            rule_mask = torch.cuda.ByteTensor(beam_size, self.rule_cand_num)
            rule_mask.fill_(True)
            leaf_mask = torch.cuda.ByteTensor(beam_size, self.lex_cand_num)
            leaf_mask.fill_(False)
            curr_nt = np.array(['' for x in range(beam_size)], dtype=object)

            for x in range(beam_size):
                if current[x] is None:
                    rule_mask[x] = False
                    word_mask[x] = False
                else:
                    par_nt, par_lex, rule = current[x].name.split('__')
                    curr_nt[x] = rule.split()[len(current[x].children)]
                    curr_nt[x] = curr_nt[x].replace('_ROOT', '')
                    if self.lex_level == 0:
                        rule_count[x] += 1
                        '''
                        if not is_preterminal(x, t):
                            rule_count[x] += 1
                            word_mask[x] = False
                        else:
                            word_count[x] += 1
                            rule_mask[x] = False
                        '''
                    else:
                        #if self.lex_level == 3:
                        if False:
                            if not is_preterminal(x, t):
                                rule_count[x] += 1
                            else:
                                #rule_count[x] += 1
                                rule_mask[x] = False
                                num_leaves[x] += 1
                                leaf_mask[x] = True
                        elif self.lex_level > 0:
                            rule_count[x] += 1
                            if is_preterminal(x, t):
                                num_leaves[x] += 1
                                leaf_mask[x] = True
                        if len(current[x].children) not in inheritance(rule):
                            word_count[x] += 1
                        else:
                            word_mask[x] = False
                    ans_nt[x] = self.nt_dictionary[curr_nt[x]]
                    anc_lex[x] = int(par_lex)
                    if '[' in rule:
                        ans_par_rule[x] = self.rule_dictionary['RULE: {}'.format(rule[:rule.find('[')-1])]
                    else:
                        ans_par_rule[x] = self.rule_dictionary['RULE: {}'.format(rule)]
                    depth[x] = min(current[x].depth + 2, 9)
            
            ans_nt = self.constituent(ans_nt)
            ans_par_rule = self.rule(ans_par_rule)
            depth = self.depth(depth)
            '''
            anc_lex1 = self.lexicon(anc_lex1)
            anc_lex2 = self.lexicon(anc_lex2)
            '''
            #tree_input = torch.cat((anc_lex1, anc_lex2), dim=1).unsqueeze(1)
            ans_embed = torch.cat((ans_nt, ans_par_rule, depth), dim=1)
            decoder_input = ans_embed.unsqueeze(1)
               
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            if self.lex_level == 0:
                context = self.attention((leaf_decoder_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'leaf_') 
            elif self.lex_level > 0:
                decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0].transpose(0, 1), anc_decoder_hidden[0].transpose(0, 1)), dim=2)
                #decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0].transpose(0, 1), lex_decoder_hidden[0].transpose(0, 1)), dim=2)
                if self.data in ['persona', 'movie']:
                    '''
                    leaf_context = self.attention((leaf_decoder_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths, 'leaf_') 
                    anc_context = self.attention((anc_decoder_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths, 'anc_') 
                    #lex_context = self.attention((lex_decoder_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, psn_hidden, psn_hidden.size(1), psn_lengths, 'lex_') 
                    context = torch.cat((leaf_context, anc_context), dim=2)
                    '''
                    context = self.attention((decoder_output.transpose(0, 1)[:, :, self.hidden_size:self.hidden_size*3], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'leaf_') 
                    if self.rerank:
                        lm_context = self.attention((lm_decoder_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'lm_') 
                    decoder_output = torch.cat((decoder_output, context), dim=2)
                else:
                    context = self.attention2((decoder_output.transpose(0, 1)[:, :, self.hidden_size:self.hidden_size*3], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'lex') 
                    '''
                    syn_context = self.attention2((torch.cat((decoder_output.transpose(0, 1)[:, :, :self.hidden_size], 
                                                              decoder_output.transpose(0, 1)[:, :, self.hidden_size*3:self.hidden_size*4]), 
                                                             dim=2), None
                                                  ), psn_hidden, psn_hidden.size(1), psn_lengths, psn_last_hidden, 'syn') 
                    '''
                    syn_context = self.attention2((decoder_output.transpose(0, 1)[:, :, :self.hidden_size], None), psn_hidden, psn_hidden.size(1), psn_lengths.tolist(), psn_last_hidden, 'syn')
                    lm_context = self.attention2((lm_decoder_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'lex') 
                    decoder_output = torch.cat((decoder_output, context, syn_context), dim=2)
            #dup_mask = ~(word_mask | rule_mask)

            if self.lex_level == 0:
                decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0].transpose(0, 1), context), dim=2)
                '''
                word_logits = self.lex_dist(F.tanh(self.lex_hidden_out((decoder_output[:, :, self.hidden_size:]))))
                word_logprobs, word_argtop = torch.topk(F.log_softmax(word_logits.squeeze(1), dim=1), top_k, dim=1)
                word_beam_logprobs = ((word_beam_probs).expand(top_k, beam_size).transpose(0, 1) + word_logprobs.data * word_mask.float()) 

                rule_logits = self.rule_dist(F.tanh(self.rule_hidden_out(decoder_output[:, :, :self.hidden_size*2])))
                dup_mask[:, 0] = 0
                word_beam_logprobs[dup_mask] = -np.inf
                rule_logprobs, rule_argtop = torch.topk(F.log_softmax(rule_logits.squeeze(1), dim=1), top_k, dim=1)
                rule_beam_logprobs = ((rule_beam_probs).expand(top_k, beam_size).transpose(0, 1) + rule_logprobs.data * rule_mask.float())
                rule_beam_logprobs[dup_mask] = -np.inf
                if (word_count != 0).all():
                    total_logprobs = word_beam_logprobs / word_count.float() + rule_beam_logprobs / rule_count.float()
                else:
                    total_logprobs = rule_beam_logprobs / rule_count.float()
                #total_logprobs = (word_beam_logprobs + rule_beam_logprobs) / (t + 1)
                #total_logprobs = word_beam_logprobs / np.sqrt(t + 1) + rule_beam_logprobs / (t + 1)
                best_probs, best_args = total_logprobs.view(-1).topk(beam_size)
                '''
                rule_logits = self.rule_dist(F.tanh(self.rule_hidden_out(decoder_output[:, :, :self.hidden_size*2])))
                for x in range(beam_size):
                    if current[x] is not None and current[x].depth + 2 >= 11:
                        rule_logits_x = rule_logits[x].data
                        rule_logits_x[:, self.non_ending_rules] = -np.inf
                        rule_logits[x].data = rule_logits_x
                        #assert (rule_logits[x].data != -np.inf).long().sum() == self.lex_cand_num
                        #dup_mask[x, :, 1:self.rule_cand_num] = True
                rule_logits[:, :, 0] = -np.inf
                rule_logits[:, :, 1] = -np.inf
                rule_logits[:, :, 2] = -np.inf

                rule_logprobs, rule_argtop = torch.topk(F.log_softmax(rule_logits.squeeze(1), dim=1), self.rule_cand_num, dim=1)
                rule_beam_logprobs = rule_logprobs.data.expand(self.lex_cand_num, beam_size, self.rule_cand_num).permute(1, 0, 2)

                word_logits = self.lex_dist(F.tanh(self.lex_hidden_out((decoder_output[:, :, self.hidden_size:]))))
                curr_nt_ind = [self.nt_dictionary[x] for x in curr_nt]
                word_logits[~self.nt_word[curr_nt_ind]] = -10e8
                word_logprobs, word_argtop = torch.topk(F.log_softmax(word_logits.squeeze(1), dim=1), self.lex_cand_num, dim=1)
                #word_beam_logprobs = ((word_beam_probs).expand(self.lex_cand_num, beam_size).transpose(0, 1) + word_logprobs.data * word_mask.float())
                word_beam_logprobs = word_logprobs.data.expand(self.rule_cand_num, beam_size, self.lex_cand_num).permute(1, 2, 0)

                _word_mask = torch.from_numpy(np.isin(rule_argtop.data.cpu().numpy(), self.ending_rules).astype(np.float32)).cuda()
                _word_mask = _word_mask.expand(self.lex_cand_num, beam_size, self.rule_cand_num).permute(1, 0, 2)
                _word_count = word_count[:, 0].expand(self.lex_cand_num, self.rule_cand_num, beam_size).permute(2, 0, 1).float()
                _word_count += _word_mask
                _rule_count = rule_count[:, 0].expand(self.lex_cand_num, self.rule_cand_num, beam_size).permute(2, 0, 1).float()
                word_beam_logprobs = word_beam_logprobs * _word_mask + word_beam_probs.expand(self.lex_cand_num, self.rule_cand_num, beam_size).permute(2, 0, 1)
                rule_beam_logprobs = rule_beam_logprobs + rule_beam_probs.expand(self.lex_cand_num, self.rule_cand_num, beam_size).permute(2, 0, 1)
                if (word_count[:, 0] > 0).all():
                    total_logprobs = word_beam_logprobs / _word_count + rule_beam_logprobs / _rule_count
                else:
                    total_logprobs = rule_beam_logprobs / _rule_count
                    total_logprobs[:, 1:, :]  = -np.inf
                logprobs, argtop = torch.topk(total_logprobs.view(-1).contiguous(), beam_size, dim=0)

                argtop_ind = torch.cuda.LongTensor(beam_size, 3)
                argtop_ind[:, 0] = argtop / (self.lex_cand_num * self.rule_cand_num)
                argtop_ind[:, 1] = (argtop % (self.lex_cand_num * self.rule_cand_num)) / self.rule_cand_num
                argtop_ind[:, 2] = (argtop % (self.lex_cand_num * self.rule_cand_num)) % self.rule_cand_num
            else:
                total_logprobs = torch.zeros(beam_size, self.lex_cand_num, self.rule_cand_num).cuda()
                decoder_output = decoder_output.squeeze(1)
                word_mask = word_mask[:, 0].expand(self.lex_cand_num, self.rule_cand_num, beam_size).permute(2, 0, 1).float()
                _word_count = word_count[:, 0].expand(self.lex_cand_num, self.rule_cand_num, beam_size).permute(2, 0, 1).float()
                rule_mask = rule_mask[:, 0].expand(self.lex_cand_num, self.rule_cand_num, beam_size).permute(2, 0, 1).float()
                _rule_count = rule_count[:, 0].expand(self.lex_cand_num, self.rule_cand_num, beam_size).permute(2, 0, 1).float()

                if self.data in ['persona', 'movie']:
                    #word_logits = self.lex_dist(self.lex_hidden_out(decoder_output))
                    #word_logits = self.lex_dist(F.tanh(self.lex_hidden_out(torch.cat((F.tanh(self.merge_out(decoder_output[:, self.hidden_size:self.hidden_size*3])), decoder_output[:, -self.context_size*1:]), dim=1))))
                    word_logits = self.lex_dist(F.tanh(self.lex_hidden_out(torch.cat((decoder_output[:, self.hidden_size:self.hidden_size*3], decoder_output[:, -self.context_size*1:]), dim=1))))
                else:
                    word_logits = self.lex_dist(F.tanh(self.lex_hidden_out(torch.cat((decoder_output[:, self.hidden_size:self.hidden_size*3], decoder_output[:, -self.context_size*2:-self.context_size]), dim=1))))
                word_logits = F.log_softmax(word_logits, dim=1)
                curr_nt_ind = [self.nt_dictionary[x] for x in curr_nt]
                word_logits[~self.nt_word[curr_nt_ind]] = -10e8
                word_logits[~self.word_word[anc_lex.data]] = -10e8
                word_logits[:, 0] = -np.inf
                word_logits[:, 1] = -np.inf
                word_logits[:, 2] = -np.inf
                word_logprobs, word_argtop = torch.topk(word_logits, self.lex_cand_num, dim=1)
                word_beam_logprobs = word_logprobs.data.expand(self.rule_cand_num, beam_size, self.lex_cand_num).permute(1, 2, 0)

                dup_mask = torch.ByteTensor(beam_size, self.lex_cand_num, self.rule_cand_num).fill_(False)
                for x in range(beam_size):
                    if rule_mask[x, 0, 0] == 0:
                        if word_mask[x, 0, 0] == 0:
                            dup_mask[x, :, :] = True
                            dup_mask[x, 0, 0] = False
                            if current[x] is not None:
                                inherit = current[x].name.split('__')[1]
                                word_argtop[x] = int(inherit)
                        else:
                            dup_mask[x, :, :] = True
                            dup_mask[x, :,  0] = False
                    elif word_mask[x, 0, 0] == 0:
                        dup_mask[x, :, :] = True
                        dup_mask[x, 0, :] = False
                        inherit = current[x].name.split('__')[1]
                        word_argtop[x] = int(inherit)
                dup_mask = dup_mask.cuda()

                if self.rerank:
                    lm_word_logits = self.lex_dist(F.tanh(self.lex_hidden_out_2(torch.cat((lm_decoder_hidden[0].squeeze(0), lm_context.squeeze(1)), dim=1))))
                    lm_word_logits = F.log_softmax(lm_word_logits, dim=1)
                    leaf_beam_logprobs = torch.zeros_like(word_argtop).float().data
                    for x in range(beam_size):
                        leaf_beam_logprobs[x] = lm_word_logits[x][word_argtop[x]].data
                    leaf_beam_logprobs = (leaf_beam_logprobs * leaf_mask.float()).expand(self.rule_cand_num, beam_size, self.lex_cand_num).permute(1, 2, 0) + leaf_beam_probs.expand(self.lex_cand_num, self.rule_cand_num, beam_size).permute(2, 0, 1)

                decoder_output = decoder_output.unsqueeze(1).expand(beam_size, self.lex_cand_num, self.decoder_output_size)
                word_embed = self.lexicon(word_argtop).squeeze(0)
                word_embed = word_embed.expand(beam_size, self.lex_cand_num, self.lex_input_size)
                if self.data in ['persona', 'movie']:
                    rule_logits = self.rule_dist(F.tanh(self.rule_hidden_out(decoder_output[:, :, :self.hidden_size * 1])))
                    #rule_logits = self.rule_dist(F.tanh(self.rule_hidden_out(torch.cat((decoder_output[:, :, :self.hidden_size*1], F.tanh(self.merge_out(decoder_output[:, :, self.hidden_size:self.hidden_size*3]))), dim=2))))
                else:
                    #rule_logits = self.rule_dist(self.rule_hidden_out(torch.cat((decoder_output[:, :, self.hidden_size*3:self.hidden_size*4], decoder_output[:, :, :self.hidden_size*2], decoder_output[:, :, -self.context_size:]), dim=2)))
                    rule_logits = self.rule_dist(F.tanh(self.rule_hidden_out(torch.cat((decoder_output[:, :, :self.hidden_size*2], decoder_output[:, :, -self.context_size:]), dim=2))))
                rule_logits = F.log_softmax(rule_logits, dim=2)
                #if self.lex_level == 3: 
                if False:
                    ending_rules = rule_logits[:, :, self.ending_rules].data
                    rule_select_by_word = self.rule_by_word[word_argtop.view(-1).data].view(beam_size, self.lex_cand_num, -1)
                    #curr_nt_ind = [self.nt_dictionary[nt] for nt in curr_nt]
                    rule_select_by_nt = self.nt_rule[curr_nt_ind].expand(self.lex_cand_num, beam_size, self.rule_vocab_size).transpose(0, 1)
                    rule_logits[~rule_select_by_word] = -10e8
                    rule_logits[~rule_select_by_nt] = -10e8
                elif self.lex_level > 0:
                    rule_select_by_nt = self.nt_rule[curr_nt_ind].expand(self.lex_cand_num, beam_size, self.rule_vocab_size).transpose(0, 1)
                    rule_logits[~rule_select_by_nt] = -np.inf
                else:
                    ending_rules = rule_logits[:, :, [self.rule_dictionary['RULE: PT PT']]].data
                rule_logits[:, :, 0] = -np.inf
                rule_logits[:, :, 1] = -np.inf
                rule_logits[:, :, 2] = -np.inf
                for x in range(beam_size):
                    if current[x] is not None and current[x].depth + 2 >= 13 and not is_preterminal(x, t):
                        rule_logits_x = rule_logits[x].data
                        rule_logits_x[:, self.non_ending_rules] = -np.inf
                        #if self.lex_level == 3:
                        if False:
                            for y in range(self.lex_cand_num):
                                if (rule_logits_x[y] == -np.inf).all():
                                    #rule_logits_x[[y], self.ending_rules] = ending_rules[x, y]
                                    pdb.set_trace()
                        rule_logits[x].data = rule_logits_x
                        if self.lex_level > 0:
                            #assert (rule_logits[x].data != -np.inf).long().sum() == self.lex_cand_num
                            dup_mask[x, :, 1:self.rule_cand_num] = True
                    '''
                    if self.preterminal[curr_nt[x]]:
                        rule_argtop[x] = 2
                        rule_logprobs[x].data = rule_eod_logprobs[x].data
                    '''
                rule_logprobs, rule_argtop = torch.topk(rule_logits, self.rule_cand_num, dim=2)
                rule_beam_logprobs = rule_logprobs.data

                word_beam_logprobs = word_beam_logprobs * word_mask + word_beam_probs.expand(self.lex_cand_num, self.rule_cand_num, beam_size).permute(2, 0, 1)
                rule_beam_logprobs = rule_beam_logprobs * rule_mask + rule_beam_probs.expand(self.lex_cand_num, self.rule_cand_num, beam_size).permute(2, 0, 1)

                leaf_count = torch.cuda.FloatTensor(num_leaves.tolist()).expand(self.lex_cand_num, self.rule_cand_num, beam_size).permute(2, 0, 1)
                if (num_leaves > 1).all() and self.lm_weight > 0:
                    total_logprobs = self.lex_weight * word_beam_logprobs / _word_count + self.syn_weight * rule_beam_logprobs / _rule_count + self.lm_weight * leaf_beam_logprobs / leaf_count
                else:
                    #total_logprobs = self.lex_weight * word_beam_logprobs / _word_count + self.syn_weight * rule_beam_logprobs / _rule_count + first_word_probs.expand(self.lex_cand_num, self.rule_cand_num, beam_size).permute(2, 0, 1)
                    total_logprobs = self.lex_weight * word_beam_logprobs / _word_count + self.syn_weight * rule_beam_logprobs / _rule_count
                total_logprobs[dup_mask] = -np.inf
                total_logprobs[total_logprobs != total_logprobs] = -np.inf

                logprobs, argtop = torch.topk(total_logprobs.view(-1).contiguous(), beam_size, dim=0)
                argtop_ind = torch.cuda.LongTensor(beam_size, 3)
                argtop_ind[:, 0] = argtop / (self.lex_cand_num * self.rule_cand_num)
                argtop_ind[:, 1] = (argtop % (self.lex_cand_num * self.rule_cand_num)) / self.rule_cand_num
                argtop_ind[:, 2] = (argtop % (self.lex_cand_num * self.rule_cand_num)) % self.rule_cand_num

            def reshape_hidden(decoder_hidden, dim2):
                return (decoder_hidden[0].view(1, dim2, beam_size, -1),
                        decoder_hidden[1].view(1, dim2, beam_size, -1))

            decoder_hidden = reshape_hidden(decoder_hidden, batch_size)
            leaf_decoder_hidden = reshape_hidden(leaf_decoder_hidden, batch_size)
            if self.lex_level > 0:
                anc_decoder_hidden = reshape_hidden(anc_decoder_hidden, batch_size)
                syn_decoder_hidden = reshape_hidden(syn_decoder_hidden, batch_size)
                #lex_decoder_hidden = reshape_hidden(lex_decoder_hidden, batch_size)

            if self.lex_level == 0:
                last = argtop_ind[:, 0]
                #last = (best_args / top_k)
                #curr = (best_args % top_k)
                current = current[last.tolist()]
                leaves = leaves[last.tolist()]
                lexicons = lexicons[last.tolist()]
                curr_nt = curr_nt[last.tolist()]
                #final = final[last.tolist()]
                word_beam = word_beam[last]
                rule_beam = rule_beam[last]
                word_count = word_count[last]
                rule_count = rule_count[last]
                '''
                word_beam_probs = word_beam_logprobs[last, curr]
                rule_beam_probs = rule_beam_logprobs[last, curr]
                word_beam[:, t+1] = word_argtop[last, curr].data
                rule_beam[:, t+1] = rule_argtop[last, curr].data
                '''
                word_beam[:, t+1] = word_argtop.data[last, argtop_ind[:, 1]]
                rule_beam[:, t+1] = rule_argtop.data[last, argtop_ind[:, 2]]
                word_beam_probs = word_beam_logprobs[last, argtop_ind[:, 1], argtop_ind[:, 2]]
                rule_beam_probs = rule_beam_logprobs[last, argtop_ind[:, 1], argtop_ind[:, 2]]
            else:
                last = argtop_ind[:, 0]
                current = current[last.tolist()]
                leaves = leaves[last.tolist()]
                lexicons = lexicons[last.tolist()]
                curr_nt = curr_nt[last.tolist()]
                #final = final[last.tolist()]
                num_leaves = num_leaves[last.tolist()]
                first_word_probs = first_word_probs[last]
                word_beam = word_beam[last]
                rule_beam = rule_beam[last]
                word_count = word_count[last]
                rule_count = rule_count[last]
                word_beam[:, t+1] = word_argtop.data[last, argtop_ind[:, 1]]
                rule_beam[:, t+1] = rule_argtop.data[last, argtop_ind[:, 1], argtop_ind[:, 2]]
                word_beam_probs = word_beam_logprobs[last, argtop_ind[:, 1], argtop_ind[:, 2]]
                rule_beam_probs = rule_beam_logprobs[last, argtop_ind[:, 1], argtop_ind[:, 2]]
                if self.rerank:
                    leaf_beam_probs = leaf_beam_logprobs[last, argtop_ind[:, 1], argtop_ind[:, 2]]

            def merge_hidden(decoder_hidden, dim2):
                return (decoder_hidden[0].view(1, dim2 * beam_size, -1),
                        decoder_hidden[1].view(1, dim2 * beam_size, -1))

            decoder_hidden[0][:, 0, :, :] = decoder_hidden[0][:, 0, :, :][:, last, :]
            decoder_hidden[1][:, 0, :, :] = decoder_hidden[1][:, 0, :, :][:, last, :]
            decoder_hidden = (decoder_hidden[0].view(1, batch_size * beam_size, -1), decoder_hidden[1].view(1, batch_size * beam_size, -1))

            leaf_decoder_hidden[0][:, 0, :, :] = leaf_decoder_hidden[0][:, 0, :, :][:, last, :]
            leaf_decoder_hidden[1][:, 0, :, :] = leaf_decoder_hidden[1][:, 0, :, :][:, last, :]
            leaf_decoder_hidden = (leaf_decoder_hidden[0].view(1, batch_size * beam_size, -1),
                                   leaf_decoder_hidden[1].view(1, batch_size * beam_size, -1))


            if self.lex_level > 0:
                if self.rerank:
                    lm_decoder_hidden = (lm_decoder_hidden[0][:, last, :], lm_decoder_hidden[1][:, last, :])
                lex_decoder_hidden = (lex_decoder_hidden[0][:, last, :], lex_decoder_hidden[1][:, last, :])
                anc_decoder_hidden[0][:, 0, :, :] = anc_decoder_hidden[0][:, 0, :, :][:, last, :]
                anc_decoder_hidden[1][:, 0, :, :] = anc_decoder_hidden[1][:, 0, :, :][:, last, :]
                syn_decoder_hidden[0][:, 0, :, :] = syn_decoder_hidden[0][:, 0, :, :][:, last, :]
                syn_decoder_hidden[1][:, 0, :, :] = syn_decoder_hidden[1][:, 0, :, :][:, last, :]
                
                anc_decoder_hidden = merge_hidden(anc_decoder_hidden, batch_size)
                syn_decoder_hidden = merge_hidden(syn_decoder_hidden, batch_size)

            if self.lex_level == 0:
                for x in range(beam_size):
                    current[x] = deepcopy(current[x])
                    if current[x] is None:
                        continue
                    if is_preterminal(x, t+1):
                        word = word_beam[x, t+1]
                        ch = Node('{}__{}__ '.format(curr_nt[x], word), parent=current[x])

                        lex_decoder_input = self.lexicon(Variable(torch.cuda.LongTensor([int(word)]))).unsqueeze(1)
                        _, _leaf_decoder_hidden = self.leaf_decoder(lex_decoder_input, (leaf_decoder_hidden[0][:, x, :].unsqueeze(1), leaf_decoder_hidden[1][:, x, :].unsqueeze(1)))
                        leaf_decoder_hidden[0][:, x, :] = _leaf_decoder_hidden[0][0]
                        leaf_decoder_hidden[1][:, x, :] = _leaf_decoder_hidden[1][0]

                        word_count[x, :] += 1

                        while True:
                            if current[x] is None:
                                break
                            if current[x].parent is None:
                                if logprobs[x] > final_score:
                                    final = deepcopy(current[x])
                                    final_score = logprobs[x]
                                    finished += 1
                                word_beam_probs[x] = -10e08
                                rule_beam_probs[x] = -10e08

                            _, _, rule = current[x].name.split('__')
                            num_children = len(rule.split())
                            if num_children > len(current[x].children):
                                break
                            else:
                                current[x] = current[x].parent
                    else:
                        rule = self.rules_by_id[rule_beam[x, t+1]]
                        word = 0
                        current[x] = Node('{}__{}__{}'.format(curr_nt[x], word, rule), parent=current[x])
            else:
                anc_select = []
                anc_lex_select = []
                leaf_select = []
                anc_word = []
                anc_nt = []
                anc_rule = []
                leaf_word = []
                next_nt = []
                lex_select = []
                lex_word = []
                if self.rerank:
                    lm_context = self.attention((lm_decoder_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths, src_last_hidden, 'lm_') 
                    lm_word_logits = self.lex_dist(F.tanh(self.lex_hidden_out_2(torch.cat((lm_decoder_hidden[0].squeeze(0), lm_context.squeeze(1)), dim=1))))
                    lm_word_logits = F.log_softmax(lm_word_logits, dim=1)
                    _leaf_beam_probs = leaf_beam_probs + lm_word_logits[:, self.eou].data
                for x in range(beam_size):
                    if logprobs[x] == -np.inf:
                        current[x] = None
                        continue
                    current[x] = deepcopy(current[x])
                    #tree_dict[x] = deepcopy(tree_dict[x])
                    if current[x] is None:
                        continue
                    #nt, word, rule = current[x].name.split('__')
                    if is_preterminal(x, t+1):
                        _, word, rule = current[x].name.split('__')
                        if len(current[x].children) not in inheritance(rule):
                            word = word_beam[x, t+1]
                            lex_select.append(x)
                            lex_word.append(word)
                        ch = Node('{}__{}__ []'.format(curr_nt[x], word), parent=current[x])

                        while True:
                            if self.nt_vocab_size > 3:
                                if current[x] is None:
                                    break
                                if current[x].parent is None:
                                    if self.rerank:
                                        if logprobs[x] + _leaf_beam_probs[x] / (num_leaves[x] + 1) > final_score:
                                            final = deepcopy(current[x])
                                            final_score = logprobs[x] + _leaf_beam_probs[x] / (num_leaves[x] + 1)
                                            finished += 1
                                    else:
                                        if logprobs[x] > final_score:
                                        #if word_beam_probs[x] / num_leaves[x] > final_score:
                                            final = deepcopy(current[x])
                                            final_score = logprobs[x]
                                            #final_score = word_beam_probs[x] / num_leaves[x]
                                            finished += 1
                                    word_beam_probs[x] = -10e08
                                    rule_beam_probs[x] = -10e08
                                    leaf_beam_probs[x] = -10e08
                                _, _, rule = current[x].name.split('__')
                                num_children = len(rule[:rule.find('[')].split())
                                if num_children > len(current[x].children):
                                    break
                                else:
                                    current[x] = current[x].parent
                            else:
                                _, _, rule = current[x].name.split('__')
                                num_children = len(rule[:rule.find('[')].split())
                                if num_children > len(current[x].children):
                                    break
                                elif current[x].parent is None:
                                    if logprobs[x] > final_score:
                                        final = deepcopy(current[x])
                                        final_score = logprobs[x]
                                        finished += 1
                                    word_beam_probs[x] = -10e08
                                    rule_beam_probs[x] = -10e08
                                    leaf_beam_probs[x] = -10e08
                                    current[x] = None
                                    break
                                else:
                                    current[x] = current[x].parent

                        leaf_select.append(x)
                        leaf_word.append(int(word))
                        if current[x] is not None:
                            new_hidden = tree_dict[current[x].id]
                            anc_decoder_hidden[0][:, x, :].data = new_hidden[0][0].data.clone()
                            anc_decoder_hidden[1][:, x, :].data = new_hidden[0][1].data.clone()
                            syn_decoder_hidden[0][:, x, :].data = new_hidden[1][0].data.clone()
                            syn_decoder_hidden[1][:, x, :].data = new_hidden[1][1].data.clone()
                    else:
                        nt, word, par_rule = current[x].name.split('__')
                        rule = self.rules_by_id[rule_beam[x, t+1]]
                        if len(current[x].children) not in inheritance(par_rule):
                            word = word_beam[x, t+1]

                            anc_lex_select.append(x)
                            anc_word.append(int(word))
                            lex_select.append(x)
                            lex_word.append(word)

                        anc_select.append(x)
                        anc_nt.append(self.nt_dictionary[nt])
                        anc_rule.append(self.rule_dictionary['RULE: {}'.format(par_rule[:par_rule.find('[')-1])])

                        tag = rule.split()[-1]
                        #if self.lex_level == 3:
                        if False:
                            current[x] = Node('{}__{}__{} [{}]'.format(curr_nt[x], word, rule, len(rule.split())-1), parent=current[x])
                        elif self.lex_level > 0:
                            rule = rule.replace('_ROOT', '')
                            assert rule.find('[') == -1
                            inh = len(rule.split()) - rule.split()[::-1].index(curr_nt[x]) - 1
                            current[x] = Node('{}__{}__{} [{}]'.format(curr_nt[x], word, rule, rule.split().index(curr_nt[x])), parent=current[x])
                        current[x].id = node_ind
                        node_ind += 1
                        next_nt.append(self.nt_dictionary[rule.split()[len(current[x].children)]])
                if anc_lex_select:
                    anc_decoder_input = self.lexicon(Variable(torch.cuda.LongTensor(anc_word))).unsqueeze(1)
                    _, _anc_decoder_hidden = self.anc_decoder(anc_decoder_input, (anc_decoder_hidden[0][:, anc_lex_select, :], anc_decoder_hidden[1][:, anc_lex_select, :]))
                    anc_decoder_hidden[0][:, anc_lex_select, :] = _anc_decoder_hidden[0][0].clone()
                    anc_decoder_hidden[1][:, anc_lex_select, :] = _anc_decoder_hidden[1][0].clone()
                '''
                if anc_select:
                    #syn_decoder_input = torch.cat((self.constituent(Variable(torch.cuda.LongTensor(anc_nt))), self.rule(Variable(torch.cuda.LongTensor(anc_rule)))), dim=1).unsqueeze(1)
                    #curr_nt_id = [self.nt_dictionary[nt] for nt in curr_nt[anc_select]]
                    syn_decoder_input = torch.cat((self.constituent(Variable(torch.cuda.LongTensor(next_nt))), self.rule(Variable(torch.cuda.LongTensor(anc_rule)))), dim=1).unsqueeze(1)
                    _, _syn_decoder_hidden = self.anc_syn_decoder(syn_decoder_input, (syn_decoder_hidden[0][:, anc_select, :], syn_decoder_hidden[1][:, anc_select, :]))
                    syn_decoder_hidden[0][:, anc_select, :] = _syn_decoder_hidden[0][0].clone()
                    syn_decoder_hidden[1][:, anc_select, :] = _syn_decoder_hidden[1][0].clone()
                if lex_select:
                    lex_decoder_input = self.lexicon(Variable(torch.cuda.LongTensor(lex_word))).unsqueeze(1)
                    _, _lex_decoder_hidden = self.lex_decoder(lex_decoder_input, (lex_decoder_hidden[0][:, lex_select, :], lex_decoder_hidden[1][:, lex_select, :]))
                    lex_decoder_hidden[0][:, lex_select, :] = _lex_decoder_hidden[0][0]
                    lex_decoder_hidden[1][:, lex_select, :] = _lex_decoder_hidden[1][0]
                '''
                if leaf_select:
                    leaf_decoder_input = self.lexicon(Variable(torch.cuda.LongTensor(leaf_word))).unsqueeze(1)
                    _, _leaf_decoder_hidden = self.leaf_decoder(leaf_decoder_input, (leaf_decoder_hidden[0][:, leaf_select, :], leaf_decoder_hidden[1][:, leaf_select, :]))
                    leaf_decoder_hidden[0][:, leaf_select, :] = _leaf_decoder_hidden[0][0]
                    leaf_decoder_hidden[1][:, leaf_select, :] = _leaf_decoder_hidden[1][0]

                    if self.rerank:
                        _, _lm_decoder_hidden = self.lm_decoder(leaf_decoder_input, (lm_decoder_hidden[0][:, leaf_select, :], lm_decoder_hidden[1][:, leaf_select, :]))
                        lm_decoder_hidden[0][:, leaf_select, :] = _lm_decoder_hidden[0][0]
                        lm_decoder_hidden[1][:, leaf_select, :] = _lm_decoder_hidden[1][0]
                for x in range(beam_size):
                    if current[x] is not None:
                        tree_dict[current[x].id] = (
                                                    (anc_decoder_hidden[0][:, x, :].clone(), anc_decoder_hidden[1][:, x, :].clone()),
                                                    (syn_decoder_hidden[0][:, x, :].clone(), syn_decoder_hidden[1][:, x, :].clone())
                                                   )
            
            #if (current == None).all():
            if finished >= beam_size or (current == None).all():
                break

        '''
        best, best_arg = beam_probs.max(1)
        generations = beam[torch.arange(_batch_size).long().cuda(), best_arg.data].data.cpu()
        return generations, best
        '''
        print(final_score)
        if final is None:
            return current[0].ancestors[0]
        else:
            return final
        


if __name__ == '__main__':
    train()
