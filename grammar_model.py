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
from anytree import Node
from copy import deepcopy


Preterminal = literal_eval(open('preterminal.txt').readlines()[0])

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
            self.tree_input_size = 512
            self.tree = nn.Linear(lex_input_size * 4 + nt_input_size * 0 + rule_input_size * 0, self.tree_input_size)
            self.leaf_decoder = nn.LSTM(lex_input_size, hidden_size, batch_first=True, dropout=0.0)
            self.lex_decoder = nn.LSTM(lex_input_size, hidden_size, batch_first=True, dropout=0.0)
            self.anc_decoder = nn.LSTM(lex_input_size + nt_input_size + rule_input_size, hidden_size, batch_first=True, dropout=0.0)
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
                self.lex_hidden_out = nn.Linear(hidden_size * 3 + context_size, hidden_size)
                self.rule_hidden_out = nn.Linear(hidden_size * 3 + context_size + nt_input_size + lex_input_size, hidden_size)
                self.nt_hidden_out = nn.Linear(hidden_size * 3 + context_size + lex_input_size, hidden_size)

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
            self.a_key = nn.Linear(hidden_size, 100)
        else:
            self.a_key = nn.Linear(hidden_size * 2, 100)
        self.q_key = nn.Linear(hidden_size, 100)
        self.q_value = nn.Linear(hidden_size, context_size)

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

        self.rules_by_nt = {}
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

    def attention(self, decoder_hidden, src_hidden, src_max_len, src_lengths):
        a_key = F.tanh(self.a_key(decoder_hidden[0].squeeze(0)))
        q_key = F.tanh(self.q_key(src_hidden))
        q_value = F.tanh(self.q_value(src_hidden))
        q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
        q_mask  = torch.arange(src_max_len).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(src_max_len, 1).transpose(0, 1)
        q_energy[~q_mask] = -np.inf
        q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
        q_context = torch.bmm(q_weights, q_value).squeeze(1)
        context = q_context.unsqueeze(1)

        return context

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

    def forward(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, positions, ancestors):
        batch_size = src_seqs.size(0)

        src_embed = self.lexicon(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 
        #rule_decoder_hidden = self.init_hidden(src_last_hidden, 'rule_') 
        leaf_init_hidden = self.init_hidden(src_last_hidden, 'leaf_') 
        leaf_decoder_hidden, _leaf_indices = self.encode(trg_seqs, leaf_indices, 'leaf', leaf_init_hidden)
        leaf_decoder_hidden = torch.cat((leaf_init_hidden[0].transpose(0, 1), leaf_decoder_hidden), dim=1)
        if self.lex_level >= 2:
            _indices = [np.arange(len(lex_indices[x])) + 1 for x in range(len(lex_indices))]
            lex_init_hidden = self.init_hidden(src_last_hidden, 'lex_') 
            lex_decoder_hidden, _lex_indices = self.encode(trg_seqs, _indices, 'lex', lex_init_hidden)
            lex_decoder_hidden = torch.cat((lex_init_hidden[0].transpose(0, 1), lex_decoder_hidden), dim=1)

        ans_nt = self.constituent(Variable(trg_seqs[0]).cuda())
        '''
        ans_par_nt = self.constituent(Variable(parent_seqs[0][0]).cuda())
        ans_par2_nt = self.constituent(Variable(parent_seqs[1][0]).cuda())
        ans_par_lex = self.lexicon(Variable(parent_seqs[0][1]).cuda())
        ans_par2_lex = self.lexicon(Variable(parent_seqs[1][1]).cuda())
        ans_leaf_nt = self.constituent(Variable(leaf_seqs[0][0]).cuda())
        ans_leaf2_nt = self.constituent(Variable(leaf_seqs[1][0]).cuda())
        ans_leaf3_nt = self.constituent(Variable(leaf_seqs[2][0]).cuda())
        ans_leaf_lex = self.lexicon(Variable(leaf_seqs[0][1]).cuda())
        ans_leaf2_lex = self.lexicon(Variable(leaf_seqs[1][1]).cuda())
        ans_leaf3_lex = self.lexicon(Variable(leaf_seqs[2][1]).cuda())
        ans_lex_nt = self.constituent(Variable(lex_seqs[0][0]).cuda())
        ans_lex2_nt = self.constituent(Variable(lex_seqs[1][0]).cuda())
        ans_lex3_nt = self.constituent(Variable(lex_seqs[2][0]).cuda())
        ans_lex_lex = self.lexicon(Variable(lex_seqs[0][1]).cuda())
        ans_lex2_lex = self.lexicon(Variable(lex_seqs[1][1]).cuda())
        ans_lex3_lex = self.lexicon(Variable(lex_seqs[2][1]).cuda())
        ans_sib_nt = self.constituent(Variable(sibling_seqs[0][0]).cuda())
        ans_sib2_nt = self.constituent(Variable(sibling_seqs[1][0]).cuda())
        ans_sib_lex = self.lexicon(Variable(sibling_seqs[0][1]).cuda())
        ans_sib2_lex = self.lexicon(Variable(sibling_seqs[1][1]).cuda())
        '''

        ans_rule_embed = rule_seqs.clone()
        ans_rule_embed[:, 1:] = ans_rule_embed[:, :-1]
        ans_rule_embed[:, 0] = 0
        ans_rule_embed = self.rule(Variable(ans_rule_embed).cuda())

        '''
        flatten = [anc for lst in ancestors for anc in lst]
        f_lengths = [len(f) for f in flatten]
        flatten_ind = np.repeat(np.arange(len(ancestors)), trg_lengths)
        anc_embed = Variable(torch.zeros(len(flatten), max(f_lengths), self.lex_input_size + self.nt_input_size + self.rule_input_size).float().cuda())
        for i, seq in enumerate(flatten):
            end = len(flatten[i])
            if end:
                j = flatten_ind[i]
                anc_embed[i, :end, :] = torch.cat((ans_lex_lex[j, :end, :], ans_lex_nt[j, :end, :], ans_rule_embed[j, :end, :]), dim=1)
        f_lengths, perm_idx = torch.LongTensor(f_lengths).sort(0, descending=True)
        nonzeros = f_lengths.nonzero().squeeze(1)
        zeros = (f_lengths == 0).nonzero().squeeze(1)
        packed_input = pack_padded_sequence(anc_embed[nonzeros.cuda()], f_lengths[nonzeros].numpy(), batch_first=True)
        anc_output, anc_last_hidden = self.anc_decoder(packed_input)
        #anc_hidden, _ = pad_packed_sequence(anc_output, batch_first=True)
        anc_init_hidden = self.init_hidden(src_last_hidden, 'anc_') 
        #anc_hidden = torch.cat((anc_last_hidden[0].squeeze(0), anc_init_hidden), dim=2)
        _anc_hidden = Variable(torch.cuda.FloatTensor(len(flatten), self.hidden_size).fill_(0))
        _anc_hidden[:nonzeros.size(0)] = anc_last_hidden[0].squeeze(0)
        _anc_hidden = _anc_hidden[perm_idx.sort()[1].cuda()]
        _anc_hidden[perm_idx[zeros].cuda()] = anc_init_hidden[0].squeeze(0)

        anc_hidden = Variable(torch.cuda.FloatTensor(batch_size, max(trg_lengths), self.hidden_size).fill_(0))
        start = 0
        for x in range(batch_size):
            anc_hidden[x, :trg_lengths[x], :] = _anc_hidden[start:start+trg_lengths[x]]
            start += trg_lengths[x]
        '''

        ans_embed = torch.cat((ans_nt, ans_rule_embed), dim=2)
        '''
        else:
            ans_embed = torch.cat((ans_nt,
                                   ans_rule_embed,
                                   ans_leaf_nt, ans_leaf_lex,
                                   ans_leaf2_nt, ans_leaf2_lex,
                                   ans_leaf3_nt, ans_leaf3_lex,
                                   ans_lex_nt, ans_lex_lex,
                                   ans_lex2_nt, ans_lex2_lex,
                                   ans_lex3_nt, ans_lex3_lex
                                  ), dim=2)
            ans_embed = F.tanh(self.tree(ans_embed))
        else:
            tree_input = F.tanh(self.tree(torch.cat((ans_par_nt, ans_par_lex, 
                                                    ans_par2_nt, ans_par2_lex,
                                                    ans_sib_nt, ans_sib_lex,
                                                    ans_sib2_nt, ans_sib2_lex,
                                                    ans_leaf_nt, ans_leaf_lex,
                                                    ans_leaf2_nt, ans_leaf2_lex,
                                                    ans_lex_nt, ans_lex_lex,
                                                    ans_lex2_nt, ans_lex2_lex
                                                   ), dim=2)))
            ans_embed = torch.cat((ans_nt, 
                                   tree_input
                                  ), dim=2)
        else:
            tree_input = F.tanh(self.tree(torch.cat((ans_par_nt, ans_par_lex, 
                                                    ans_par2_nt, ans_par2_lex,
                                                    ans_sib_nt, ans_sib_lex,
                                                    ans_sib2_nt, ans_sib2_lex
                                                   ), dim=2)))
            ans_embed = torch.cat((ans_nt, 
                                   tree_input
                                  ), dim=2)
        '''
        trg_l = max(trg_lengths)
        batch_ind = torch.arange(batch_size).long().cuda()
        if self.lex_level == 0:
            decoder_outputs = Variable(torch.FloatTensor(batch_size, trg_l, self.hidden_size * 2 + self.context_size).cuda())
            context = self.attention((leaf_init_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths) 
        else:
            decoder_outputs = Variable(torch.FloatTensor(batch_size, trg_l, self.hidden_size * 3 + self.context_size).cuda())
            context = self.attention((torch.cat((leaf_init_hidden[0], lex_init_hidden[0]), dim=2), None), src_hidden, src_hidden.size(1), src_lengths) 
        #pos_embed = torch.cat((self.depth(Variable(positions[0]).cuda()), self.breadth(Variable(positions[1] / 2).cuda())), dim=2)
        for step in range(trg_l):
            #decoder_input = torch.cat((ans_embed[:, step, :].unsqueeze(1), tree_input[:, step, :].unsqueeze(1)), dim=2)
            decoder_input = ans_embed[:, step, :].unsqueeze(1)
            #rule_decoder_input = ans_rule_embed[:, step, :].unsqueeze(1)
            if self.lex_level >= 2:
            #if False:
                leaf_select = torch.cuda.LongTensor([x[step].item() for x in _leaf_indices])
                lex_select = torch.cuda.LongTensor([x[step].item() for x in _lex_indices])
                #decoder_input = torch.cat((decoder_input, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1), lex_decoder_hidden[batch_ind, lex_select, :].unsqueeze(1)), dim=2)
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                #rule_decoder_output, rule_decoder_hidden = self.rule_decoder(rule_decoder_input, rule_decoder_hidden)
                dec_cat_output = torch.cat((decoder_output, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1), lex_decoder_hidden[batch_ind, lex_select, :].unsqueeze(1)), dim=2)
                #dec_cat_output = torch.cat((decoder_output, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1), lex_decoder_hidden[batch_ind, lex_select, :].unsqueeze(1), anc_hidden[:, step, :].unsqueeze(1)), dim=2)
                #decoder_outputs[:, step, self.hidden_size * 3:] = tree_input[:, step, :]
                '''
                if step:
                    context = self.attention((decoder_outputs[:, step-1, self.hidden_size:self.hidden_size * 3].unsqueeze(0), None), src_hidden, src_hidden.size(1), src_lengths) 
                else:
                    context = self.attention((torch.cat((leaf_init_hidden[0], lex_init_hidden[0]), dim=2), None), src_hidden, src_hidden.size(1), src_lengths) 
                '''
                decoder_outputs[:, step, :self.hidden_size * 3] = dec_cat_output.squeeze(1)
                decoder_outputs[:, step, self.hidden_size * 3:] = context.squeeze(1)
                context = self.attention((dec_cat_output[:, :, self.hidden_size:].transpose(0, 1), None), src_hidden, src_hidden.size(1), src_lengths) 
            else:
                leaf_select = torch.cuda.LongTensor([x[step].item() for x in _leaf_indices])
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                dec_cat_output = torch.cat((decoder_output, leaf_decoder_hidden[batch_ind, leaf_select, :].unsqueeze(1)), dim=2)
                decoder_outputs[:, step, :self.hidden_size * 2] = dec_cat_output.squeeze(1)
                decoder_outputs[:, step, self.hidden_size * 2:] = context.squeeze(1)
                context = self.attention((dec_cat_output[:, :, self.hidden_size:].transpose(0, 1), None), src_hidden, src_hidden.size(1), src_lengths) 

        return decoder_outputs, ans_embed

    def masked_loss(self, logits, target, lengths, mask):
        batch_size = logits.size(0)
        max_len = lengths.data.max()
        logits_flat = logits.view(-1, logits.size(-1))
        log_probs_flat = F.log_softmax(logits_flat, dim=1)
        target_flat = target.contiguous().view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*target.size())
        l_mask  = torch.arange(max_len).long().cuda().expand(batch_size, max_len) < lengths.data.expand(max_len, batch_size).transpose(0, 1)
        _mask = Variable(l_mask * mask)
        losses = losses * _mask.float()
        loss = losses.sum() / _mask.float().sum()
        return loss

    def loss(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, word_mask, rule_mask, positions, ancestors):
        decoder_outputs, tree_input = self.forward(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, positions, ancestors)
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
        rule_loss = self.masked_loss(rules, Variable(rule_seqs).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), rule_mask.cuda().byte())
        if self.lex_level == 0:
            nt_loss = 0
        else:
            nt_loss = self.masked_loss(nts, Variable(trg_seqs[2]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), rule_mask.cuda().byte())
        return word_loss, nt_loss, rule_loss

    def generate(self, src_seqs, src_lengths, indices, max_len, beam_size, top_k):
        src_embed = self.lexicon(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 
        leaf_decoder_hidden = self.init_hidden(src_last_hidden, 'leaf_') 
        if self.lex_level == 0:
            context = self.attention((leaf_decoder_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths) 
        elif self.lex_level >= 2:
            lex_decoder_hidden = self.init_hidden(src_last_hidden, 'lex_')    
            rule_decoder_hidden = self.init_hidden(src_last_hidden, 'rule_')    
            context = self.attention((torch.cat((leaf_decoder_hidden[0], lex_decoder_hidden[0]), dim=2), None), src_hidden, src_hidden.size(1), src_lengths) 
            '''
            leaf_decoder_hidden = leaf_init_hidden[0].transpose(0, 1)
            lex_decoder_hidden = lex_init_hidden[0].transpose(0, 1)
            '''

        batch_size = src_embed.size(0)
        assert batch_size == 1
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        #context = self.attention(decoder_hidden, src_hidden, src_hidden.size(1), src_lengths) 
        if self.lex_level >= 2:
            #decoder_input = torch.cat((Variable(torch.zeros(batch_size, 1, self.decoder_input_size)).cuda(), context), dim=2)
            decoder_input = Variable(torch.zeros(batch_size, 1, self.decoder_input_size)).cuda()
            #decoder_input = torch.cat((decoder_input, leaf_decoder_hidden[0], lex_decoder_hidden[0]), dim=2)
            ans_embed = Variable(torch.zeros(self.nt_input_size * 1 + self.lex_input_size * 0 + self.rule_input_size * 0)).cuda()
            ans_embed[:self.nt_input_size] = self.constituent(Variable(torch.cuda.LongTensor([self.ROOT])))
            rule_decoder_input = self.rule(Variable(torch.cuda.LongTensor([0]))).unsqueeze(1)
            #tree_input = F.tanh(self.tree(Variable(torch.zeros(batch_size, 1, self.lex_input_size * 4)).cuda()))
            '''
            for x in range(3):
                ans_embed[self.nt_input_size+self.rule_input_size+(self.nt_input_size+self.lex_input_size)*x:self.nt_input_size*2+self.rule_input_size+(self.nt_input_size+self.lex_input_size)*x] = self.constituent(Variable(torch.cuda.LongTensor([0])))
                ans_embed[self.nt_input_size*2+self.rule_input_size+(self.nt_input_size+self.lex_input_size)*x:self.nt_input_size*2+self.rule_input_size+(self.nt_input_size+self.lex_input_size)*x+self.lex_input_size] = self.lexicon(Variable(torch.cuda.LongTensor([0])))
            ans_embed = F.tanh(self.tree(ans_embed)).unsqueeze(0).unsqueeze(0)
            '''
        else:
            decoder_input = Variable(torch.zeros(batch_size, 1, self.decoder_input_size)).cuda()
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        #rule_decoder_output, rule_decoder_hidden = self.rule_decoder(rule_decoder_input, rule_decoder_hidden)

        word_beam = torch.zeros(beam_size, max_len).long().cuda()
        rule_beam = torch.zeros(beam_size, max_len).long().cuda()
        nt_beam = torch.zeros(beam_size, max_len).long().cuda()
        word_count = torch.zeros(beam_size, top_k).long().cuda()
        rule_count = torch.zeros(beam_size, top_k).long().cuda()
        nt_count = torch.zeros(beam_size, top_k).long().cuda()
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
            word_count.fill_(1)
        elif self.lex_level == 2:
            decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0], lex_decoder_hidden[0]), dim=2)
            #decoder_output = self.hidden_out_fc2(F.relu(self.hidden_out_fc1(decoder_output)))
            self.nt_cand_num = 1
            nt_logits = self.nt_dist(self.nt_out(F.tanh(self.nt_hidden_out(decoder_output))))
            nt_logprobs, nt_argtop = torch.topk(F.log_softmax(nt_logits.squeeze(1), dim=1), self.nt_cand_num, dim=1)
            tag_embed = self.constituent(nt_argtop)
            rule_logits = self.rule_dist(F.tanh(self.rule_hidden_out(torch.cat((decoder_output, tag_embed), dim=2))))
            rule_logits = rule_logits.expand(self.nt_cand_num, 1, self.rule_vocab_size).clone()
            rule_select = torch.cuda.ByteTensor(self.nt_cand_num, 1, self.rule_vocab_size).fill_(0)
            for y in range(self.nt_cand_num):
                rule_select[y, 0, :] = self.rules_by_nt[nt_argtop.data[0, y]]
            rule_logits[~rule_select] = -np.inf
            rule_logprobs, rule_argtop = torch.topk(F.log_softmax(rule_logits.squeeze(1), dim=1), beam_size, dim=1)
            nt_rule_logprobs = rule_logprobs + nt_logprobs.expand(beam_size, self.nt_cand_num).transpose(0, 1)

            #word_logits = self.lex_dist(self.lex_out(torch.cat((decoder_output, tag_embed), dim=2)))
            word_logits = self.lex_dist(self.lex_out(F.tanh(self.lex_hidden_out(torch.cat((decoder_output, tag_embed), dim=2)))))
            word_logprobs, word_argtop = torch.topk(F.log_softmax(word_logits.squeeze(1), dim=1), beam_size, dim=1)
            total_logprobs = word_logprobs.expand(beam_size * self.nt_cand_num, beam_size).transpose(0, 1) + \
                             nt_rule_logprobs.view(-1).expand(beam_size, beam_size * self.nt_cand_num)
            logprobs, argtop = torch.topk(total_logprobs.view(-1), beam_size, dim=0)
            word_beam[:, 0] = word_argtop.squeeze(0).data[argtop.data / (beam_size * self.nt_cand_num)]
            word_beam_probs = word_logprobs.squeeze(0).data[argtop.data / (beam_size * self.nt_cand_num)]
            nt_rule_x = (argtop.data % (self.nt_cand_num * beam_size)) / beam_size
            nt_rule_y = (argtop.data % (self.nt_cand_num * beam_size)) % beam_size
            rule_beam[:, 0] = rule_argtop.data[nt_rule_x, nt_rule_y]
            rule_beam_probs = rule_logprobs.data[nt_rule_x, nt_rule_y]
            nt_beam[:, 0] = nt_argtop.squeeze(0).data[nt_rule_x]
            nt_beam_probs = nt_logprobs.squeeze(0).data[nt_rule_x]
            word_count.fill_(1)
            rule_count.fill_(1)
            nt_count.fill_(1)
            self.nt_cand_num = 3
        else:
            #decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0], lex_decoder_hidden[0], anc_decoder_hidden[0]), dim=2)
            decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0], lex_decoder_hidden[0], context), dim=2)
            if self.lex_level == 3:
                #context = self.attention((decoder_output, None), src_hidden, src_hidden.size(1), src_lengths) 
                #position = torch.cat((self.depth(Variable(torch.cuda.LongTensor([0]))), self.breadth(Variable(torch.cuda.LongTensor([0])))), dim=1).unsqueeze(0)
                #decoder_output = torch.cat((decoder_output, position), dim=2)
                pass
            self.nt_cand_num = 2
            self.rule_cand_num = 5
            self.lex_cand_num = 15
            total_logprobs = torch.zeros(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num).cuda()

            word_logits = self.lex_dist(self.lex_out(F.tanh(self.lex_hidden_out(decoder_output))))
            word_logprobs, word_argtop = torch.topk(F.log_softmax(word_logits.squeeze(1), dim=1), self.lex_cand_num, dim=1)
            total_logprobs += word_logprobs.data.expand(self.rule_cand_num, self.nt_cand_num, self.lex_cand_num).transpose(0, 2)

            decoder_output = decoder_output.squeeze(0).expand(self.lex_cand_num, self.hidden_size * 3 + self.context_size)
            word_embed = self.lexicon(word_argtop).squeeze(0)
            nt_logits = self.nt_dist(self.nt_out(F.tanh(self.nt_hidden_out(torch.cat((decoder_output, word_embed), dim=1)))))
            nt_logprobs, nt_argtop = torch.topk(F.log_softmax(nt_logits, dim=1), self.nt_cand_num, dim=1)
            total_logprobs += nt_logprobs.data.expand(self.rule_cand_num, self.lex_cand_num, self.nt_cand_num).transpose(0, 1).transpose(1, 2)

            tag_embed = self.constituent(nt_argtop)
            word_embed = word_embed.expand(self.nt_cand_num, self.lex_cand_num, self.lex_input_size).transpose(0, 1)
            decoder_output = decoder_output.expand(self.nt_cand_num, self.lex_cand_num, self.hidden_size * 3 + self.context_size).transpose(0, 1)
            rule_logits = self.rule_dist(self.rule_out(F.tanh(self.rule_hidden_out(torch.cat((decoder_output, tag_embed, word_embed), dim=2)))))
            rule_select = torch.cuda.ByteTensor(self.lex_cand_num, self.nt_cand_num, self.rule_vocab_size).fill_(False)
            for x in range(self.lex_cand_num):
                for y in range(self.nt_cand_num):
                    rule_select[x, y, :] = self.rules_by_nt[nt_argtop.data[x, y]]
            rule_logits[~rule_select] = -np.inf
            rule_logprobs, rule_argtop = torch.topk(F.log_softmax(rule_logits, dim=2), self.rule_cand_num, dim=2)
            total_logprobs += rule_logprobs.data

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
            nt_count.fill_(1)
        hidden = (decoder_hidden[0].clone(), decoder_hidden[1].clone())
        decoder_hidden = (decoder_hidden[0].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1),
                          decoder_hidden[1].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1))
        '''
        rule_decoder_hidden = (rule_decoder_hidden[0].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1),
                               rule_decoder_hidden[1].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1))
        '''
        leaf_decoder_hidden = (leaf_decoder_hidden[0].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1),
                               leaf_decoder_hidden[1].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1))
        if self.lex_level >= 2:
            lex_decoder_hidden = (lex_decoder_hidden[0].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1),
                                  lex_decoder_hidden[1].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1))
            lex_decoder_input = self.lexicon(word_beam[:, 0]).unsqueeze(1)
            _, lex_decoder_hidden = self.lex_decoder(lex_decoder_input, lex_decoder_hidden)
        src_hidden = src_hidden.expand(beam_size, src_hidden.size(1), self.hidden_size)

        current = np.array([None for x in range(beam_size)])
        num_leaves = torch.cuda.LongTensor(beam_size).fill_(0)
        leaves = torch.cuda.LongTensor([[[0, 0], [0, 0], [0, 0]] for x in range(beam_size)])
        lexicons = torch.cuda.LongTensor([[[0, 0], [0, 0], [0, 0]] for x in range(beam_size)])
        final = np.array([None for x in range(beam_size)])
        tree_dict = np.array([{} for x in range(beam_size)])
        for y in range(beam_size):
            rule = self.rules_by_id[rule_beam[y, 0]]
            if self.lex_level == 0:
                current[y] = Node('{}__{}__{}'.format('ROOT', word_beam[y, 0], rule))
            else:
                tag = self.nt_by_id[nt_beam[y, 0]]
                current[y] = Node('{}__{}__{} [{}]'.format('ROOT', word_beam[y, 0], rule, rule.split().index(tag)))
                #tree_dict[y][current[y]] = anc_decoder_hidden

        def inheritance(rule):
            return literal_eval(rule[rule.find('['):rule.find(']')+1])

        for t in range(max_len-1):
            ans_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_par_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_par_rule = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_par2_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_par_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_par2_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_leaf_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_leaf2_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_leaf3_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_leaf_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_leaf2_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_leaf3_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_lex_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_lex2_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_lex3_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_lex_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_lex2_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_lex3_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_sib_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_sib2_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_sib_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_sib2_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_depth = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
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
                    ans_par_nt[x] = self.nt_dictionary[par_nt]
                    ans_par_lex[x] = int(par_lex)
                    ans_par_rule[x] = self.rule_dictionary['RULE: {}'.format(rule[:rule.find('[')-1])]
                    ans_depth[x] = current[x].depth
                    if current[x].parent is not None:
                        par2_nt, par2_lex, __ = current[x].parent.name.split('__')
                        ans_par2_nt[x] = self.nt_dictionary[par2_nt]
                        ans_par2_lex[x] = int(par2_lex)
                        pos = current[x].parent.children.index(current[x])
                        if pos > 0:
                            sib_nt, sib_lex, __ = current[x].parent.children[pos-1].name.split('__')
                            ans_sib_nt[x] = self.nt_dictionary[sib_nt]
                            ans_sib_lex[x] = int(sib_lex)
                        if current[x].parent.parent is not None:
                            pos = current[x].parent.parent.children.index(current[x].parent)
                            if pos > 0:
                                sib_nt, sib_lex, __ = current[x].parent.parent.children[pos-1].name.split('__')
                                ans_sib2_nt[x] = self.nt_dictionary[sib_nt]
                                ans_sib2_lex[x] = int(sib_lex)
                    ans_leaf_nt[x] = leaves[x][0][0]
                    ans_leaf_lex[x] = leaves[x][0][1]
                    ans_leaf2_nt[x] = leaves[x][1][0]
                    ans_leaf2_lex[x] = leaves[x][1][1]
                    ans_leaf3_nt[x] = leaves[x][2][0]
                    ans_leaf3_lex[x] = leaves[x][2][1]
                    ans_lex_nt[x] = lexicons[x][0][0]
                    ans_lex_lex[x] = lexicons[x][0][1]
                    ans_lex2_nt[x] = lexicons[x][1][0]
                    ans_lex2_lex[x] = lexicons[x][1][1]
                    ans_lex3_nt[x] = lexicons[x][2][0]
                    ans_lex3_lex[x] = lexicons[x][2][1]
            ans_nt = self.constituent(ans_nt)
            ans_par_nt = self.constituent(ans_par_nt)
            ans_par2_nt = self.constituent(ans_par2_nt)
            ans_par_lex = self.lexicon(ans_par_lex)
            ans_par2_lex = self.lexicon(ans_par2_lex)
            ans_leaf_nt = self.constituent(ans_leaf_nt)
            ans_leaf2_nt = self.constituent(ans_leaf2_nt)
            ans_leaf3_nt = self.constituent(ans_leaf3_nt)
            ans_leaf_lex = self.lexicon(ans_leaf_lex)
            ans_leaf2_lex = self.lexicon(ans_leaf2_lex)
            ans_leaf3_lex = self.lexicon(ans_leaf3_lex)
            ans_lex_nt = self.constituent(ans_lex_nt)
            ans_lex2_nt = self.constituent(ans_lex2_nt)
            ans_lex3_nt = self.constituent(ans_lex3_nt)
            ans_lex_lex = self.lexicon(ans_lex_lex)
            ans_lex2_lex = self.lexicon(ans_lex2_lex)
            ans_lex3_lex = self.lexicon(ans_lex3_lex)
            ans_sib_nt = self.constituent(ans_sib_nt)
            ans_sib2_nt = self.constituent(ans_sib2_nt)
            ans_sib_lex = self.lexicon(ans_sib_lex)
            ans_sib2_lex = self.lexicon(ans_sib2_lex)
            ans_par_rule = self.rule(ans_par_rule)
            if self.lex_level == 0:
                ans_embed = torch.cat((ans_nt, ans_par_rule), dim=1)
            else:
                if self.lex_level == 1:
                    tree_input = F.tanh(self.tree(torch.cat((ans_par_nt, ans_par_lex, 
                                                            ans_par2_nt, ans_par2_lex,
                                                            ans_sib_nt, ans_sib_lex,
                                                            ans_sib2_nt, ans_sib2_lex
                                                           ), dim=1)))
                    ans_embed = torch.cat((ans_nt, 
                                           ans_leaf_nt, ans_leaf_lex,
                                           ans_leaf2_nt, ans_leaf2_lex,
                                           ans_lex_nt, ans_lex_lex,
                                           ans_lex2_nt, ans_lex2_lex,
                                           tree_input
                                          ), dim=1)
                else:
                    '''
                    tree_input = F.tanh(self.tree(torch.cat((ans_par_nt, ans_par_lex, 
                                                            ans_par2_nt, ans_par2_lex,
                                                            ans_sib_nt, ans_sib_lex,
                                                            ans_sib2_nt, ans_sib2_lex,
                                                            ans_leaf_nt, ans_leaf_lex,
                                                            ans_leaf2_nt, ans_leaf2_lex,
                                                            ans_lex_nt, ans_lex_lex,
                                                            ans_lex2_nt, ans_lex2_lex
                                                           ), dim=1)))
                    '''
                    ans_embed = torch.cat((ans_nt, ans_par_rule), dim=1)
                    #ans_embed = ans_nt
                    rule_decoder_input = ans_par_rule.unsqueeze(1)
                    tree_input = F.tanh(self.tree(torch.cat((ans_par_lex, 
                                                     ans_par2_lex,
                                                     ans_sib_lex,
                                                     ans_sib2_lex
                                                    ), dim=1)))
                    #pos_embed = torch.cat((self.depth(ans_depth), self.breadth(Variable(num_leaves / 2).cuda())), dim=1).unsqueeze(1)
                    '''
                    ans_embed = torch.cat((ans_nt,
                                           ans_par_rule,
                                           ans_leaf_nt, ans_leaf_lex,
                                           ans_leaf2_nt, ans_leaf2_lex,
                                           ans_leaf3_nt, ans_leaf3_lex,
                                           ans_lex_nt, ans_lex_lex,
                                           ans_lex2_nt, ans_lex2_lex,
                                           ans_lex3_nt, ans_lex3_lex
                                          ), dim=1)
                    ans_embed = F.tanh(self.tree(ans_embed))
                    '''
            #context = self.attention(decoder_hidden, src_hidden, src_hidden.size(1), src_lengths) 
            #decoder_input = torch.cat((ans_embed.unsqueeze(1), context), dim=2)
            decoder_input = ans_embed.unsqueeze(1)
               
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            if self.lex_level == 0:
                context = self.attention((leaf_decoder_hidden[0], None), src_hidden, src_hidden.size(1), src_lengths) 
            elif self.lex_level >= 2:
                decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0].transpose(0, 1), lex_decoder_hidden[0].transpose(0, 1)), dim=2)
                context = self.attention((decoder_output[:, :, self.hidden_size:].transpose(0, 1), None), src_hidden, src_hidden.size(1), src_lengths) 
                decoder_output = torch.cat((decoder_output, context), dim=2)
                if self.lex_level == 3:
                    #context = self.attention((decoder_output.unsqueeze(0), None), src_hidden, src_hidden.size(1), src_lengths) 
                    #decoder_output = torch.cat((decoder_output, pos_embed), dim=2)
                    pass
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
            elif self.lex_level == 2:
                nt_logits = self.nt_dist(self.nt_out(F.tanh(self.nt_hidden_out(decoder_output))))
                nt_logits[:, :, 0] = -np.inf
                nt_logprobs, nt_argtop = torch.topk(F.log_softmax(nt_logits.squeeze(1), dim=1), self.nt_cand_num, dim=1)
                tag_embed = self.constituent(nt_argtop.view(-1))
                ans_embed = ans_embed.expand(self.nt_cand_num, beam_size, self.tree_input_size).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1)
                decoder_output = decoder_output.expand(beam_size, self.nt_cand_num, self.hidden_size * 3).contiguous().view(beam_size * self.nt_cand_num, -1)

                #rule_logits = self.rule_dist(decoder_output)
                rule_logits = self.rule_dist(F.tanh(self.rule_hidden_out(torch.cat((decoder_output, tag_embed), dim=1))))
                rule_logits = rule_logits.view(beam_size, self.nt_cand_num, self.rule_vocab_size)
                rule_logits[:, :, 0] = -np.inf
                #rule_logits = rule_logits.expand(beam_size, self.nt_cand_num, self.rule_vocab_size).clone()
                rule_select = torch.cuda.ByteTensor(beam_size, self.nt_cand_num, self.rule_vocab_size).fill_(0)
                dup_mask = torch.cuda.ByteTensor(beam_size * self.nt_cand_num, top_k).fill_(False)
                for x in range(beam_size):
                    for y in range(self.nt_cand_num):
                        rule_select[x, y, :] = self.rules_by_nt[nt_argtop.data[x, y]]
                    if rule_mask[x, 0] == 0:
                        if word_mask[x, 0] == 0:
                            dup_mask[x*self.nt_cand_num:(x+1)*self.nt_cand_num, :] = True
                            dup_mask[x*self.nt_cand_num, 0] = False
                        else:
                            nt_argtop[x] = self.nt_dictionary[curr_nt[x]]
                            dup_mask[x*self.nt_cand_num+1:(x+1)*self.nt_cand_num, :] = True

                #word_logits = self.lex_dist(self.lex_out(torch.cat((decoder_output, tag_embed), dim=1)))
                word_logits = self.lex_dist(self.lex_out(F.tanh(self.lex_hidden_out(torch.cat((decoder_output, tag_embed), dim=1)))))
                word_logits[:, 0] = -np.inf
                word_logprobs, word_argtop = torch.topk(F.log_softmax(word_logits.squeeze(1), dim=1), top_k, dim=1)
                
                rule_logits[~rule_select] = -np.inf
                rule_logprobs, rule_argtop = torch.topk(F.log_softmax(rule_logits.view(beam_size * self.nt_cand_num, -1), dim=1), top_k, dim=1)
                nt_rule_logprobs = rule_logprobs + nt_logprobs.view(-1).expand(top_k, beam_size * self.nt_cand_num).transpose(0, 1)
                rule_mask = rule_mask.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1)
                word_mask = word_mask.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1)
                '''
                dup_mask = dup_mask.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1)
                dup_mask[~rule_mask] = 0
                dup_mask[torch.arange(beam_size).long().cuda() * self.nt_cand_num, torch.zeros(beam_size).long().cuda()] = 0
                '''
                rule_beam_logprobs = rule_beam_probs.expand(top_k, self.nt_cand_num, beam_size).transpose(0, 2).contiguous().view(beam_size * self.nt_cand_num, -1) + \
                                     nt_rule_logprobs.data * rule_mask.float()
                rule_beam_logprobs[dup_mask] = -np.inf
                #word_beam_logprobs = word_beam_logprobs.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1)
                word_beam_logprobs = (word_beam_probs).expand(top_k, self.nt_cand_num, beam_size).transpose(0, 2).contiguous().view(beam_size * self.nt_cand_num, -1) + word_logprobs.data * word_mask.float()
                word_beam_logprobs[dup_mask] = -np.inf
                #word_logprobs = word_logprobs.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1)
                #word_argtop = word_argtop.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1)
                total_logprobs = (word_beam_logprobs - 0) / word_count.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1).float() + \
                                 (rule_beam_logprobs - 0) / rule_count.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1).float()
                #total_logprobs = word_beam_logprobs / np.sqrt(t + 1) + rule_beam_logprobs / (t + 1)
                total_logprobs[total_logprobs != total_logprobs] = -np.inf
                best_probs, best_args = total_logprobs.view(-1).topk(beam_size)
            else:
                total_logprobs = torch.zeros(beam_size, self.lex_cand_num, self.nt_cand_num, self.rule_cand_num).cuda()
                decoder_output = decoder_output.squeeze(1)
                word_mask = word_mask[:, 0].expand(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num, beam_size).permute(3, 0, 1, 2).float()
                _word_count = word_count[:, 0].expand(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num, beam_size).permute(3, 0, 1, 2).float()
                rule_mask = rule_mask[:, 0].expand(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num, beam_size).permute(3, 0, 1, 2).float()
                _rule_count = rule_count[:, 0].expand(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num, beam_size).permute(3, 0, 1, 2).float()

                word_logits = self.lex_dist(self.lex_out(F.tanh(self.lex_hidden_out(decoder_output))))
                word_logprobs, word_argtop = torch.topk(F.log_softmax(word_logits, dim=1), self.lex_cand_num, dim=1)
                word_beam_logprobs = word_logprobs.data.expand(self.nt_cand_num, self.rule_cand_num, beam_size, self.lex_cand_num).permute(2, 3, 0, 1)

                dup_mask = torch.cuda.ByteTensor(beam_size, self.lex_cand_num, self.nt_cand_num, self.rule_cand_num).fill_(False)
                for x in range(beam_size):
                    if rule_mask[x, 0, 0, 0] == 0:
                        if word_mask[x, 0, 0, 0] == 0:
                            dup_mask[x, :, :, :] = True
                            dup_mask[x, 0, 0, 0] = False
                        else:
                            dup_mask[x, :, :, :] = True
                            dup_mask[x, :, 0, 0] = False
                    elif word_mask[x, 0, 0, 0] == 0:
                        dup_mask[x, :, :, :] = True
                        dup_mask[x, 0, :, :] = False
                        inherit = current[x].name.split('__')[1]
                        word_argtop[x] = int(inherit)

                word_embed = self.lexicon(word_argtop)
                decoder_output = decoder_output.expand(self.lex_cand_num, beam_size, self.hidden_size * 4 + self.pos_embed_size).transpose(0, 1)
                nt_logits = self.nt_dist(self.nt_out(F.tanh(self.nt_hidden_out(torch.cat((decoder_output, word_embed), dim=2)))))
                nt_logits = F.log_softmax(nt_logits, dim=2)
                nt_logits[:, :, 0] = -np.inf
                nt_logits[:, :, 1] = -np.inf
                nt_logprobs, nt_argtop = torch.topk(nt_logits, self.nt_cand_num, dim=2)
                nt_beam_logprobs = nt_logprobs.data.expand(self.rule_cand_num, beam_size, self.lex_cand_num, self.nt_cand_num).permute(1, 2, 3, 0)

                tag_embed = self.constituent(nt_argtop.view(beam_size * self.lex_cand_num, -1)).view(beam_size, self.lex_cand_num, self.nt_cand_num, -1)
                decoder_output = decoder_output.expand(self.nt_cand_num, beam_size, self.lex_cand_num, self.hidden_size * 4 + self.pos_embed_size).permute(1, 2, 0, 3)
                word_embed = word_embed.expand(self.nt_cand_num, beam_size, self.lex_cand_num, self.lex_input_size).permute(1, 2, 0, 3)
                rule_logits = self.rule_dist(self.rule_out(F.tanh(self.rule_hidden_out(torch.cat((decoder_output, tag_embed, word_embed), dim=3)))))
                rule_logits = F.log_softmax(rule_logits, dim=3)
                rule_select = torch.cuda.ByteTensor(beam_size, self.lex_cand_num, self.nt_cand_num, self.rule_vocab_size).fill_(False)
                for x in range(self.lex_cand_num):
                    for y in range(self.nt_cand_num):
                        for z in range(beam_size):
                            rule_select[z, x, y, :] = self.rules_by_nt[nt_argtop.data[z, x, y]]
                rule_logits[~rule_select] = -np.inf
                rule_logprobs, rule_argtop = torch.topk(rule_logits, self.rule_cand_num, dim=3)
                rule_beam_logprobs = rule_logprobs.data

                word_beam_logprobs = word_beam_logprobs * word_mask + word_beam_probs.expand(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num, beam_size).permute(3, 0, 1, 2)
                nt_beam_logprobs = nt_beam_logprobs * rule_mask + nt_beam_probs.expand(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num, beam_size).permute(3, 0, 1, 2)
                rule_beam_logprobs = rule_beam_logprobs * rule_mask + rule_beam_probs.expand(self.lex_cand_num, self.nt_cand_num, self.rule_cand_num, beam_size).permute(3, 0, 1, 2)

                total_logprobs = word_beam_logprobs / _word_count + nt_beam_logprobs / _rule_count + rule_beam_logprobs / _rule_count
                total_logprobs[dup_mask] = -np.inf

                logprobs, argtop = torch.topk(total_logprobs.view(-1), beam_size, dim=0)
                argtop_ind = torch.cuda.LongTensor(beam_size, 4)
                argtop_ind[:, 0] = argtop / (self.lex_cand_num * self.nt_cand_num * self.rule_cand_num)
                argtop_ind[:, 1] = (argtop % (self.lex_cand_num * self.nt_cand_num * self.rule_cand_num)) / (self.nt_cand_num * self.rule_cand_num)
                argtop_ind[:, 2] = ((argtop % (self.lex_cand_num * self.nt_cand_num * self.rule_cand_num)) % (self.nt_cand_num * self.rule_cand_num)) / self.rule_cand_num
                argtop_ind[:, 3] = ((argtop % (self.lex_cand_num * self.nt_cand_num * self.rule_cand_num)) % (self.nt_cand_num * self.rule_cand_num)) % self.rule_cand_num

            decoder_hidden = (decoder_hidden[0].view(1, batch_size, beam_size, -1), decoder_hidden[1].view(1, batch_size, beam_size, -1))
            leaf_decoder_hidden = (leaf_decoder_hidden[0].view(1, batch_size, beam_size, -1),
                                   leaf_decoder_hidden[1].view(1, batch_size, beam_size, -1))
            if self.lex_level >= 2:
                lex_decoder_hidden = (lex_decoder_hidden[0].view(1, batch_size, beam_size, -1),
                                      lex_decoder_hidden[1].view(1, batch_size, beam_size, -1))
                #rule_decoder_hidden = (rule_decoder_hidden[0].view(1, batch_size, beam_size, -1), rule_decoder_hidden[1].view(1, batch_size, beam_size, -1))

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
                last = argtop_ind[:, 0]
                current = current[last.tolist()]
                leaves = leaves[last.tolist()]
                lexicons = lexicons[last.tolist()]
                curr_nt = curr_nt[last.tolist()]
                final = final[last.tolist()]
                tree_dict = tree_dict[last.tolist()]
                num_leaves = num_leaves[last.tolist()]
                word_beam = word_beam[last]
                rule_beam = rule_beam[last]
                word_count = word_count[last]
                rule_count = rule_count[last]
                word_beam[:, t+1] = word_argtop.data[last, argtop_ind[:, 1]]
                nt_beam[:, t+1] = nt_argtop.data[last, argtop_ind[:, 1], argtop_ind[:, 2]]
                rule_beam[:, t+1] = rule_argtop.data[last, argtop_ind[:, 1], argtop_ind[:, 2], argtop_ind[:, 3]]
                word_beam_probs = word_beam_logprobs[last, argtop_ind[:, 1], argtop_ind[:, 2], argtop_ind[:, 3]]
                nt_beam_probs = nt_beam_logprobs[last, argtop_ind[:, 1], argtop_ind[:, 2], argtop_ind[:, 3]]
                rule_beam_probs = rule_beam_logprobs[last, argtop_ind[:, 1], argtop_ind[:, 2], argtop_ind[:, 3]]

            decoder_hidden[0][:, 0, :, :] = decoder_hidden[0][:, 0, :, :][:, last, :]
            decoder_hidden[1][:, 0, :, :] = decoder_hidden[1][:, 0, :, :][:, last, :]
            decoder_hidden = (decoder_hidden[0].view(1, batch_size * beam_size, -1), decoder_hidden[1].view(1, batch_size * beam_size, -1))

            leaf_decoder_hidden[0][:, 0, :, :] = leaf_decoder_hidden[0][:, 0, :, :][:, last, :]
            leaf_decoder_hidden[1][:, 0, :, :] = leaf_decoder_hidden[1][:, 0, :, :][:, last, :]
            leaf_decoder_hidden = (leaf_decoder_hidden[0].view(1, batch_size * beam_size, -1),
                                   leaf_decoder_hidden[1].view(1, batch_size * beam_size, -1))
            if self.lex_level >= 2:
                lex_decoder_hidden[0][:, 0, :, :] = lex_decoder_hidden[0][:, 0, :, :][:, last, :]
                lex_decoder_hidden[1][:, 0, :, :] = lex_decoder_hidden[1][:, 0, :, :][:, last, :]
                
                lex_decoder_hidden = (lex_decoder_hidden[0].view(1, batch_size * beam_size, -1),
                                      lex_decoder_hidden[1].view(1, batch_size * beam_size, -1))

                '''
                rule_decoder_hidden[0][:, 0, :, :] = rule_decoder_hidden[0][:, 0, :, :][:, last, :]
                rule_decoder_hidden[1][:, 0, :, :] = rule_decoder_hidden[1][:, 0, :, :][:, last, :]
                rule_decoder_hidden = (rule_decoder_hidden[0].view(1, batch_size * beam_size, -1), rule_decoder_hidden[1].view(1, batch_size * beam_size, -1))
                '''

            #beam_eos = beam_eos | (beam[:, :, t+1] == self.eou).data
            
            if self.lex_level == 0:
                for x in range(beam_size):
                    current[x] = deepcopy(current[x])
                    if current[x] is None:
                        continue
                    if self.preterminal[curr_nt[x]]:
                        word = word_beam[x, t+1]
                        ch = Node('{}__{}__ '.format(curr_nt[x], word), parent=current[x])
                        leaves[x][1] = leaves[x][0]
                        leaves[x][0][0] = self.nt_dictionary[curr_nt[x]]
                        leaves[x][0][1] = int(word)

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
                        leaves[x][2] = leaves[x][1]
                        leaves[x][1] = leaves[x][0]
                        leaves[x][0][0] = self.nt_dictionary[curr_nt[x]]
                        leaves[x][0][1] = int(word)

                        lex_decoder_input = self.lexicon(Variable(torch.cuda.LongTensor([int(word)]))).unsqueeze(1)
                        _, _lex_decoder_hidden = self.lex_decoder(lex_decoder_input, (lex_decoder_hidden[0][:, x, :].unsqueeze(1), lex_decoder_hidden[1][:, x, :].unsqueeze(1)))
                        lex_decoder_hidden[0][:, x, :] = _lex_decoder_hidden[0][0]
                        lex_decoder_hidden[1][:, x, :] = _lex_decoder_hidden[1][0]

                        _, _leaf_decoder_hidden = self.leaf_decoder(lex_decoder_input, (leaf_decoder_hidden[0][:, x, :].unsqueeze(1), leaf_decoder_hidden[1][:, x, :].unsqueeze(1)))
                        leaf_decoder_hidden[0][:, x, :] = _leaf_decoder_hidden[0][0]
                        leaf_decoder_hidden[1][:, x, :] = _leaf_decoder_hidden[1][0]

                        num_leaves[x] += 1
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
                    else:
                        _, word, par_rule = current[x].name.split('__')
                        rule = self.rules_by_id[rule_beam[x, t+1]]
                        if len(current[x].children) not in inheritance(par_rule):
                            word = word_beam[x, t+1]
                            lexicons[x][2] = lexicons[x][1]
                            lexicons[x][1] = lexicons[x][0]
                            lexicons[x][0][0] = self.nt_dictionary[curr_nt[x]]
                            lexicons[x][0][1] = int(word)

                        lex_decoder_input = self.lexicon(Variable(torch.cuda.LongTensor([int(word)]))).unsqueeze(1)
                        _, _lex_decoder_hidden = self.lex_decoder(lex_decoder_input, (lex_decoder_hidden[0][:, x, :].unsqueeze(1), lex_decoder_hidden[1][:, x, :].unsqueeze(1)))
                        lex_decoder_hidden[0][:, x, :] = _lex_decoder_hidden[0][0]
                        lex_decoder_hidden[1][:, x, :] = _lex_decoder_hidden[1][0]

                        tag = self.nt_by_id[nt_beam[x, t+1]]
                        current[x] = Node('{}__{}__{} [{}]'.format(curr_nt[x], word, rule, rule.split().index(tag)), parent=current[x])
            if (current == None).all():
                break

        '''
        best, best_arg = beam_probs.max(1)
        generations = beam[torch.arange(_batch_size).long().cuda(), best_arg.data].data.cpu()
        return generations, best
        '''
        return final[0]


class LexicalizedGrammarDualDecoder(nn.Module):
    def __init__(self, lex_input_size, nt_input_size, rule_input_size, 
                 context_size, hidden_size, 
                 lex_vocab_size, nt_vocab_size, rule_vocab_size, 
                 lex_vectors, nt_vectors,
                 lex_dictionary, nt_dictionary, rule_dictionary,
                 lex_level):
        super(LexicalizedGrammarDualDecoder, self).__init__()
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
        self.encoder = nn.LSTM(lex_input_size, hidden_size)
        self.lex_out = nn.Linear(hidden_size * 2 + nt_input_size, lex_input_size)
        self.nt_dist = nn.Linear(nt_input_size, nt_vocab_size)
        self.nt_out = nn.Linear(hidden_size * 1, nt_input_size)
        self.nt_dist.weight = self.constituent.weight
        self.tree_input_size = lex_input_size + nt_input_size
        self.tree = nn.Linear((lex_input_size + nt_input_size) * 8, self.tree_input_size)
        self.lex_tree = nn.Linear((lex_input_size + nt_input_size) * 8, self.tree_input_size)
        self.leaf_tree = nn.Linear((lex_input_size + nt_input_size) * 8, self.tree_input_size)
        self.leaf_decoder = nn.LSTM(lex_input_size + nt_input_size + context_size + self.tree_input_size, hidden_size, batch_first=True, dropout=0.5)
        self.lex_decoder = nn.LSTM(lex_input_size + nt_input_size + context_size + self.tree_input_size, hidden_size, batch_first=True, dropout=0.5)
        self.decoder_input_size = nt_input_size + self.tree_input_size + hidden_size * 2

        self.leaf_hidden_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.leaf_hidden_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.leaf_cell_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.leaf_cell_fc2 = nn.Linear(hidden_size * 2, hidden_size)

        self.lex_hidden_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.lex_hidden_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.lex_cell_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.lex_cell_fc2 = nn.Linear(hidden_size * 2, hidden_size)

        self.decoder = nn.LSTM(self.decoder_input_size + context_size, hidden_size, batch_first=True, dropout=0.5)
        #self.rule_out = nn.Linear(hidden_size, rule_input_size)
        self.lex_dist = nn.Linear(lex_input_size, lex_vocab_size)
        self.rule_dist = nn.Linear(hidden_size * 1, rule_vocab_size)
        self.hidden_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.hidden_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.cell_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.lex_dictionary = lex_dictionary
        self.nt_dictionary = nt_dictionary
        self.rule_dictionary = rule_dictionary
        self.eou = lex_dictionary['__eou__']
        self.lex_dist.weight = self.lexicon.weight

        self.a_key = nn.Linear(hidden_size, 200)
        self.leaf_a_key = nn.Linear(hidden_size, 200)
        self.lex_a_key = nn.Linear(hidden_size, 200)
        self.q_key = nn.Linear(hidden_size, 200)
        self.leaf_q_key = nn.Linear(hidden_size, 200)
        self.lex_q_key = nn.Linear(hidden_size, 200)
        self.q_value = nn.Linear(hidden_size, context_size)
        self.leaf_q_value = nn.Linear(hidden_size, context_size)
        self.lex_q_value = nn.Linear(hidden_size, context_size)

        self.init_forget_bias(self.encoder, 0)
        self.init_forget_bias(self.decoder, 0)
        self.init_forget_bias(self.leaf_decoder, 0)
        self.init_forget_bias(self.lex_decoder, 0)

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

        self.rules_by_nt = {}
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
        self.leaf_decoder.flatten_parameters()
        self.lex_decoder.flatten_parameters()

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

    def attention(self, decoder_hidden, src_hidden, src_max_len, src_lengths, name=''):
        a_key = eval('F.tanh(self.{}a_key(decoder_hidden[0].squeeze(0)))'.format(name))
        q_key = eval('F.tanh(self.{}q_key(src_hidden))'.format(name))
        q_value = eval('self.{}q_value(src_hidden)'.format(name))
        q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
        q_mask  = torch.arange(src_max_len).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(src_max_len, 1).transpose(0, 1)
        q_energy[~q_mask] = -np.inf
        q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
        q_context = torch.bmm(q_weights, q_value).squeeze(1)
        context = q_context.unsqueeze(1)

        return context

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
        tar_nt_embed = self.constituent(Variable(tar_seqs[0]).cuda())
        _tar_embed = torch.cat((tar_lex_embed, tar_nt_embed), dim=2)
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

    def forward(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices):
        batch_size = src_seqs.size(0)

        src_embed = self.lexicon(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 
        leaf_decoder_hidden = self.init_hidden(src_last_hidden, 'leaf_') 
        lex_decoder_hidden = self.init_hidden(src_last_hidden, 'lex_') 
        '''
        leaf_decoder_hidden, _leaf_indices = self.encode(trg_seqs, leaf_indices, 'leaf', leaf_init_hidden)
        lex_decoder_hidden, _lex_indices = self.encode(trg_seqs, lex_indices, 'lex', lex_init_hidden)
        leaf_decoder_hidden = torch.cat((leaf_init_hidden[0].transpose(0, 1), leaf_decoder_hidden), dim=1)
        lex_decoder_hidden = torch.cat((lex_init_hidden[0].transpose(0, 1), lex_decoder_hidden), dim=1)
        '''

        ans_nt = self.constituent(Variable(trg_seqs[0]).cuda())
        ans_par_nt = self.constituent(Variable(parent_seqs[0][0]).cuda())
        ans_par2_nt = self.constituent(Variable(parent_seqs[1][0]).cuda())
        ans_par_lex = self.lexicon(Variable(parent_seqs[0][1]).cuda())
        ans_par2_lex = self.lexicon(Variable(parent_seqs[1][1]).cuda())
        ans_leaf_nt = self.constituent(Variable(leaf_seqs[0][0]).cuda())
        ans_leaf2_nt = self.constituent(Variable(leaf_seqs[1][0]).cuda())
        ans_leaf_lex = self.lexicon(Variable(leaf_seqs[0][1]).cuda())
        ans_leaf2_lex = self.lexicon(Variable(leaf_seqs[1][1]).cuda())
        ans_lex_nt = self.constituent(Variable(lex_seqs[0][0]).cuda())
        ans_lex2_nt = self.constituent(Variable(lex_seqs[1][0]).cuda())
        ans_lex_lex = self.lexicon(Variable(lex_seqs[0][1]).cuda())
        ans_lex2_lex = self.lexicon(Variable(lex_seqs[1][1]).cuda())
        ans_sib_nt = self.constituent(Variable(sibling_seqs[0][0]).cuda())
        ans_sib2_nt = self.constituent(Variable(sibling_seqs[1][0]).cuda())
        ans_sib_lex = self.lexicon(Variable(sibling_seqs[0][1]).cuda())
        ans_sib2_lex = self.lexicon(Variable(sibling_seqs[1][1]).cuda())
        
        _tree_input = torch.cat((ans_par_nt, ans_par_lex, 
                                 ans_par2_nt, ans_par2_lex,
                                 ans_sib_nt, ans_sib_lex,
                                 ans_sib2_nt, ans_sib2_lex,
                                 ans_leaf_nt, ans_leaf_lex,
                                 ans_leaf2_nt, ans_leaf2_lex,
                                 ans_lex_nt, ans_lex_lex,
                                 ans_lex2_nt, ans_lex2_lex,
                                ), dim=2)
        tree_input = F.tanh(self.tree(_tree_input))
        lex_tree_input = F.tanh(self.lex_tree(_tree_input))
        leaf_tree_input = F.tanh(self.leaf_tree(_tree_input))
        ans_embed = torch.cat((ans_nt, tree_input), dim=2)
        trg_l = max(trg_lengths)
        decoder_outputs = Variable(torch.FloatTensor(batch_size, trg_l, self.hidden_size * 3).cuda())
        batch_ind = torch.arange(batch_size).long().cuda()
        for step in range(trg_l):
            context = self.attention(decoder_hidden, src_hidden, src_hidden.size(1), src_lengths) 
            decoder_input = torch.cat((ans_embed[:, step, :].unsqueeze(1), context), dim=2)
            leaf_select = torch.LongTensor([x for x in range(batch_size) if step < len(leaf_indices[x]) and leaf_indices[x][step] == 1])
            lex_select = torch.LongTensor([x for x in range(batch_size) if step < len(lex_indices[x]) and lex_indices[x][step] == 1])
            decoder_input = torch.cat((decoder_input, leaf_decoder_hidden[0].transpose(0, 1), lex_decoder_hidden[0].transpose(0, 1)), dim=2)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs[:, step, :self.hidden_size] = decoder_output
            decoder_outputs[:, step, self.hidden_size:self.hidden_size * 2] = leaf_decoder_hidden[0]
            decoder_outputs[:, step, self.hidden_size * 2:self.hidden_size * 3] = lex_decoder_hidden[0]

            if lex_select.size():
                lex_input = torch.cat((self.lexicon(Variable(trg_seqs[1][lex_select][:, step].cuda())),
                                       self.constituent(Variable(trg_seqs[0][lex_select][:, step].cuda()))
                                      ), dim=1).unsqueeze(1)
                src_select_lengths = [src_lengths[l] for l in lex_select.tolist()]
                lex_select = lex_select.cuda()
                lex_select_hidden = (lex_decoder_hidden[0][:, lex_select, :], lex_decoder_hidden[1][:, lex_select, :])
                context = self.attention(lex_select_hidden, src_hidden[lex_select], src_hidden.size(1), src_select_lengths, 'lex_') 
                lex_input = torch.cat((lex_input, context, lex_tree_input[lex_select][:, step, :].unsqueeze(1)), dim=2)
                _, _decoder_hidden = self.lex_decoder(lex_input, lex_select_hidden)
                lex_decoder_hidden[0][:, lex_select, :] = _decoder_hidden[0]
                lex_decoder_hidden[1][:, lex_select, :] = _decoder_hidden[1]
            if leaf_select.size():
                leaf_input = self.lexicon(Variable(trg_seqs[1][leaf_select].cuda()))
                leaf_input = torch.cat((self.lexicon(Variable(trg_seqs[1][leaf_select][:, step].cuda())),
                                        self.constituent(Variable(trg_seqs[0][leaf_select][:, step].cuda()))
                                       ), dim=1).unsqueeze(1)
                src_select_lengths = [src_lengths[l] for l in leaf_select.tolist()]
                leaf_select = leaf_select.cuda()
                leaf_select_hidden = (leaf_decoder_hidden[0][:, leaf_select, :], leaf_decoder_hidden[1][:, leaf_select, :])
                context = self.attention(leaf_select_hidden, src_hidden[leaf_select], src_hidden.size(1), src_select_lengths, 'leaf_') 
                leaf_input = torch.cat((leaf_input, context, leaf_tree_input[leaf_select][:, step, :].unsqueeze(1)), dim=2)
                _, _decoder_hidden = self.leaf_decoder(leaf_input, (leaf_decoder_hidden[0][:, leaf_select, :], leaf_decoder_hidden[1][:, leaf_select, :]))
                leaf_decoder_hidden[0][:, leaf_select, :] = _decoder_hidden[0]
                leaf_decoder_hidden[1][:, leaf_select, :] = _decoder_hidden[1]

        return decoder_outputs

    def masked_loss(self, logits, target, lengths, mask):
        batch_size = logits.size(0)
        max_len = lengths.data.max()
        logits_flat = logits.view(-1, logits.size(-1))
        log_probs_flat = F.log_softmax(logits_flat, dim=1)
        target_flat = target.contiguous().view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*target.size())
        l_mask  = torch.arange(max_len).long().cuda().expand(batch_size, max_len) < lengths.data.expand(max_len, batch_size).transpose(0, 1)
        _mask = Variable(l_mask * mask)
        losses = losses * _mask.float()
        loss = losses.sum() / _mask.float().sum()
        return loss

    def loss(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, word_mask, rule_mask):
        decoder_outputs = self.forward(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices)
        tags = self.constituent(Variable(trg_seqs[2]).cuda())
        words = self.lex_dist(self.lex_out(torch.cat((decoder_outputs[:, :, self.hidden_size:], tags), dim=2)))
        rules = self.rule_dist(decoder_outputs[:, :, :self.hidden_size])
        
        word_loss = self.masked_loss(words, Variable(trg_seqs[1]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), word_mask.cuda().byte())
        rule_loss = self.masked_loss(rules, Variable(rule_seqs).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), rule_mask.cuda().byte())

        nts = self.nt_dist(self.nt_out(decoder_outputs[:, :, :self.hidden_size]))
        nt_loss = self.masked_loss(nts, Variable(trg_seqs[2]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), word_mask.cuda().byte())

        return word_loss, nt_loss, rule_loss

    def generate(self, src_seqs, src_lengths, indices, max_len, beam_size, top_k):
        src_embed = self.lexicon(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 
        leaf_decoder_hidden = self.init_hidden(src_last_hidden, 'leaf_') 
        lex_decoder_hidden = self.init_hidden(src_last_hidden, 'lex_')    
        '''
        leaf_decoder_hidden = leaf_init_hidden[0].transpose(0, 1)
        lex_decoder_hidden = lex_init_hidden[0].transpose(0, 1)
        '''

        batch_size = src_embed.size(0)
        assert batch_size == 1
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        context = self.attention(decoder_hidden, src_hidden, src_hidden.size(1), src_lengths) 
        decoder_input = torch.cat((Variable(torch.zeros(batch_size, 1, self.decoder_input_size - self.hidden_size * 2)).cuda(), context), dim=2)
        decoder_input = torch.cat((decoder_input, leaf_decoder_hidden[0], lex_decoder_hidden[0]), dim=2)
        decoder_input[:, :, :self.nt_input_size] = self.constituent(Variable(torch.cuda.LongTensor([self.ROOT])))
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0], lex_decoder_hidden[0]), dim=2)
        rule_logits = self.rule_dist(decoder_output[:, :, :self.hidden_size])

        word_beam = torch.zeros(beam_size, max_len).long().cuda()
        rule_beam = torch.zeros(beam_size, max_len).long().cuda()
        nt_beam = torch.zeros(beam_size, max_len).long().cuda()
        word_count = torch.zeros(beam_size, top_k).long().cuda()
        rule_count = torch.zeros(beam_size, top_k).long().cuda()
        nt_count = torch.zeros(beam_size, top_k).long().cuda()
        
        self.nt_cand_num = 1
        nt_logits = self.nt_dist(self.nt_out(decoder_output[:, :, :self.hidden_size]))
        nt_logprobs, nt_argtop = torch.topk(F.log_softmax(nt_logits.squeeze(1), dim=1), self.nt_cand_num, dim=1)
        rule_logits = rule_logits.expand(self.nt_cand_num, 1, self.rule_vocab_size).clone()
        rule_select = torch.cuda.ByteTensor(self.nt_cand_num, 1, self.rule_vocab_size).fill_(0)
        for y in range(self.nt_cand_num):
            rule_select[y, 0, :] = self.rules_by_nt[nt_argtop.data[0, y]]
        rule_logits[~rule_select] = -np.inf
        rule_logprobs, rule_argtop = torch.topk(F.log_softmax(rule_logits.squeeze(1), dim=1), beam_size, dim=1)
        nt_rule_logprobs = rule_logprobs + nt_logprobs.expand(beam_size, self.nt_cand_num).transpose(0, 1)

        tag_embed = self.constituent(nt_argtop)
        word_logits = self.lex_dist(self.lex_out(torch.cat((decoder_output[:, :, self.hidden_size:], tag_embed), dim=2)))
        word_logprobs, word_argtop = torch.topk(F.log_softmax(word_logits.squeeze(1), dim=1), beam_size, dim=1)
        total_logprobs = word_logprobs.expand(beam_size * self.nt_cand_num, beam_size).transpose(0, 1) + \
                         nt_rule_logprobs.view(-1).expand(beam_size, beam_size * self.nt_cand_num)
        logprobs, argtop = torch.topk(total_logprobs.view(-1), beam_size, dim=0)
        word_beam[:, 0] = word_argtop.squeeze(0).data[argtop.data / (beam_size * self.nt_cand_num)]
        word_beam_probs = word_logprobs.squeeze(0).data[argtop.data / (beam_size * self.nt_cand_num)]
        nt_rule_x = (argtop.data % (self.nt_cand_num * beam_size)) / beam_size
        nt_rule_y = (argtop.data % (self.nt_cand_num * beam_size)) % beam_size
        rule_beam[:, 0] = rule_argtop.data[nt_rule_x, nt_rule_y]
        rule_beam_probs = rule_logprobs.data[nt_rule_x, nt_rule_y]
        nt_beam[:, 0] = nt_argtop.squeeze(0).data[nt_rule_x]
        nt_beam_probs = nt_logprobs.squeeze(0).data[nt_rule_x]
        word_count.fill_(1)
        rule_count.fill_(1)
        nt_count.fill_(1)
        self.nt_cand_num = 2

        hidden = (decoder_hidden[0].clone(), decoder_hidden[1].clone())
        decoder_hidden = (decoder_hidden[0].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1),
                          decoder_hidden[1].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1))
        leaf_decoder_hidden = (leaf_decoder_hidden[0].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1),
                               leaf_decoder_hidden[1].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1))
        lex_decoder_hidden = (lex_decoder_hidden[0].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1),
                              lex_decoder_hidden[1].expand(beam_size, 1, self.hidden_size).contiguous().transpose(0, 1))
        src_hidden = src_hidden.expand(beam_size, src_hidden.size(1), self.hidden_size)

        current = np.array([None for x in range(beam_size)])
        leaves = torch.cuda.LongTensor([[[0, 0], [0, 0]] for x in range(beam_size)])
        lexicons = torch.cuda.LongTensor([[[0, 0], [0, 0]] for x in range(beam_size)])
        final = np.array([None for x in range(beam_size)])
        for y in range(beam_size):
            rule = self.rules_by_id[rule_beam[y, 0]]
            if self.lex_level == 0:
                current[y] = Node('{}__{}__{}'.format('ROOT', word_beam[y, 0], rule))
            else:
                tag = self.nt_by_id[nt_beam[y, 0]]
                current[y] = Node('{}__{}__{} [{}]'.format('ROOT', word_beam[y, 0], rule, rule.split().index(tag)))

        def inheritance(rule):
            return literal_eval(rule[rule.find('['):rule.find(']')+1])

        for t in range(max_len-1):
            ans_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_par_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_par2_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_par_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_par2_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_leaf_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_leaf2_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_leaf_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_leaf2_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_lex_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_lex2_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_lex_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_lex2_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_sib_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_sib2_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_sib_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_sib2_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
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
                    ans_par_nt[x] = self.nt_dictionary[par_nt]
                    ans_par_lex[x] = int(par_lex)
                    if current[x].parent is not None:
                        par2_nt, par2_lex, __ = current[x].parent.name.split('__')
                        ans_par2_nt[x] = self.nt_dictionary[par2_nt]
                        ans_par2_lex[x] = int(par2_lex)
                        pos = current[x].parent.children.index(current[x])
                        if pos > 0:
                            sib_nt, sib_lex, __ = current[x].parent.children[pos-1].name.split('__')
                            ans_sib_nt[x] = self.nt_dictionary[sib_nt]
                            ans_sib_lex[x] = int(sib_lex)
                        if current[x].parent.parent is not None:
                            pos = current[x].parent.parent.children.index(current[x].parent)
                            if pos > 0:
                                sib_nt, sib_lex, __ = current[x].parent.parent.children[pos-1].name.split('__')
                                ans_sib2_nt[x] = self.nt_dictionary[sib_nt]
                                ans_sib2_lex[x] = int(sib_lex)
                    ans_leaf_nt[x] = leaves[x][0][0]
                    ans_leaf_lex[x] = leaves[x][0][1]
                    ans_leaf2_nt[x] = leaves[x][1][0]
                    ans_leaf2_lex[x] = leaves[x][1][1]
                    ans_lex_nt[x] = lexicons[x][0][0]
                    ans_lex_lex[x] = lexicons[x][0][1]
                    ans_lex2_nt[x] = lexicons[x][1][0]
                    ans_lex2_lex[x] = lexicons[x][1][1]
            ans_nt = self.constituent(ans_nt)
            ans_par_nt = self.constituent(ans_par_nt)
            ans_par2_nt = self.constituent(ans_par2_nt)
            ans_par_lex = self.lexicon(ans_par_lex)
            ans_par2_lex = self.lexicon(ans_par2_lex)
            ans_leaf_nt = self.constituent(ans_leaf_nt)
            ans_leaf2_nt = self.constituent(ans_leaf2_nt)
            ans_leaf_lex = self.lexicon(ans_leaf_lex)
            ans_leaf2_lex = self.lexicon(ans_leaf2_lex)
            ans_lex_nt = self.constituent(ans_lex_nt)
            ans_lex2_nt = self.constituent(ans_lex2_nt)
            ans_lex_lex = self.lexicon(ans_lex_lex)
            ans_lex2_lex = self.lexicon(ans_lex2_lex)
            ans_sib_nt = self.constituent(ans_sib_nt)
            ans_sib2_nt = self.constituent(ans_sib2_nt)
            ans_sib_lex = self.lexicon(ans_sib_lex)
            ans_sib2_lex = self.lexicon(ans_sib2_lex)
            
            _tree_input = torch.cat((ans_par_nt, ans_par_lex, 
                                     ans_par2_nt, ans_par2_lex,
                                     ans_sib_nt, ans_sib_lex,
                                     ans_sib2_nt, ans_sib2_lex,
                                     ans_leaf_nt, ans_leaf_lex,
                                     ans_leaf2_nt, ans_leaf2_lex,
                                     ans_lex_nt, ans_lex_lex,
                                     ans_lex2_nt, ans_lex2_lex
                                    ), dim=1)
            tree_input = F.tanh(self.tree(_tree_input))
            leaf_tree_input = F.tanh(self.leaf_tree(_tree_input))
            lex_tree_input = F.tanh(self.lex_tree(_tree_input))
            ans_embed = torch.cat((ans_nt, tree_input), dim=1)
            context = self.attention(decoder_hidden, src_hidden, src_hidden.size(1), src_lengths) 
            decoder_input = torch.cat((ans_embed.unsqueeze(1), context), dim=2)
            decoder_input = torch.cat((decoder_input, leaf_decoder_hidden[0].transpose(0, 1), lex_decoder_hidden[0].transpose(0, 1)), dim=2)
               
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = torch.cat((decoder_output, leaf_decoder_hidden[0].transpose(0, 1), lex_decoder_hidden[0].transpose(0, 1)), dim=2)
            dup_mask = ~(word_mask | rule_mask)

            nt_logits = self.nt_dist(self.nt_out(decoder_output[:, :, :self.hidden_size]))
            nt_logprobs, nt_argtop = torch.topk(F.log_softmax(nt_logits.squeeze(1), dim=1), self.nt_cand_num, dim=1)

            rule_logits = self.rule_dist(decoder_output[:, :, :self.hidden_size])
            rule_logits = rule_logits.expand(beam_size, self.nt_cand_num, self.rule_vocab_size).clone()
            rule_select = torch.cuda.ByteTensor(beam_size, self.nt_cand_num, self.rule_vocab_size).fill_(0)
            dup_mask = torch.cuda.ByteTensor(beam_size * self.nt_cand_num, top_k).fill_(False)
            for x in range(beam_size):
                for y in range(self.nt_cand_num):
                    rule_select[x, y, :] = self.rules_by_nt[nt_argtop.data[x, y]]
                if rule_mask[x, 0] == 0:
                    if word_mask[x, 0] == 0:
                        dup_mask[x*self.nt_cand_num:(x+1)*self.nt_cand_num, :] = True
                        dup_mask[x*self.nt_cand_num, 0] = False
                    else:
                        nt_argtop[x] = self.nt_dictionary[curr_nt[x]]
                        dup_mask[x*self.nt_cand_num+1:(x+1)*self.nt_cand_num, :] = True

            decoder_output = decoder_output.expand(beam_size, self.nt_cand_num, self.hidden_size * 3).contiguous().view(beam_size * self.nt_cand_num, -1)
            tag_embed = self.constituent(nt_argtop.view(-1))
            word_logits = self.lex_dist(self.lex_out(torch.cat((decoder_output[:, self.hidden_size:], tag_embed), dim=1)))
            word_logprobs, word_argtop = torch.topk(F.log_softmax(word_logits.squeeze(1), dim=1), top_k, dim=1)
            
            rule_logits[~rule_select] = -np.inf
            rule_logprobs, rule_argtop = torch.topk(F.log_softmax(rule_logits.view(beam_size * self.nt_cand_num, -1), dim=1), top_k, dim=1)
            nt_rule_logprobs = rule_logprobs + nt_logprobs.view(-1).expand(top_k, beam_size * self.nt_cand_num).transpose(0, 1)
            rule_mask = rule_mask.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1)
            word_mask = word_mask.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1)
            '''
            dup_mask = dup_mask.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1)
            dup_mask[~rule_mask] = 0
            dup_mask[torch.arange(beam_size).long().cuda() * self.nt_cand_num, torch.zeros(beam_size).long().cuda()] = 0
            '''
            rule_beam_logprobs = rule_beam_probs.expand(top_k, self.nt_cand_num, beam_size).transpose(0, 2).contiguous().view(beam_size * self.nt_cand_num, -1) + \
                                 nt_rule_logprobs.data * rule_mask.float()
            rule_beam_logprobs[dup_mask] = -np.inf
            #word_beam_logprobs = word_beam_logprobs.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1)
            word_beam_logprobs = (word_beam_probs).expand(top_k, self.nt_cand_num, beam_size).transpose(0, 2).contiguous().view(beam_size * self.nt_cand_num, -1) + word_logprobs.data * word_mask.float()
            word_beam_logprobs[dup_mask] = -np.inf
            #word_logprobs = word_logprobs.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1)
            #word_argtop = word_argtop.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1)
            total_logprobs = (word_beam_logprobs - 0) / word_count.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1).float() + \
                             (rule_beam_logprobs - 0) / rule_count.expand(self.nt_cand_num, beam_size, top_k).transpose(0, 1).contiguous().view(beam_size * self.nt_cand_num, -1).float()
            #total_logprobs = word_beam_logprobs / np.sqrt(t + 1) + rule_beam_logprobs / (t + 1)
            total_logprobs[total_logprobs != total_logprobs] = -np.inf
            best_probs, best_args = total_logprobs.view(-1).topk(beam_size)
        
            decoder_hidden = (decoder_hidden[0].view(1, batch_size, beam_size, -1),
                              decoder_hidden[1].view(1, batch_size, beam_size, -1))
            leaf_decoder_hidden = (leaf_decoder_hidden[0].view(1, batch_size, beam_size, -1),
                                   leaf_decoder_hidden[1].view(1, batch_size, beam_size, -1))
            lex_decoder_hidden = (lex_decoder_hidden[0].view(1, batch_size, beam_size, -1),
                                  lex_decoder_hidden[1].view(1, batch_size, beam_size, -1))

            last = (best_args / top_k)
            curr = (best_args % top_k)
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
            '''
            word_beam[:, 0] = word_argtop.squeeze(0).data[last, curr]
            word_beam_probs = word_logprobs.squeeze(0).data[argtop.data / (beam_size * self.nt_cand_num)]
            nt_rule_x = (argtop.data % (self.nt_cand_num * beam_size)) / beam_size
            nt_rule_y = (argtop.data % (self.nt_cand_num * beam_size)) % beam_size
            rule_beam[:, 0] = rule_argtop.data[nt_rule_x, nt_rule_y]
            rule_beam_probs = rule_logprobs.data[nt_rule_x, nt_rule_y]
            nt_beam[:, 0] = nt_argtop.squeeze(0).data[nt_rule_x]
            nt_beam_probs = nt_logprobs.squeeze(0).data[nt_rule_x]
            '''
            nt_beam = nt_beam[(last / self.nt_cand_num)]
            nt_beam[:, t+1] = nt_argtop[last / self.nt_cand_num, last % self.nt_cand_num].data
            last /= self.nt_cand_num
            tree_input = tree_input[last]
            leaf_tree_input = leaf_tree_input[last]
            lex_tree_input = lex_tree_input[last]

            decoder_hidden[0][:, 0, :, :] = decoder_hidden[0][:, 0, :, :][:, last, :]
            decoder_hidden[1][:, 0, :, :] = decoder_hidden[1][:, 0, :, :][:, last, :]
            decoder_hidden = (decoder_hidden[0].view(1, batch_size * beam_size, -1),
                              decoder_hidden[1].view(1, batch_size * beam_size, -1))

            leaf_decoder_hidden[0][:, 0, :, :].data = leaf_decoder_hidden[0][:, 0, :, :][:, last, :].data
            leaf_decoder_hidden[1][:, 0, :, :].data = leaf_decoder_hidden[1][:, 0, :, :][:, last, :].data
            lex_decoder_hidden[0][:, 0, :, :].data = lex_decoder_hidden[0][:, 0, :, :][:, last, :].data
            lex_decoder_hidden[1][:, 0, :, :].data = lex_decoder_hidden[1][:, 0, :, :][:, last, :].data
            leaf_decoder_hidden = (leaf_decoder_hidden[0].view(1, batch_size * beam_size, -1),
                                   leaf_decoder_hidden[1].view(1, batch_size * beam_size, -1))
            lex_decoder_hidden = (lex_decoder_hidden[0].view(1, batch_size * beam_size, -1),
                                  lex_decoder_hidden[1].view(1, batch_size * beam_size, -1))

            #beam_eos = beam_eos | (beam[:, :, t+1] == self.eou).data
            for x in range(beam_size):
                current[x] = deepcopy(current[x])
                if current[x] is None:
                    continue
                #nt, word, rule = current[x].name.split('__')
                if self.preterminal[curr_nt[x]]:
                    _, word, rule = current[x].name.split('__')
                    if len(current[x].children) not in inheritance(rule):
                        word = word_beam[x, t+1]
                    ch = Node('{}__{}__ []'.format(curr_nt[x], word), parent=current[x])
                    leaves[x][1] = leaves[x][0]
                    leaves[x][0][0] = self.nt_dictionary[curr_nt[x]]
                    leaves[x][0][1] = int(word)

                    l_decoder_input = torch.cat((self.lexicon(Variable(torch.cuda.LongTensor([int(word)]))), 
                                                   self.constituent(Variable(torch.cuda.LongTensor([self.nt_dictionary[curr_nt[x]]])))
                                                  ), dim=1).unsqueeze(1)
                    lex_select_hidden = (lex_decoder_hidden[0][:, x, :].unsqueeze(1), lex_decoder_hidden[1][:, x, :].unsqueeze(1))
                    context = self.attention(lex_select_hidden, src_hidden[0].unsqueeze(0), src_hidden.size(1), src_lengths, 'lex_') 
                    lex_decoder_input = torch.cat((l_decoder_input, context, lex_tree_input[x].unsqueeze(0).unsqueeze(0)), dim=2)
                    _, _lex_decoder_hidden = self.lex_decoder(lex_decoder_input, lex_select_hidden)
                    lex_decoder_hidden[0][:, x, :].data = _lex_decoder_hidden[0][0].data
                    lex_decoder_hidden[1][:, x, :].data = _lex_decoder_hidden[1][0].data

                    leaf_select_hidden = (leaf_decoder_hidden[0][:, x, :].unsqueeze(1), leaf_decoder_hidden[1][:, x, :].unsqueeze(1))
                    context = self.attention(leaf_select_hidden, src_hidden[0].unsqueeze(0), src_hidden.size(1), src_lengths, 'leaf_') 
                    leaf_decoder_input = torch.cat((l_decoder_input, context, leaf_tree_input[x].unsqueeze(0).unsqueeze(0)), dim=2)
                    _, _leaf_decoder_hidden = self.leaf_decoder(lex_decoder_input, leaf_select_hidden)
                    leaf_decoder_hidden[0][:, x, :].data = _leaf_decoder_hidden[0][0].data
                    leaf_decoder_hidden[1][:, x, :].data = _leaf_decoder_hidden[1][0].data
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
                else:
                    _, word, par_rule = current[x].name.split('__')
                    rule = self.rules_by_id[rule_beam[x, t+1]]
                    if len(current[x].children) not in inheritance(par_rule):
                        word = word_beam[x, t+1]
                        lexicons[x][1] = lexicons[x][0]
                        lexicons[x][0][0] = self.nt_dictionary[curr_nt[x]]
                        lexicons[x][0][1] = int(word)
                    tag = self.nt_by_id[nt_beam[x, t+1]]
                    current[x] = Node('{}__{}__{} [{}]'.format(curr_nt[x], word, rule, rule.split().index(tag)), parent=current[x])
                    if self.lex_level == 2:
                        lex_decoder_input = torch.cat((self.lexicon(Variable(torch.cuda.LongTensor([int(word)]))), 
                                                       self.constituent(Variable(torch.cuda.LongTensor([self.nt_dictionary[curr_nt[x]]])))
                                                      ), dim=1).unsqueeze(1)
                        lex_select_hidden = (lex_decoder_hidden[0][:, x, :].unsqueeze(1), lex_decoder_hidden[1][:, x, :].unsqueeze(1))
                        context = self.attention(lex_select_hidden, src_hidden[0].unsqueeze(0), src_hidden.size(1), src_lengths, 'lex_') 
                        lex_decoder_input = torch.cat((lex_decoder_input, context, lex_tree_input[x].unsqueeze(0).unsqueeze(0)), dim=2)
                        _, _lex_decoder_hidden = self.lex_decoder(lex_decoder_input, lex_select_hidden)
                        lex_decoder_hidden[0][:, x, :].data = _lex_decoder_hidden[0][0].data
                        lex_decoder_hidden[1][:, x, :].data = _lex_decoder_hidden[1][0].data
            if (current == None).all():
                break

        '''
        best, best_arg = beam_probs.max(1)
        generations = beam[torch.arange(_batch_size).long().cuda(), best_arg.data].data.cpu()
        return generations, best
        '''
        return final[0]


if __name__ == '__main__':
    train()
