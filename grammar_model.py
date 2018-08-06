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
                 lex_vectors, nt_vectors,
                 lex_dictionary, nt_dictionary, rule_dictionary):
        super(LexicalizedGrammarDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.lex_input_size = lex_input_size
        self.nt_input_size = nt_input_size
        self.rule_input_size = rule_input_size
        self.lex_vocab_size = lex_vocab_size
        self.nt_vocab_size = nt_vocab_size
        self.rule_vocab_size = rule_vocab_size
        self.lexicon = Embedding(lex_vocab_size, lex_input_size, lex_vectors, trainable=False)
        self.constituent = Embedding(nt_vocab_size, nt_input_size, nt_vectors, trainable=True)
        self.encoder = nn.LSTM(lex_input_size, hidden_size)
        self.decoder_input_size = lex_input_size + nt_input_size * 2
        self.decoder = nn.LSTM(self.decoder_input_size + context_size, hidden_size, batch_first=True)
        self.lex_out = nn.Linear(hidden_size, lex_input_size)
        #self.rule_out = nn.Linear(hidden_size, rule_input_size)
        self.lex_dist = nn.Linear(lex_input_size, lex_vocab_size)
        self.rule_dist = nn.Linear(hidden_size, rule_vocab_size)
        self.hidden_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.hidden_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.cell_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.lex_dictionary = lex_dictionary
        self.nt_dictionary = nt_dictionary
        self.rule_dictionary = rule_dictionary
        self.eou = lex_dictionary['__eou__']
        self.lex_dist.weight = self.lexicon.weight

        self.a_key = nn.Linear(hidden_size, 100)
        self.q_key = nn.Linear(hidden_size, 100)
        self.q_value = nn.Linear(hidden_size, 300)

        self.init_forget_bias(self.encoder, 0)
        self.init_forget_bias(self.decoder, 1)

        '''
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
        '''

    def init_rules(self):
        self.ROOT = self.nt_dictionary['ROOT']
        self.rules_by_id = defaultdict(str)
        for k, v in self.rule_dictionary.items():
            self.rules_by_id[v] = k[6:]

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

    def attention(self, decoder_hidden, src_hidden, src_max_len, src_lengths):
        a_key = F.tanh(self.a_key(decoder_hidden[0].squeeze(0)))
        q_key = F.tanh(self.q_key(src_hidden))
        q_value = self.q_value(src_hidden)
        q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
        q_mask  = torch.arange(src_max_len).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(src_max_len, 1).transpose(0, 1)
        q_energy[~q_mask] = -np.inf
        q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
        q_context = torch.bmm(q_weights, q_value).squeeze(1)
        context = q_context.unsqueeze(1)

        return context

    def forward(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, parent_seqs, rule_seqs):
        batch_size = src_seqs.size(0)

        src_embed = self.lexicon(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 

        ans_nt = self.constituent(Variable(trg_seqs[0]).cuda())
        ans_par_nt = self.constituent(Variable(parent_seqs[0][0]).cuda())
        ans_par_lex = self.lexicon(Variable(parent_seqs[0][1]).cuda())
        ans_embed = torch.cat((ans_nt, ans_par_nt, ans_par_lex), dim=2)

        trg_l = max(trg_lengths)
        decoder_outputs = Variable(torch.FloatTensor(batch_size, trg_l, self.hidden_size).cuda())
        for step in range(trg_l):
            context = self.attention(decoder_hidden, src_hidden, src_hidden.size(1), src_lengths) 
            decoder_input = torch.cat((ans_embed[:, step, :].unsqueeze(1), context), dim=2)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs[:, step, :] = decoder_output

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

    def loss(self, src_seqs, src_lengths, indices, trg_seqs, trg_lengths, parent_seqs, rule_seqs, word_mask, rule_mask):
        decoder_outputs = self.forward(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, parent_seqs, rule_seqs)
        words = self.lex_dist(self.lex_out(decoder_outputs))
        rules = self.rule_dist(decoder_outputs)
        
        word_loss = self.masked_loss(words, Variable(trg_seqs[1]).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), word_mask.cuda().byte())
        rule_loss = self.masked_loss(rules, Variable(rule_seqs).cuda(), Variable(torch.LongTensor(trg_lengths)).cuda(), rule_mask.cuda().byte())

        return word_loss, rule_loss

    def generate(self, src_seqs, src_lengths, indices, max_len, beam_size, top_k):
        src_embed = self.lexicon(Variable(src_seqs).cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 

        batch_size = src_embed.size(0)
        assert batch_size == 1
        eos_filler = Variable(torch.zeros(beam_size).long().cuda().fill_(self.eou))
        context = self.attention(decoder_hidden, src_hidden, src_hidden.size(1), src_lengths) 
        decoder_input = torch.cat((Variable(torch.zeros(batch_size, 1, self.decoder_input_size)).cuda(), context), dim=2)
        decoder_input[:, :, :self.nt_input_size] = self.constituent(Variable(torch.cuda.LongTensor([self.ROOT])))
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        word_logits = self.lex_dist(self.lex_out(decoder_output))
        rule_logits = self.rule_dist(decoder_output)

        word_logprobs, word_argtop = torch.topk(F.log_softmax(word_logits.squeeze(1), dim=1), beam_size, dim=1)
        rule_logprobs, rule_argtop = torch.topk(F.log_softmax(rule_logits.squeeze(1), dim=1), beam_size, dim=1)
        total_logprobs = word_logprobs.expand(beam_size, beam_size).transpose(0, 1) + rule_logprobs.expand(beam_size, beam_size)
        logprobs, argtop = torch.topk(total_logprobs.view(-1), beam_size, dim=0)
        word_beam = torch.zeros(beam_size, max_len).long().cuda()
        rule_beam = torch.zeros(beam_size, max_len).long().cuda()
        word_count = torch.ones(beam_size).long().cuda()
        rule_count = torch.ones(beam_size).long().cuda()
        word_beam[:, 0] = word_argtop.squeeze(0).data[argtop.data / beam_size]
        rule_beam[:, 0] = rule_argtop.squeeze(0).data[argtop.data % beam_size]
        word_beam_probs = word_logprobs.squeeze(0).data[argtop.data / beam_size]
        rule_beam_probs = rule_logprobs.squeeze(0).data[argtop.data % beam_size]
        hidden = (decoder_hidden[0].clone(), decoder_hidden[1].clone())
        decoder_hidden = (decoder_hidden[0].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1),
                          decoder_hidden[1].unsqueeze(2).expand(1, batch_size, beam_size, self.hidden_size).contiguous().view(1, batch_size * beam_size, -1))
        src_hidden = src_hidden.expand(beam_size, src_hidden.size(1), self.hidden_size)

        current = np.array([None for x in range(beam_size)])
        final = np.array([None for x in range(beam_size)])
        for y in range(beam_size):
            rule = self.rules_by_id[rule_beam[y, 0]]
            current[y] = Node('{}__{}__{}'.format('ROOT', word_beam[y, 0], rule))

        def inheritance(rule):
            return literal_eval(rule[rule.find('['):rule.find(']')+1])

        for t in range(max_len-1):
            ans_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_par_nt = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            ans_par_lex = Variable(torch.cuda.LongTensor(beam_size)).fill_(0)
            word_mask = torch.cuda.ByteTensor(beam_size)
            word_mask.fill_(True)
            rule_mask = torch.cuda.ByteTensor(beam_size)
            rule_mask.fill_(True)
            curr_nt = np.array(['' for x in range(beam_size)], dtype=object)

            for x in range(beam_size):
                if current[x] is None:
                    rule_mask[x] = False
                    word_mask[x] = False
                else:
                    par_nt, par_lex, rule = current[x].name.split('__')
                    curr_nt[x] = rule.split()[len(current[x].children)]
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
            ans_nt = self.constituent(ans_nt)
            ans_par_nt = self.constituent(ans_par_nt)
            ans_par_lex = self.lexicon(ans_par_lex)
            ans_embed = torch.cat((ans_nt, ans_par_nt, ans_par_lex), dim=1)
            context = self.attention(decoder_hidden, src_hidden, src_hidden.size(1), src_lengths) 
            decoder_input = torch.cat((ans_embed.unsqueeze(1), context), dim=2)
               
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            word_logits = self.lex_dist(self.lex_out(decoder_output))
            rule_logits = self.rule_dist(decoder_output)

            word_logprobs, word_argtop = torch.topk(F.log_softmax(word_logits.squeeze(1), dim=1), beam_size, dim=1)
            rule_logprobs, rule_argtop = torch.topk(F.log_softmax(rule_logits.squeeze(1), dim=1), beam_size, dim=1)
            total_logprobs = word_logprobs.expand(beam_size, beam_size).transpose(0, 1) + rule_logprobs.expand(beam_size, beam_size)
            logprobs, argtop = torch.topk(total_logprobs.view(-1), beam_size, dim=0)
            word_beam_logprobs = ((word_beam_probs * word_count.float()).expand(top_k, beam_size).transpose(0, 1) + word_logprobs.data * word_mask.float()) / (word_count.expand(top_k, beam_size).float() + word_mask.expand(top_k, beam_size).float()).transpose(0, 1)
            rule_beam_logprobs = ((rule_beam_probs * rule_count.float()).expand(top_k, beam_size).transpose(0, 1) + rule_logprobs.data * rule_mask.float()) / (rule_count.expand(top_k, beam_size).float() + rule_mask.expand(top_k, beam_size).float()).transpose(0, 1)
            #best_word_probs, best_word_args = word_beam_logprobs.view(-1).topk(beam_size)
            #best_rule_probs, best_rule_args = rule_beam_logprobs.view(-1).topk(beam_size)
            #logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), top_k, dim=1)
            #best_probs, best_args = (beam_probs.view(-1).unsqueeze(1).expand(_batch_size * beam_size, top_k) + logprobs).view(_batch_size, beam_size, -1).view(_batch_size, -1).topk(beam_size)
            best_probs, best_args = (word_beam_logprobs + rule_beam_logprobs).view(-1).topk(beam_size)
            decoder_hidden = (decoder_hidden[0].view(1, batch_size, beam_size, -1),
                              decoder_hidden[1].view(1, batch_size, beam_size, -1))

            last = (best_args / top_k)
            curr = (best_args % top_k)
            current = current[last.tolist()]
            curr_nt = curr_nt[last.tolist()]
            final = final[last.tolist()]
            #beam[x, :, :] = beam[x][last, :]
            word_beam = word_beam[last]
            rule_beam = rule_beam[last]
            #beam_eos[x, :] = beam_eos[x][last.data]
            #beam_probs[x, :] = beam_probs[x][last.data]
            word_beam_probs = word_beam_logprobs[last, curr]
            rule_beam_probs = rule_beam_logprobs[last, curr]
            #beam[x, :, t+1] = argtop.view(_batch_size, beam_size, top_k)[x][last.data, curr.data] * Variable(~beam_eos[x]).long() + eos_filler * Variable(beam_eos[x]).long()
            word_beam[:, t+1] = word_argtop[curr, last].data
            rule_beam[:, t+1] = rule_argtop[curr, last].data
            '''
            mask = torch.cuda.ByteTensor(_batch_size, beam_size).fill_(0)
            mask[x] = ~beam_eos[x]
            beam_probs[mask] = (beam_probs[mask] * (t+1) + best_probs[mask]) / (t+2)
            '''
            decoder_hidden[0][:, 0, :, :] = decoder_hidden[0][:, 0, :, :][:, last, :]
            decoder_hidden[1][:, 0, :, :] = decoder_hidden[1][:, 0, :, :][:, last, :]

            #beam_eos = beam_eos | (beam[:, :, t+1] == self.eou).data
            decoder_hidden = (decoder_hidden[0].view(1, batch_size * beam_size, -1),
                              decoder_hidden[1].view(1, batch_size * beam_size, -1))
            #decoder_input = self.embed(beam[:, :, t+1].contiguous().view(-1)).unsqueeze(1)

            for x in range(beam_size):
                current[x] = deepcopy(current[x])
                if current[x] is None:
                    continue
                #nt, word, rule = current[x].name.split('__')
                if self.preterminal[curr_nt[x]]:
                    _, word, rule = current[x].name.split('__')
                    if len(current[x].children) not in inheritance(rule):
                        word = word_beam[x, t+1]
                    ch = Node('LEAF: {} {}'.format(curr_nt[x], word), parent=current[x])
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
                    current[x] = Node('{}__{}__{}'.format(curr_nt[x], word, rule), parent=current[x])

        '''
        best, best_arg = beam_probs.max(1)
        generations = beam[torch.arange(_batch_size).long().cuda(), best_arg.data].data.cpu()
        return generations, best
        '''
        return final[0]


if __name__ == '__main__':
    train()
