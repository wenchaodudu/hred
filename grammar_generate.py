import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import json
import pdb
import time
import numpy as np
import argparse
from grammar_data_loader import get_loader

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys


def main(config):
    dictionary = json.load(open('./{}.parse.dictionary.json'.format(config.data)))
    vocab_size = len(dictionary)
    word_embedding_dim = 300
    print("Vocabulary size:", len(dictionary))

    test_loader = get_loader('./data/{}.test.src'.format(config.data), './data/{}.test.parse.trg'.format(config.data), dictionary, 1, shuffle=False)

    hidden_size = 512
    cenc_input_size = hidden_size * 2

    start_batch = 75000

    if config.path:
        hred = torch.load(config.path)
        hred.flatten_parameters()
    else:
        if config.vhred:
            hred = torch.load('vhred.pt')
            hred.flatten_parameters()
        else:
            hred = torch.load('hred-ad-2116.pt')
            hred.flatten_parameters()

    max_len = 200
    id2word = dict()
    for k, v in dictionary.items():
        id2word[v] = k

    def print_response(responses):
        for x in range(responses.size(0)):
            for y in range(max_len):
                sys.stdout.write(id2word[responses[x, y]])
                if responses[x, y] == dictionary['__eou__']:
                    sys.stdout.write('\n')
                    break
                else:
                    sys.stdout.write(' ')
            sys.stdout.write('\n')
    '''
    disc = torch.load('discriminator.pt')
    disc.flatten_parameters()
    qnet = torch.load('q_network_adv.pt')
    '''

    def word_embed_dist(trg_embed, trg_lengths, generations):
        gen_embed = hred.embedding(Variable(generations))
        gen_embed = gen_embed / gen_embed.norm(dim=2).unsqueeze(2)
        dots = torch.bmm(gen_embed, trg_embed.transpose(1, 2))
        dist = 0
        leng = []
        for x in range(generations.size(0)):
            g_len = generations[x][1:].min(0)[1][0] + 1
            leng.append(g_len)
            t_len = trg_lengths[x]
            d = dots[x, :g_len, :t_len]
            dist += d.sum() / (g_len * t_len)
        gen_embed = hred.embedding(Variable(generations))
        leng, perm_idx = torch.LongTensor(leng).sort(0, descending=True)
        gen_embed = gen_embed[perm_idx]
        packed_input = pack_padded_sequence(gen_embed, leng.cpu().numpy(), batch_first=True)
        gen_output = disc.u_encoder(packed_input)
        gen_output = gen_output[perm_idx.sort()[1]]
        return (dist / generations.size(0)).data.cpu(), np.mean(leng), gen_output

    dist = [0, 0, 0]
    lengths = [0, 0, 0]
    scores = [0, 0, 0]
    for _, batch in enumerate(test_loader):
        src_seqs, src_lengths, trg_seqs, trg_lengths, indices, slots = batch
        '''
        if _ % config.print_every_n_batches == 1:
            print(_)
            print(np.asarray(dist) / _)
            print(np.asarray(lengths) / _)
            print(np.asarray(scores) / _)
        _batch_size = len(turn_len)
        src_output = disc.encode_context(ctc_seqs, ctc_lengths, ctc_indices)
        trg_embed = hred.embedding(Variable(trg_seqs).cuda())
        trg_embed = trg_embed / trg_embed.norm(dim=2).unsqueeze(2)
        trg_embed = trg_embed[torch.from_numpy(np.argsort(trg_indices)).cuda()]
        trg_lengths = torch.cuda.LongTensor(trg_lengths)[torch.from_numpy(np.argsort(trg_indices)).cuda()]
        '''
        pdb.set_trace()
        responses, nll = hred.fsm_generate(src_seqs, src_lengths, indices, torch.LongTensor([dictionary[slot[0]] for slot in slots[0]]), max_len, 10, 10)
        print_response(responses)
        '''
        d, l, gen = word_embed_dist(trg_embed, trg_lengths, responses)
        dist[0] += d
        lengths[0] += l
        scores[0] += (nll.sum().data[0] + 5 * torch.log(disc.score_(src_output, gen)).sum().data[0]) / _batch_size
        print()
        responses, nll = hred.random_sample(src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len, max_len, 10, 10)
        d, l, gen = word_embed_dist(trg_embed, trg_lengths, responses)
        dist[1] += d
        lengths[1] += l
        scores[1] += (nll.sum().data[0] + 5 * torch.log(disc.score_(src_output, gen)).sum().data[0]) / _batch_size
        print_response(responses)
        print()
        responses, nll = hred.random_sample(src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len, max_len, 10, 10)
        loss, score, generations = hred.train_decoder(src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len, max_len, 10, 10, disc, qnet, 5)
        d, l, gen = word_embed_dist(trg_embed, trg_lengths, responses)
        dist[2] += d
        lengths[2] += l
        scores[2] += score.data[0]
        print_response(generations)
        print()
        responses, nll = hred.random_sample(src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len, max_len, 10, 10)
        '''
        if _ >= 20:
            break

    '''
    print(np.asarray(dist) / _)
    print(np.asarray(lengths) / _)
    print(np.asarray(scores) / _)
    '''

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vhred', type=bool, default=False)
    parser.add_argument('--use_saved', type=bool, default=False)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--print_every_n_batches', type=int, default=10)
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--data', type=str, default='persona')
    parser.add_argument('--attn', default=True, action='store_true')
    config = parser.parse_args()
    main(config)
