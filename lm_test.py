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
import pickle
from lm_data_loader import get_loader

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys


def main(config):
    dictionary = json.load(open('msr.lex.dictionary.json'))
    test_loader = get_loader('./data/msr.test.lex.dat'.format(config.data), dictionary, 1, shuffle=False)

    hidden_size = 512
    cenc_input_size = hidden_size * 2

    start_batch = 75000

    hred = torch.load('lm.msr.pt')
    hred.flatten_parameters()

    def print_response(responses, out=sys.stdout):
        if responses.is_leaf:
            out.write(id2word[int(responses.name.split('__')[1])])
            out.write(' ')
        else:
            for ch in responses.children:
                print_response(ch, out)
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
    hred.eval()
    output = open('test.txt', 'w')
    for _, batch in enumerate(test_loader):
        print(_)
        indices, trg_seqs, trg_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, word_mask, rule_mask, positions, ancestors, anc_lengths     = batch
        w_loss, nt_loss, r_loss = hred.loss(indices, trg_seqs, trg_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, word_mask, rule_mask, positions, ancestors, anc_lengths)
        loss = w_loss + nt_loss + r_loss
        output.write('{} {} {} {}\n'.format(loss.data[0], w_loss.data[0], r_loss.data[0], nt_loss.data[0]))
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vhred', type=bool, default=False)
    parser.add_argument('--use_saved', type=bool, default=False)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--print_every_n_batches', type=int, default=10)
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--data', type=str, default='persona')
    parser.add_argument('--attn', default=True, action='store_true')
    parser.add_argument('--unlex', default=False, action='store_true')
    config = parser.parse_args()
    main(config)
