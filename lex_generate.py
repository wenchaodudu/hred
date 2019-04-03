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
from lex_data_loader import get_loader

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys


def main(config):
    lex_type = 'lex1' if config.type == 'lex0' else config.type
    dictionary = json.load(open('{}.{}.dictionary.json'.format(config.data, lex_type)))
    if config.data in ['persona', 'movie']:
        test_loader = get_loader('./data/{}.test.src'.format(config.data), 
                                 './data/{}.test.{}.dat'.format(config.data, lex_type), 
                                 './data/{}.test.trg'.format(config.data),
                                 './data/{}.test.psn'.format(config.data), 
                                 dictionary, 1, shuffle=False)
    else:
        test_loader = get_loader('./data/{}.test.lex3.src.dat'.format(config.data), 
                                 './data/{}.test.lex3.trg.dat'.format(config.data), 
                                 None, 
                                 dictionary, 1, shuffle=False)
    hidden_size = 512
    cenc_input_size = hidden_size * 2

    start_batch = 75000

    nt_word = np.load('{}.{}_nt_word_count.npy'.format(lex_type, config.data))
    nt_rule = np.load('{}.{}_nt_rule_count.npy'.format(lex_type, config.data))
    word_rule = np.load('{}.{}_word_rule_count.npy'.format(lex_type, config.data))
    word_word = np.load('{}.{}_word_word_count.npy'.format(lex_type, config.data))

    id2word = dict()
    for k, v in dictionary['word'].items():
        id2word[v] = k
    if config.path:
        hred = torch.load(config.path)
        hred.flatten_parameters()
    else:
        if config.vhred:
            hred = torch.load('vhred.pt')
            hred.flatten_parameters()
        else:
            '''
            if config.stanford:
                if config.unlex:
                    hred = torch.load('unlex.{}.pt'.format(config.data))
                else:
                    hred = torch.load('lex.stanford.right.{}.pt'.format(config.data))
                #hred = torch.load('lex.bin.{}.pt'.format(config.data))
            else:
                hred = torch.load('lex.{}.pt'.format(config.data))
            '''
            hred = torch.load('{}.{}.pt'.format(config.type, config.data))
            hred.init_rules(id2word)
            hred.flatten_parameters()
            hred.check_grammar(nt_word, nt_rule, word_rule, word_word)

    max_len = 150
    
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
    align = pickle.load(open('./data/persona.test.align', 'rb'))
    persona = pickle.load(open('./data/persona.test.parse.psn', 'rb'))

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
    output = open('{}.txt'.format(config.type), 'w')
    #output = open('alt.txt', 'w')
    for _, batch in enumerate(test_loader):
        sys.stderr.write(str(_))
        sys.stderr.write('\n')
        src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, rule_seqs, lex_seqs, leaf_indices, lex_indices, word_mask, rule_mask, noun_mask, positions, ancestors, anc_lengths = batch
        #if align[_][0] == -1: 
        if True:
            with torch.no_grad():
                responses = hred.generate(src_seqs, src_lengths, psn_seqs, psn_lengths, indices, max_len, 20, 20, _)
        else:
            slots = [dictionary['word'][w] for w in align[_][1]]
            print([w for w in align[_][1]])
            responses = hred.generate(src_seqs, src_lengths, psn_seqs, psn_lengths, indices, max_len, 30, 20, slots)
        #responses = hred.generate(src_seqs, src_lengths, psn_seqs, psn_lengths, indices, max_len, 30, 20)
        print_response(responses, out=output)
        output.write('\n')

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
    parser.add_argument('--type', type=str, default='lex2')
    parser.add_argument('--attn', default=True, action='store_true')
    parser.add_argument('--unlex', default=False, action='store_true')
    parser.add_argument('--stanford', default=False, action='store_true')
    config = parser.parse_args()
    main(config)
