import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
import json
import pdb
import time
import numpy as np
import argparse
from data_loader import get_loader
from masked_cel import compute_loss

from model import AttnDecoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def main(config):
    print(config)

    dictionary = json.load(open('./{}.lex2.dictionary.json'.format(config.data)))
    noun_id = []
    for k, v in dictionary['const'].items():
        if k[:2] == 'NN':
            noun_id.append(v)
    vocab_size = len(dictionary['word']) + 1
    word_embedding_dim = 300
    print("Vocabulary size:", vocab_size)

    word_vectors = np.random.uniform(low=-0.1, high=0.1, size=(vocab_size, word_embedding_dim))
    

    batch_size = 40 if config.data == 'persona' else 10
    if config.data in ['persona', 'movie']:
        train_loader = get_loader('./data/{}.train.src'.format(config.data), './data/{}.train.lex2.dat'.format(config.data), './data/{}.train.psn'.format(config.data), dictionary, batch_size)
        dev_loader = get_loader('./data/{}.valid.src'.format(config.data), './data/{}.valid.lex2.dat'.format(config.data), './data/{}.valid.psn'.format(config.data), dictionary, 10)
    else:
        train_loader = get_loader('./data/{}.train.src'.format(config.data), './data/{}.train.trg'.format(config.data), None, dictionary, 20)
        dev_loader = get_loader('./data/{}.valid.src'.format(config.data), './data/{}.valid.trg'.format(config.data), None, dictionary, 200)

    hidden_size = 512
    cenc_input_size = hidden_size * 2

    start_batch = 50000
    start_kl_weight = config.start_kl_weight

    if not config.use_saved:
        hred = AttnDecoder(word_embedding_dim, hidden_size, hidden_size, vocab_size, word_vectors, dictionary['word'], config.data, 0.5).cuda()
        for p in hred.parameters():
            torch.nn.init.uniform(p.data, a=-0.1, b=0.1)
        if config.glove:
            print("Loading word vecotrs.")
            word2vec_file = open('./glove.42B.300d.txt')
            next(word2vec_file)
            found = 0
            for line in word2vec_file:
                word, vec = line.split(' ', 1)
                if word in dictionary['word']:
                    word_vectors[dictionary['word'][word]] = np.fromstring(vec, dtype=np.float32, sep=' ')
                    found += 1
            print(found)
    else:
        hred = torch.load('attn.{}.pt'.format(config.data)).cuda()
        hred.flatten_parameters()
    hred.data = config.data
    params = filter(lambda x: x.requires_grad, hred.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)

    best_loss = np.inf
    last_dev_loss = np.inf
    power = 2
    for it in range(18, 30):
        ave_loss = 0
        last_time = time.time()
        params = filter(lambda x: x.requires_grad, hred.parameters())
        optimizer = torch.optim.SGD(params, lr=.1 * 0.95 ** it, momentum=0.9)
        hred.train()
        for _, (src_seqs, src_lengths, trg_seqs, trg_lengths, psn_seqs, psn_lengths, indices, pos_seqs) in enumerate(train_loader):
            if _ % config.print_every_n_batches == 1:
                print(ave_loss / min(_, config.print_every_n_batches), time.time() - last_time)
                ave_loss = 0
            loss, noun_loss, count = hred.loss(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, pos_seqs, noun_id, 1)
            ave_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(params, .1)
            optimizer.step()

        # eval on dev
        dev_loss = 0
        dev_nn_loss = 0
        count = 0
        nn_total_count = 0
        hred.eval()
        for i, (src_seqs, src_lengths, trg_seqs, trg_lengths, psn_seqs, psn_lengths, indices, pos_seqs) in enumerate(dev_loader):
            loss, noun_loss, nn_count = hred.loss(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, pos_seqs, noun_id, 1)
            dev_loss += loss.data[0]
            dev_nn_loss += noun_loss.data[0] * nn_count.data[0]
            nn_total_count += nn_count.data[0]
            count += 1
        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(hred, 'attn.{}.pt'.format(config.data))
        if dev_loss > last_dev_loss:
            power += 1
            hred = torch.load('attn.{}.pt'.format(config.data))
        last_dev_loss = dev_loss
        print('dev loss: {} {}'.format(dev_loss / count, dev_nn_loss / nn_total_count))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vhred', type=bool, default=False)
    parser.add_argument('--use_saved', type=bool, default=False)
    parser.add_argument('--print_every_n_batches', type=int, default=200)
    parser.add_argument('--kl_weight', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.25)
    parser.add_argument('--save_path', type=str, default='./attn.pt')
    parser.add_argument('--data', type=str, default='persona')
    parser.add_argument('--start_kl_weight', type=float, default=0)
    parser.add_argument('--glove', action='store_true', default=False)
    config = parser.parse_args()
    main(config)
