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

    dictionary = json.load(open('./{}.lex.dictionary.json'.format(config.data)))['word']
    vocab_size = len(dictionary) + 1
    word_embedding_dim = 300
    print("Vocabulary size:", len(dictionary))

    print("Loading word vecotrs.")
    word2vec_file = open('./glove.6B.300d.txt')
    word_vectors = np.random.uniform(low=-0.25, high=0.25, size=(vocab_size, word_embedding_dim))
    next(word2vec_file)
    for line in word2vec_file:
        word, vec = line.split(' ', 1)
        if word in dictionary:
            word_vectors[dictionary[word]] = np.fromstring(vec, dtype=np.float32, sep=' ')

    train_loader = get_loader('./data/{}.train.src'.format(config.data), './data/{}.train.trg'.format(config.data), './data/{}.train.psn'.format(config.data), dictionary, 40)
    dev_loader = get_loader('./data/{}.valid.src'.format(config.data), './data/{}.valid.trg'.format(config.data), './data/{}.valid.psn'.format(config.data), dictionary, 200)

    hidden_size = 512
    cenc_input_size = hidden_size * 2

    start_batch = 50000
    start_kl_weight = config.start_kl_weight

    if not config.use_saved:
        hred = AttnDecoder(word_embedding_dim, hidden_size, vocab_size, word_vectors, dictionary).cuda()
    else:
        hred = torch.load(config.save_path)
        hred.flatten_parameters()
    params = filter(lambda x: x.requires_grad, hred.parameters())
    #optimizer = torch.optim.SGD(params, lr=config.lr, momentum=0.99)
    optimizer = torch.optim.Adam(params, lr=0.001)

    best_loss = np.inf
    for it in range(0, 10):
        ave_loss = 0
        last_time = time.time()
        for _, (src_seqs, src_lengths, trg_seqs, trg_lengths, psn_seqs, psn_lengths, indices) in enumerate(train_loader):
            if _ % config.print_every_n_batches == 1:
                print(ave_loss / min(_, config.print_every_n_batches), time.time() - last_time)
                ave_loss = 0
            loss = hred.loss(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, 1)
            ave_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm(params, 0.1)
            optimizer.step()

        # eval on dev
        dev_loss = 0
        count = 0
        for i, (src_seqs, src_lengths, trg_seqs, trg_lengths, psn_seqs, psn_lengths, indices) in enumerate(dev_loader):
            dev_loss += hred.loss(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, 1).data[0]
            count += 1
        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(hred, 'attn.{}.pt'.format(config.data))
        print('dev loss: {}'.format(dev_loss / count))



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
    config = parser.parse_args()
    main(config)
