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

from model import AttnDecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys


def main(config):
    dictionary = json.load(open('./dictionary.json'))
    vocab_size = len(dictionary) + 1
    word_embedding_dim = 300
    print("Vocabulary size:", len(dictionary))

    print("Loading word vecotrs.")
    word2vec_file = open('./word2vec.vector')
    word_vectors = np.random.uniform(low=-0.25, high=0.25, size=(vocab_size, word_embedding_dim))
    next(word2vec_file)
    for line in word2vec_file:
        word, vec = line.split(' ', 1)
        if word in dictionary:
            word_vectors[dictionary[word]] = np.fromstring(vec, dtype=np.float32, sep=' ')

    test_loader = get_loader('./data/test.src', './data/test.tgt', dictionary, 64)

    hidden_size = 512
    cenc_input_size = hidden_size * 2

    start_batch = 75000

    hred = torch.load(config.path)
    hred.flatten_parameters()
    max_len = 30
    id2word = dict()
    for k, v in dictionary.items():
        id2word[v] = k

    for src_seqs, src_lengths, trg_seqs, trg_lengths in test_loader:
        responses = hred.generate(src_seqs, src_lengths, max_len, 5, 5)
        for x in range(responses.size(0)):
            for y in range(max_len):
                sys.stdout.write(id2word[responses[x, y]])
                if responses[x, y] == dictionary['<end>'] or y == max_len - 1:
                    sys.stdout.write('\n')
                    break
                else:
                    sys.stdout.write(' ')
        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vhred', type=bool, default=False)
    parser.add_argument('--use_saved', type=bool, default=False)
    parser.add_argument('--print_every_n_batches', type=int, default=1000)
    parser.add_argument('--path', type=str, default='./cnn_attn.pt')
    config = parser.parse_args()
    main(config)
