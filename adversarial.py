import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import json
import numpy as np
from gensim.models import Word2Vec
import time

from model import HRED
from gumbel_softmax import *
from hred_data_loader import get_loader

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, input):
        pass


def train(config):
    dictionary = json.load(open('./dictionary.json'))
    inverse_dict = {}
    for word, wid in dictionary.items():
        inverse_dict[wid] = word
    inverse_dict[0] = '<0>'
    EOU = dictionary['__eou__']

    vocab_size = len(dictionary) + 1
    word_embedding_dim = 300
    print("Vocabulary size:", len(dictionary))

    word_vectors = np.random.uniform(low=-0.1, high=0.1, size=(vocab_size, word_embedding_dim))
    found = 0
    print("Loading word vecotrs.")
    word2vec = Word2Vec.load('./word2vec.vector')
    for word in word2vec.wv.vocab:
        if word in dictionary:
            word_vectors[dictionary[word]] = word2vec.wv[word]
            found += 1
    print(found)

    train_loader = get_loader('./data/train.src', './data/train.tgt', dictionary, 64)
    dev_loader = get_loader('./data/valid.src', './data/valid.tgt', dictionary, 128)

    hidden_size = 512
    sampling_rate = 0.4

    if config.use_saved:
        hred = torch.load('hred-ad.pt')
        hred.flatten_parameters()
        disc = torch.load('disc-ad.pt')
        disc.flatten_parameters()
    else:
        hred = HRED(dictionary, vocab_size, word_embedding_dim, word_vectors, hidden_size, None)
        disc = torch.load('discriminator.pt')
        disc.flatten_parameters()

    optim_G = torch.optim.SGD(filter(lambda x: x.requires_grad, hred.parameters()), lr=config.lr, momentum=0.9)
    optim_D = torch.optim.SGD(filter(lambda x: x.requires_grad, disc.parameters()), lr=config.lr, momentum=0.9)

    for it in range(0, 10):
        ave_g_loss, ave_d_loss, ave_ml_loss = 0, 0, 0
        last_time = time.time()

        for _, (src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices,
                trg_seqs, trg_lengths, trg_indices, turn_len) in enumerate(train_loader):

            if _ % config.print_every_n_batches == 1:
                print('***')
                print(ave_ml_loss / min(_, config.print_every_n_batches / 0.3),
                      ave_g_loss / min(_, config.print_every_n_batches / 0.6),
                      ave_d_loss / min(_, config.print_every_n_batches / 0.1),
                      time.time() - last_time)
                ave_g_loss, ave_d_loss, ave_ml_loss = 0, 0, 0
                torch.save(hred, 'hred-ad.pt')
                torch.save(disc, 'disc-ad.pt')

            cenc_out = hred.encode(src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len)
            decode_out, gumbel_out = hred.decode(cenc_out, trg_seqs, trg_lengths, trg_indices,
                                                 sampling_rate=sampling_rate, gumbel=True)

            gumbel_lengths = [x - 1 for x in trg_lengths]
            gumbel_indices = trg_indices

            if _ % config.print_every_n_batches == 1:
                for idx in range(min(gumbel_out.size()[0], 10)):
                    print('---')
                    print(reconstruct_sent(trg_seqs[idx][:trg_lengths[idx]], inverse_dict))
                    print(reconstruct_sent(gumbel_out[idx][:gumbel_lengths[idx]], inverse_dict))

            # train generator for 6 batches, then discriminator for 4 batches
            if _ % 100 < 30:
                trainD, trainG, ML = False, False, True
            elif _ % 100 < 90:
                trainD, trainG, ML = False, True, False
            else:
                trainD, trainG, ML = True, False, False

            if trainD:
                optim_D.zero_grad()
                disc_label = np.zeros(trg_seqs.size()[0])
                loss = disc.loss(ctc_seqs, ctc_lengths, ctc_indices, gumbel_out, gumbel_lengths, gumbel_indices, disc_label, None)
                loss.backward()
                ave_d_loss += loss.data[0]
                disc_label = np.ones(trg_seqs.size()[0])
                loss = disc.loss(ctc_seqs, ctc_lengths, ctc_indices, trg_seqs, trg_lengths, trg_indices, disc_label, None)
                loss.backward()
                ave_d_loss += loss.data[0]
                optim_D.step()
                # partial outputs

            elif trainG:
                optim_G.zero_grad()
                disc_label = np.ones(trg_seqs.size()[0])
                loss = disc.loss(ctc_seqs, ctc_lengths, ctc_indices, gumbel_out, gumbel_lengths, gumbel_indices, disc_label, None)
                loss.backward()
                optim_G.step()
                ave_g_loss += loss.data[0]

            elif ML:
                optim_G.zero_grad()
                loss = hred.loss(src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, trg_seqs,
                                 trg_lengths, trg_indices, turn_len, sampling_rate)
                loss.backward()
                optim_G.step()
                ave_ml_loss += loss.data[0]


def reconstruct_sent(seq, dictionary):
    return ' '.join([dictionary[wid] for wid in filter(lambda x: x != 0, seq)])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vhred', type=bool, default=False)
    parser.add_argument('--use_saved', type=bool, default=False)
    parser.add_argument('--print_every_n_batches', type=int, default=1000)
    parser.add_argument('--kl_weight', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.25)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--start_kl_weight', type=float, default=0)
    config = parser.parse_args()
    train(config)