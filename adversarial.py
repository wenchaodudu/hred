import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import json
import numpy as np
from gensim.models import Word2Vec

from model import HRED
from gumbel_softmax import *
from hred_data_loader import get_loader


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, input):
        pass


def train():
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

    train_loader = get_loader('./data/train.src', './data/train.tgt', dictionary, 32)
    dev_loader = get_loader('./data/valid.src', './data/valid.tgt', dictionary, 64)

    hidden_size = 300
    hred = HRED(dictionary, vocab_size, word_embedding_dim, word_vectors, hidden_size, None)
    disc = torch.load('discriminator.pt')

    trainD, trainG = False, True
    optim_G = torch.optim.SGD(filter(lambda x: x.requires_grad, hred.parameters()), lr=0.1, momentum=0.9)
    optim_D = torch.optim.SGD(filter(lambda x: x.requires_grad, disc.parameters()), lr=0.1, momentum=0.9)

    for it in range(0, 10):
        for _, (src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices,
                trg_seqs, trg_lengths, trg_indices, turn_len) in enumerate(train_loader):
            cenc_out = hred.encode(src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, turn_len)
            _, decode_out = hred.decode(cenc_out, trg_seqs, trg_lengths, trg_indices, sampling_rate=0.2, gumbel=True)

            # print(ctc_seqs.size(), ctc_lengths, ctc_indices)
            # print(trg_seqs.size(), trg_lengths, trg_indices)
            # print(decode_out.size())
            # print(decode_out)

            if _ % 10 < 6:
                trainD, trainG = False, True
            else:
                trainD, trainG = True, False

            if trainD:
                optim_D.zero_grad()
                disc_label = torch.zeros(trg_seqs.size()[0])
                loss = disc.loss(src_seqs, src_lengths, src_indices, decode_out, trg_lengths, trg_indices, disc_label, None)
                disc_label = torch.ones(trg_seqs.size()[0])
                loss += disc.loss(src_seqs, src_lengths, src_indices, trg_seqs, trg_lengths, trg_indices, disc_label, None)
                loss.backward()
                optim_D.step()

            elif trainG:
                optim_G.zero_grad()
                disc_label = torch.ones(trg_seqs.size()[0])
                loss = disc.loss(src_seqs, src_lengths, src_indices, decode_out, trg_lengths, trg_indices, disc_label, None)
                loss.backward()
                optim_G.step()


def reconstruct_sent(seq, dictionary):
    return ' '.join([dictionary[wid] for wid in filter(lambda x: x != 0, seq)])

if __name__ == '__main__':
    train()