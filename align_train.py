import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
import json
import pdb
import time
import numpy as np
import argparse
from align_data_loader import get_loader
from masked_cel import compute_loss
from gensim.models import Word2Vec

from model import PersonaAligner


def main(config):
    print(config)

    dictionary = json.load(open('./{}.parse.dictionary.json'.format(config.data)))
    vocab_size = len(dictionary) + 1
    word_embedding_dim = 300
    print("Vocabulary size:", len(dictionary))

    word_vectors = np.random.uniform(low=-0.5, high=0.5, size=(vocab_size, word_embedding_dim))
    found = 0
    print("Loading word vecotrs.")
    if config.glove:
        word2vec_file = open('./glove.6B.300d.txt')
        next(word2vec_file)
        for line in word2vec_file:
            word, vec = line.split(' ', 1)
            if word in dictionary:
                word_vectors[dictionary[word]] = np.fromstring(vec, dtype=np.float32, sep=' ')
                found += 1
    else:
        word2vec = Word2Vec.load('./word2vec.vector')
        for word in word2vec.wv.vocab:
            if word in dictionary:
                word_vectors[dictionary[word]] = word2vec.wv[word]
                found += 1
    print(found)

    hidden_size = 512
    cenc_input_size = hidden_size * 2

    start_batch = 50000
    start_kl_weight = config.start_kl_weight

    if config.vhred:
        train_loader = get_hr_loader('./data/{}.train.src'.format(config.data), './data/{}.train.trg'.format(config.data), dictionary, 40)
        dev_loader = get_hr_loader('./data/{}.valid.src'.format(config.data), './data/{}.valid.trg'.format(config.data), dictionary, 200)
        if not config.use_saved:
            hred = VHRED(dictionary, vocab_size, word_embedding_dim, word_vectors, hidden_size)
            print('load hred param')
            _hred = torch.load('hred.pt')
            hred.u_encoder = _hred.u_encoder
            hred.c_encoder = _hred.c_encoder
            hred.decoder.rnn = _hred.decoder.rnn
            hred.decoder.output_transform = _hred.decoder.output_transform
            hred.decoder.context_hidden_transform.weight.data[:,0:hidden_size] = \
                _hred.decoder.context_hidden_transform.weight.data
            hred.flatten_parameters()
        else:
            hred = torch.load('vhred.pt')
            hred.flatten_parameters()
    elif config.attn:
        if config.data == 'persona':
            train_loader = get_ctc_loader('./data/{}.train.src'.format(config.data),
                                          './data/{}.train.trg'.format(config.data),
                                          './data/{}.train.psn'.format(config.data),
                                          dictionary, 40)
            dev_loader = get_ctc_loader('./data/{}.valid.src'.format(config.data),
                                        './data/{}.valid.trg'.format(config.data),
                                        './data/{}.valid.psn'.format(config.data),
                                        dictionary, 200)
            if not config.use_saved:
                hred = PersonaAttnDecoder(word_embedding_dim, hidden_size, vocab_size, word_vectors, dictionary).cuda()
            else:
                hred = torch.load('attn.pt')
                hred.flatten_parameters()
        else:
            train_loader = get_loader('./data/{}.train.src'.format(config.data), './data/{}.train.trg'.format(config.data), dictionary, 40)
            dev_loader = get_loader('./data/{}.valid.src'.format(config.data), './data/{}.valid.trg'.format(config.data), dictionary, 200)
            if not config.use_saved:
                hred = AttnDecoder(word_embedding_dim, hidden_size, vocab_size, word_vectors, dictionary).cuda()
            else:
                hred = torch.load('attn.pt')
                hred.flatten_parameters()
    else:
        train_loader = get_loader('./data/{}.train.src'.format(config.data), './data/{}.train.psn'.format(config.data), './data/{}.train.align'.format(config.data), dictionary, 40)
        dev_loader = get_loader('./data/{}.valid.src'.format(config.data), './data/{}.valid.psn'.format(config.data), './data/{}.valid.align'.format(config.data), dictionary, 200)
        if not config.use_saved:
            hred = PersonaAligner(word_embedding_dim, hidden_size, vocab_size, word_vectors).cuda()
        else:
            hred = torch.load('aligner.pt')
            hred.flatten_parameters()
    params = filter(lambda x: x.requires_grad, hred.parameters())
    #optimizer = torch.optim.SGD(params, lr=config.lr, momentum=0.99)
    #q_optimizer = torch.optim.SGD(hred.q_network.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(params, lr=0.001)

    best_loss = 0
    for it in range(0, 20):
        ave_loss = 0
        last_time = time.time()
        for _, batch in enumerate(train_loader):
            src_seqs, src_lengths, ctc_seqs, ctc_lengths, indices, labels = batch
            if _ % config.print_every_n_batches == 1:
                print(ave_loss / min(_, config.print_every_n_batches), time.time() - last_time)
                ave_loss = 0
            if config.vhred and config.kl_weight and it * len(train_loader) + _ <= start_batch:
                kl_weight = start_kl_weight + (1 - start_kl_weight) * float(it * len(train_loader) + _) / start_batch
                # kl_weight = 0.5
                loss = hred.loss(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, ctc_lengths, kl_weight)
            else:
                loss = hred.loss(src_seqs, src_lengths, ctc_seqs, ctc_lengths, indices, labels)
                #loss = hred.augmented_loss(src_seqs, src_lengths, indices, ctc_seqs, ctc_lengths, ctc_indices, trg_seqs, trg_lengths, trg_indices, turn_len, 0.1)
            ave_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(params, 0.1)
            optimizer.step()

        # eval on dev
        dev_loss = 0
        count = 0
        correct = 0
        recall = 0
        recall_count = 0
        for _, batch in enumerate(dev_loader):
            src_seqs, src_lengths, ctc_seqs, ctc_lengths, indices, labels = batch
            prediction = hred.evaluate(src_seqs, src_lengths, ctc_seqs, ctc_lengths, indices, labels).data.cpu().numpy()
            labels = np.asarray(labels)
            recall_count += np.sum(labels != -1)
            labels[labels==-1] = max(ctc_lengths)
            count += len(labels)
            correct += np.sum(prediction == labels)
            recall += np.sum(prediction == labels & labels != max(ctc_lengths))
        dev_loss = correct / count
        recall /= recall_count
        print('dev loss: {}. recall: {}'.format(dev_loss, recall))
        if dev_loss > best_loss:
            torch.save(hred, 'aligner.pt')
            best_loss = dev_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vhred', type=bool, default=False)
    parser.add_argument('--use_saved', type=bool, default=False)
    parser.add_argument('--print_every_n_batches', type=int, default=200)
    parser.add_argument('--kl_weight', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.25)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--data', type=str, default='persona')
    parser.add_argument('--start_kl_weight', type=float, default=0)
    parser.add_argument('--attn', default=False, action='store_true')
    parser.add_argument('--glove', default=False, action='store_true')
    config = parser.parse_args()
    main(config)
