import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
import json
import pdb
import time
import numpy as np
import argparse
import pickle
#from context_data_loader import get_ctc_loader
from masked_cel import compute_loss
from gensim.models import Word2Vec

from grammar_model import LexicalizedGrammarDecoder, DummyLexicalizedGrammarDecoder


def main(config):
    print(config)

    if config.unlex:
        from unlex_data_loader import get_loader
        config.type = 1
    else:
        from lex_data_loader import get_loader
    dictionary = json.load(open('./{}.lex{}.dictionary.json'.format(config.data, config.type)))
    vocab_size = len(dictionary['word']) + 1
    nt_vocab_size = len(dictionary['const']) + 1
    rule_vocab_size = len(dictionary['rule']) + 1
    word_embedding_dim = 300
    nt_embedding_dim = 300
    rule_embedding_dim = 300
    print("Vocabulary size:", vocab_size, nt_vocab_size, rule_vocab_size)

    word_vectors = np.random.uniform(low=-0.1, high=0.1, size=(vocab_size, word_embedding_dim))
    nt_vectors = np.random.uniform(low=-0.1, high=0.1, size=(nt_vocab_size, nt_embedding_dim))
    rule_vectors = np.random.uniform(low=-0.1, high=0.1, size=(rule_vocab_size, rule_embedding_dim))
    found = 0
    nt_found = 0
    rule_found = 0
    '''
        word2vec = open('./data/{}.vector'.format(config.data))
        next(word2vec)
        for line in word2vec:
            tokens = line.rsplit(' ', 300)
            word = tokens[0]
            vec = ' '.join(tokens[1:])
            if word in dictionary['const']:
                word_vectors[dictionary['const'][word]] = np.fromstring(vec, dtype=np.float32, sep=' ')
                nt_found += 1
            elif word in dictionary['rule']:
                word_vectors[dictionary['rule'][word]] = np.fromstring(vec, dtype=np.float32, sep=' ')
                rule_found += 1
        '''

    '''
    graph_dict = pickle.load(open('{}.graph.dictionary'.format(config.data), 'rb'))
    node2vec_file = open('./{}.node2vec'.format(config.data))
    next(node2vec_file)
    for line in node2vec_file:
        ind, vec = line.split(' ', 1)
        vec = np.fromstring(vec, dtype=np.float32, sep=' ')
        nt = graph_dict['ind2nt'][int(ind)]
        if nt in dictionary['const']:
            nt_vectors[dictionary['const'][nt]] = vec
            nt_found += 1
        rule = 'RULE: {}'.format(nt)
        if rule in dictionary['rule']:
            rule_vectors[dictionary['rule'][rule]] = vec
            rule_found += 1
    '''


    if config.data == 'persona':
        hidden_size = 512
        batch_size = 40
    else:
        hidden_size = 256
        batch_size = 10

    if config.unlex:
        lex_level = 0
    else:
        lex_level = config.type

    grammar_file_substr = ''
    if config.grammar:
        grammar_file_substr = '.parse'
    if config.rules:
        grammar_file_substr = '.rules'
    if config.data not in  ['persona', 'movie']:
        train_loader = get_loader('./data/{}.train.lex3.src.dat'.format(config.data), 
                                  './data/{}.train.lex3.trg.dat'.format(config.data, grammar_file_substr),
                                  None,
                                  dictionary, 20)
        dev_loader = get_loader('./data/{}.valid.lex3.src.dat'.format(config.data),
                                './data/{}.valid.lex3.trg.dat'.format(config.data, grammar_file_substr),
                                None,
                                dictionary, 40)
        if not config.use_saved:
            hred = DummyLexicalizedGrammarDecoder(word_embedding_dim, nt_embedding_dim, rule_embedding_dim, 
                                                  hidden_size, hidden_size, 
                                                  vocab_size, nt_vocab_size, rule_vocab_size,
                                                  word_vectors, nt_vectors, rule_vectors,
                                                  dictionary['word'], dictionary['const'], dictionary['rule'],
                                                  lex_level=lex_level,
                                                  data=config.data
                                                 ).cuda()
            for p in hred.parameters():
                torch.nn.init.uniform(p.data, a=-0.1, b=0.1)
            if config.glove:
                print("Loading word vecotrs.")
                word2vec_file = open('./glove.42B.300d.txt')
                next(word2vec_file)
                for line in word2vec_file:
                    word, vec = line.split(' ', 1)
                    if word in dictionary['word']:
                        word_vectors[dictionary['word'][word]] = np.fromstring(vec, dtype=np.float32, sep=' ')
                        found += 1
            print(found, nt_found, rule_found)
        else:
            hred = torch.load('lex.{}.pt'.format(config.data))
            hred.flatten_parameters()
    else:
        train_loader = get_loader('./data/{}.train.src'.format(config.data), 
                                  './data/{}.train.lex{}.dat'.format(config.data, config.type),
                                  './data/{}.train.trg'.format(config.data),
                                  './data/{}.train.psn'.format(config.data),
                                  dictionary, batch_size)
        dev_loader = get_loader('./data/{}.valid.src'.format(config.data),
                                './data/{}.valid.lex{}.dat'.format(config.data, config.type),
                                './data/{}.valid.trg'.format(config.data),
                                './data/{}.valid.psn'.format(config.data),
                                dictionary, batch_size)
        if not config.use_saved:
            if config.dual:
                hred = LexicalizedGrammarDualDecoder(word_embedding_dim, nt_embedding_dim, rule_embedding_dim, 
                                                     1200, hidden_size, 
                                                     vocab_size, nt_vocab_size, rule_vocab_size,
                                                     word_vectors, nt_vectors,
                                                     dictionary['word'], dictionary['const'], dictionary['rule'],
                                                     lex_level=lex_level
                                                    ).cuda()
            else:
                hred = DummyLexicalizedGrammarDecoder(word_embedding_dim, nt_embedding_dim, rule_embedding_dim, 
                                                 hidden_size, hidden_size, 
                                                 vocab_size, nt_vocab_size, rule_vocab_size,
                                                 word_vectors, nt_vectors, rule_vectors,
                                                 dictionary['word'], dictionary['const'], dictionary['rule'],
                                                 dropout=0.0,
                                                 lex_level=lex_level,
                                                 data=config.data
                                                ).cuda()
            hred.init_rules(None)
        else:
            hred = torch.load('lex{}.{}.pt'.format(config.type, config.data))
            #hred = torch.load('lex.stanford.right2.{}.pt'.format(config.data))
            hred.flatten_parameters()
    params = filter(lambda x: x.requires_grad, hred.parameters())
    #optimizer = torch.optim.SGD(params, lr=config.lr, momentum=0.99)
    #q_optimizer = torch.optim.SGD(hred.q_network.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(params, lr=0.001)

    if config.unlex:
        config.type = 0

    best_loss = np.inf
    last_dev_loss = np.inf
    power = 0
    for it in range(0, 25):
        ave_loss, ave_w_loss, ave_r_loss, ave_nt_loss = 0, 0, 0, 0
        last_time = time.time()
        hred.train()
        params = filter(lambda x: x.requires_grad, hred.parameters())
        optimizer = torch.optim.SGD(params, lr=0.1 * 0.9 ** it, momentum=0.9)
        for _, batch in enumerate(train_loader):
            src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, rule_seqs, lex_seqs, leaf_indices, lex_indices, word_mask, rule_mask, noun_maks, positions, ancestors, anc_lengths = batch
            if _ % config.print_every_n_batches == 1:
                print(np.array([ave_loss, ave_w_loss, ave_r_loss, ave_nt_loss]) / min(_, config.print_every_n_batches), time.time() - last_time)
                ave_loss, ave_w_loss, ave_r_loss, ave_nt_loss = 0, 0, 0, 0
            w_loss, nt_loss, r_loss, _, _ = hred.loss(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, rule_seqs, lex_seqs, leaf_indices, lex_indices, word_mask, rule_mask, noun_maks, positions, ancestors, anc_lengths)
            if lex_level > 0:
                #loss = w_loss + 1 * (r_loss + nt_loss)
                loss = w_loss + r_loss
            else:
                loss = w_loss + r_loss
            ave_loss += loss.data[0]
            ave_w_loss += w_loss.data[0]
            ave_r_loss += r_loss.data[0]
            if lex_level > 0:
                ave_nt_loss += nt_loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(params, .1)
            optimizer.step()

        # eval on dev
        dev_loss, dev_w_loss, dev_r_loss, dev_nt_loss, dev_nn_loss = 0, 0, 0, 0, 0
        nn_total_count = 0
        count = 0
        hred.eval()
        for _, batch in enumerate(dev_loader):
            src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, rule_seqs, lex_seqs, leaf_indices, lex_indices, word_mask, rule_mask, noun_mask, positions, ancestors, anc_lengths = batch
            w_loss, nt_loss, r_loss, noun_loss, nn_count = hred.loss(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, rule_seqs, lex_seqs, leaf_indices, lex_indices, word_mask, rule_mask, noun_mask, positions, ancestors, anc_lengths)
            if lex_level > 0:
                #loss = w_loss + 1 * (r_loss + nt_loss)
                loss = w_loss + r_loss
            else:
                loss = w_loss + r_loss
            dev_loss += loss.data[0]
            dev_r_loss += r_loss.data[0]
            dev_w_loss += w_loss.data[0]
            dev_nn_loss += noun_loss.data[0] * nn_count
            nn_total_count += nn_count
            if lex_level > 0:
                dev_nt_loss += nt_loss.data[0]
            count += 1
        print('dev loss:', np.array([dev_loss, dev_w_loss, dev_r_loss, dev_nt_loss, dev_nn_loss]) / count, dev_nn_loss / nn_total_count)
        if dev_loss < best_loss:
            if config.glove:
                torch.save(hred, 'lex{}.{}.glove.pt'.format(config.type, config.data))
            else:
                torch.save(hred, 'lex{}.{}.pt'.format(config.type, config.data))
            best_loss = dev_loss
        if dev_loss > last_dev_loss:
            power += 1
            #hred = torch.load('lex{}.{}.pt'.format(config.type, config.data))
        last_dev_loss = dev_loss

    for it in range(0, 0):
        ave_loss = 0
        last_time = time.time()
        for _, (src_seqs, src_lengths, indices, ctc_seqs, ctc_lengths, ctc_indices, trg_seqs, trg_lengths, trg_indices, turn_len) in enumerate(train_loader):
            loss = hred.train_decoder(src_seqs, src_lengths, indices, turn_len, 30, 5, 5)
            ave_loss += loss.data[0]
            q_optimizer.zero_grad()
            loss.backward()
            q_optimizer.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vhred', type=bool, default=False)
    parser.add_argument('--use_saved', default=False, action='store_true')
    parser.add_argument('--print_every_n_batches', type=int, default=200)
    parser.add_argument('--kl_weight', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.25)
    parser.add_argument('--type', type=int, default=2)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--data', type=str, default='persona')
    parser.add_argument('--start_kl_weight', type=float, default=0)
    parser.add_argument('--attn', default=False, action='store_true')
    parser.add_argument('--glove', default=False, action='store_true')
    parser.add_argument('--grammar', default=False, action='store_true')
    parser.add_argument('--rules', default=False, action='store_true')
    parser.add_argument('--dual', default=False, action='store_true')
    parser.add_argument('--unlex', default=False, action='store_true')
    config = parser.parse_args()
    main(config)
