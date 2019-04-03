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
from lm_data_loader import get_loader
#from context_data_loader import get_ctc_loader
from masked_cel import compute_loss
from gensim.models import Word2Vec

from grammar_model import LexicalizedGrammarLM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def main(config):
    print(config)

    dictionary = json.load(open('./{}.lex.dictionary.json'.format(config.data)))
    vocab_size = len(dictionary['word']) + 1
    nt_vocab_size = len(dictionary['const']) + 1
    rule_vocab_size = len(dictionary['rule']) + 1
    word_embedding_dim = 200
    nt_embedding_dim = 200
    rule_embedding_dim = 200
    print("Vocabulary size:", vocab_size, nt_vocab_size, rule_vocab_size)

    word_vectors = np.random.uniform(low=-0.1, high=0.1, size=(vocab_size, word_embedding_dim))
    nt_vectors = np.random.uniform(low=-0.1, high=0.1, size=(nt_vocab_size, nt_embedding_dim))
    rule_vectors = np.random.uniform(low=-0.1, high=0.1, size=(rule_vocab_size, rule_embedding_dim))
    found = 0
    nt_found = 0
    rule_found = 0
    print("Loading word vecotrs.")
    if config.glove:
        word2vec_file = open('./glove.6B.200d.txt')
        next(word2vec_file)
        for line in word2vec_file:
            word, vec = line.split(' ', 1)
            if word in dictionary['word']:
                word_vectors[dictionary['word'][word]] = np.fromstring(vec, dtype=np.float32, sep=' ')
                found += 1
    else:
        word2vec = Word2Vec.load('./word2vec.vector')
        for word in word2vec.wv.vocab:
            if word in dictionary:
                word_vectors[dictionary[word]] = word2vec.wv[word]
                found += 1

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
    print(found, nt_found, rule_found)

    hidden_size = 400
    if config.unlex:
        lex_level = 0
    else:
        lex_level = 3

    grammar_file_substr = ''
    if config.grammar:
        grammar_file_substr = '.parse'
    if config.rules:
        grammar_file_substr = '.rules'
    if config.vhred:
        train_loader = get_hr_loader('./data/{}.train.src'.format(config.data), './data/{}.train.trg'.format(config.data), dictionary, 40)
        dev_loader = get_hr_loader('./data/{}.valid.src'.format(config.data), './data/{}.valid.trg'.format(config.data), dictionary, 100)
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
    else:
        #if config.data == 'persona':
        if False:
            train_loader = get_ctc_loader('./data/{}.train.src'.format(config.data), 
                                          './data/{}.train{}.trg'.format(config.data, grammar_file_substr),
                                          './data/{}.train.psn'.format(config.data),
                                          dictionary, 40)
            dev_loader = get_ctc_loader('./data/{}.valid.src'.format(config.data),
                                        './data/{}.valid{}.trg'.format(config.data, grammar_file_substr),
                                        './data/{}.valid.psn'.format(config.data),
                                        dictionary, 80)
            if not config.use_saved:
                if config.attn:
                    hred = PersonaGrammarDecoder(word_embedding_dim, hidden_size, vocab_size, word_vectors, dictionary).cuda()
                else:
                    hred = GrammarDecoder(word_embedding_dim, hidden_size, vocab_size, word_vectors, dictionary).cuda()
            else:
                hred = torch.load('grammar.{}.pt'.format(config.data))
                hred.flatten_parameters()
        else:
            train_loader = get_loader('./data/msr.train.lex2.dat'.format(config.data), 
                                      dictionary, 20)
            dev_loader = get_loader('./data/msr.valid.lex.dat'.format(config.data), 
                                    dictionary, 20)
            if not config.use_saved:
                hred = LexicalizedGrammarLM(word_embedding_dim, nt_embedding_dim, rule_embedding_dim, 
                                                 512, hidden_size, 
                                                 vocab_size, nt_vocab_size, rule_vocab_size,
                                                 word_vectors, nt_vectors, rule_vectors,
                                                 dictionary['word'], dictionary['const'], dictionary['rule'],
                                                 lex_level=lex_level
                                                ).cuda()
                hred.init_rules()
            else:
                hred = torch.load('lm.{}.pt'.format(config.data))
                hred.flatten_parameters()
    params = filter(lambda x: x.requires_grad, hred.parameters())
    #optimizer = torch.optim.SGD(params, lr=config.lr, momentum=0.99)
    #q_optimizer = torch.optim.SGD(hred.q_network.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(params, lr=0.001)

    best_loss = np.inf
    for it in range(0, 5):
        ave_loss, ave_w_loss, ave_r_loss, ave_nt_loss = 0, 0, 0, 0
        last_time = time.time()
        hred.train()
        for _, batch in enumerate(train_loader):
            indices, trg_seqs, trg_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, word_mask, rule_mask, positions, ancestors, anc_lengths = batch
            if _ % config.print_every_n_batches == 1:
                print(np.array([ave_loss, ave_w_loss, ave_r_loss, ave_nt_loss]) / min(_, config.print_every_n_batches), time.time() - last_time)
                ave_loss, ave_w_loss, ave_r_loss, ave_nt_loss = 0, 0, 0, 0
            if config.data == 'persona':
                w_loss, nt_loss, r_loss = hred.loss(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, word_mask, rule_mask, positions, ancestors, anc_lengths)
                if lex_level > 0:
                    loss = w_loss + r_loss + nt_loss
                else:
                    loss = w_loss + r_loss
            else:
                w_loss, nt_loss, r_loss = hred.loss(indices, trg_seqs, trg_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, word_mask, rule_mask, positions, ancestors, anc_lengths)
                loss = w_loss + r_loss + nt_loss
            ave_loss += loss.data[0]
            ave_w_loss += w_loss.data[0]
            ave_r_loss += r_loss.data[0]
            if lex_level > 0:
                ave_nt_loss += nt_loss.data[0]
            optimizer.zero_grad()
            #loss.backward()
            w_loss.backward()
            torch.nn.utils.clip_grad_norm(params, 0.1)
            optimizer.step()

            if _ % 10000 == 0:
                # eval on dev
                dev_loss, dev_w_loss, dev_r_loss, dev_nt_loss = 0, 0, 0, 0
                count = 0
                hred.eval()
                for _, batch in enumerate(dev_loader):
                    indices, trg_seqs, trg_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, word_mask, rule_mask, positions, ancestors, anc_lengths = batch
                    if config.data == 'persona':
                        w_loss, nt_loss, r_loss = hred.loss(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, word_mask, rule_mask, positions, ancestors, anc_lengths)
                        if lex_level > 0:
                            loss = w_loss + r_loss + nt_loss
                        else:
                            loss = w_loss + r_loss
                    else:
                        w_loss, nt_loss, r_loss = hred.loss(indices, trg_seqs, trg_lengths, parent_seqs, sibling_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, word_mask, rule_mask, positions, ancestors, anc_lengths)
                        loss = w_loss + r_loss + nt_loss
                    dev_loss += loss.data[0]
                    dev_r_loss += r_loss.data[0]
                    dev_w_loss += w_loss.data[0]
                    if lex_level > 0:
                        dev_nt_loss += nt_loss.data[0]
                    count += 1
                print('dev loss:', np.array([dev_loss, dev_w_loss, dev_r_loss, dev_nt_loss]) / count)
                if dev_w_loss < best_loss:
                    torch.save(hred, 'lm.dropout.{}.pt'.format(config.data))
                    best_loss = dev_w_loss


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
    parser.add_argument('--grammar', default=False, action='store_true')
    parser.add_argument('--rules', default=False, action='store_true')
    parser.add_argument('--unlex', default=False, action='store_true')
    parser.add_argument('--dual', default=False, action='store_true')
    config = parser.parse_args()
    main(config)
