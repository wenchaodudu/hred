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
from hred_data_loader import get_hr_loader
from data_loader import get_loader
#from grammar_data_loader import get_loader
from context_data_loader import get_ctc_loader
from masked_cel import compute_loss

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys
import pickle


def main(config):
    dictionary = json.load(open('./{}.lex2.dictionary.json'.format(config.data)))
    vocab_size = len(dictionary['word'])
    word_embedding_dim = 300
    print("Vocabulary size:", vocab_size)

    '''
    if config.data == 'persona':
        test_loader = get_ctc_loader('./data/{}.test.src'.format(config.data),
                                     './data/{}.test.trg'.format(config.data),
                                     './data/{}.test.psn'.format(config.data),
                                     dictionary, 1, shuffle=False)
    else:
        test_loader = get_loader('./data/{}.test.src'.format(config.data), './data/{}.test.parse.trg'.format(config.data), dictionary, 1, shuffle=False)
    '''
    if config.data in ['persona', 'movie']:
        test_loader = get_loader('./data/{}.test.src'.format(config.data), 
                                 './data/{}.test.lex2.dat'.format(config.data), 
                                 './data/{}.test.psn'.format(config.data), 
                                 dictionary, 1, shuffle=False)
    else:
        test_loader = get_loader('./data/{}.test.src'.format(config.data), 
                                 './data/{}.test.trg'.format(config.data), 
                                 None, 
                                 dictionary, 1, shuffle=False)
    hidden_size = 512
    cenc_input_size = hidden_size * 2

    start_batch = 75000

    attn_model = torch.load('attn.{}.pt'.format(config.data))
    attn_model.flatten_parameters()
    '''
    grammar_model = torch.load('grammar.{}.pt'.format(config.data))
    rules_model = torch.load('rules.{}.pt'.format(config.data))
    rules_model.init_rules()
    grammar_model.flatten_parameters()
    rules_model.flatten_parameters()
    if config.path:
        hred = torch.load(config.path)
        hred.flatten_parameters()
    else:
        if config.vhred:
            hred = torch.load('vhred.pt')
            hred.flatten_parameters()
        else:
            hred = torch.load('hred-ad-2116.pt')
            hred.flatten_parameters()
    '''

    max_len = 100
    id2word = dict()
    for k, v in dictionary['word'].items():
        id2word[v] = k
    #id2word[20019] = '__eou__'
    '''
    rule_id2word = dict()
    for k, v in rules_model.dictionary.items():
        rule_id2word[v] = k
    '''

    def print_response(src, responses, output, dict_list):
        x = 0
        for x in range(src.size(0)):
            for y in range(src.size(1)):
                output.write(id2word[src[x, y]])
                if src[x, y] == dictionary['word']['__eou__']:
                    output.write('\n')
                    break
                else:
                    output.write(' ')
            output.write('\n')
            for _, res in enumerate(responses):
                linebreak = False
                for y in range(min(max_len, res.size(1))):
                    word = dict_list[_][res[x, y]]
                    if word == '__eou__':
                        output.write('\n')
                        linebreak = True
                        break
                    if _ == len(responses) - 1:
                        if (word != 'REDUCE' and word[:3] != 'NT-' and word[:4] != 'RULE') or word.find('SLOT') > -1:
                            output.write(word)
                            output.write(' ')
                    else:
                        if word != 'REDUCE' and word[:3] != 'NT-' and word[:4] != 'RULE':
                            output.write(word)
                            output.write(' ')
                if not linebreak:
                    output.write('\n')
                
    '''
    disc = torch.load('discriminator.pt')
    disc.flatten_parameters()
    qnet = torch.load('q_network_adv.pt')
    '''

    def word_embed_dist(trg_embed, trg_lengths, generations):
        gen_embed = hred.embedding(Variable(generations.cuda()))
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
        gen_embed = hred.embedding(Variable(generations.cuda()))
        leng, perm_idx = torch.cuda.LongTensor(leng).sort(0, descending=True)
        gen_embed = gen_embed[perm_idx]
        packed_input = pack_padded_sequence(gen_embed, leng.cpu().numpy(), batch_first=True)
        gen_output = disc.u_encoder(packed_input)
        en_output = gen_output[perm_idx.sort()[1]]
        return (dist / generations.size(0)).data.cpu(), np.mean(leng), gen_output

    dist = [0, 0, 0]
    lengths = [0, 0, 0]
    scores = [0, 0, 0]
    beam_size = 20
    top_k = 20
    output = open(config.output, 'w')
    align = pickle.load(open('./data/persona.test.align', 'rb'))
    persona = pickle.load(open('./data/persona.test.parse.psn', 'rb'))

    for _, batch in enumerate(test_loader):
        print(_)
        src_seqs, src_lengths, trg_seqs, trg_lengths, psn_seqs, psn_lengths, pos_seqs, indices = batch
        #attn_responses, nll = attn_model.generate(src_seqs, src_lengths, ctc_seqs, ctc_lengths, indices, max_len, beam_size, top_k)
        #grammar_responses, nll = grammar_model.generate(src_seqs, src_lengths, ctc_seqs, ctc_lengths, indices, max_len, beam_size, top_k)
        slots = []
        '''
        if align[_] > -1:
            noun = [i for i, x in enumerate(persona[_][align[_]]['parse']) if x == 'NT-NN'][-1:]
            slots = [dictionary[persona[_][align[_]]['parse'][x+1][4:]] for x in noun]
            #attn_responses, nll = attn_model.fsm_one_generate(src_seqs, src_lengths, indices, slots, max_len, beam_size, top_k, grammar=False)
            attn_responses, nll = attn_model.generate(src_seqs, src_lengths, indices, max_len, beam_size, top_k)
            grammar_responses, nll = grammar_model.fsm_one_generate(src_seqs, src_lengths, indices, slots, max_len, beam_size, top_k)
            #rules_responses, nll = grammar_model.fsm_one_generate(src_seqs, src_lengths, indices, slots, max_len, beam_size, top_k)
            rule_slots = [rules_model.dictionary[persona[_][align[_]]['parse'][x+1][4:]] for x in noun]
            rules_responses, nll = rules_model.rules_generate_cstr(src_seqs, src_lengths, indices, max_len, beam_size, top_k, rule_slots)
        else:
            #attn_responses, nll = attn_model.generate(src_seqs, src_lengths, indices, max_len, beam_size, top_k, grammar=False)
            attn_responses, nll = attn_model.generate(src_seqs, src_lengths, indices, max_len, beam_size, top_k)
            grammar_responses, nll = grammar_model.generate(src_seqs, src_lengths, indices, max_len, beam_size, top_k)
            rules_responses, nll = rules_model.rules_generate(src_seqs, src_lengths, indices, max_len, beam_size, top_k)
        '''
        #if align[_][0] == -1:
        if True:
            attn_responses, nll = attn_model.generate(src_seqs, src_lengths, psn_seqs, psn_lengths, indices, max_len, beam_size, top_k)
            print(nll.data[0])
        else:
            slots = [dictionary[w] for w in align[_][1]]
            print(slots)
            attn_responses, nll = attn_model.generate(src_seqs, src_lengths, psn_seqs, psn_lengths, indices, max_len, 15, top_k, slots)
        '''
        if len(slots[0]) < 6:
            fsm_grammar_responses, nll = grammar_model.fsm_generate(src_seqs, src_lengths, indices, torch.LongTensor([dictionary[slot[0]] for slot in slots[0]]), max_len, beam_size, top_k)
        else:
            fsm_grammar_responses, nll = grammar_model.grid_generate(src_seqs, src_lengths, indices, torch.LongTensor([dictionary[slot[0]] for slot in slots[0]]), max_len, beam_size, top_k)
        print_response(src_seqs, [trg_seqs, attn_responses, grammar_responses, fsm_grammar_responses], output)
        '''
        #print_response(src_seqs, [trg_seqs, attn_responses, grammar_responses, rules_responses], output, [id2word, id2word, id2word, rule_id2word])
        print_response(src_seqs, [trg_seqs, attn_responses], output, [id2word, id2word])
        #output.write(str(slots[0]))
        output.write('\n')
        print()

    print(np.asarray(dist) / _)
    print(np.asarray(lengths) / _)
    print(np.asarray(scores) / _)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vhred', type=bool, default=False)
    parser.add_argument('--use_saved', type=bool, default=False)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--print_every_n_batches', type=int, default=10)
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--output', type=str, default='generations.txt')
    parser.add_argument('--data', type=str, default='persona')
    parser.add_argument('--attn', default=False, action='store_true')
    config = parser.parse_args()
    main(config)
