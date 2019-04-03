import json
import pdb
import pickle
import sys
import nltk
import numpy as np
from collections import defaultdict
from anytree import Node, PreOrderIter
nltk.data.path.append('/pylon5/cc5fp3p/amadeus0/anaconda3/lib/nltk_data')
from nltk.corpus import treebank

'''
ptb_tag = defaultdict(list)
for word, tag in treebank.tagged_words():
    ptb_tag[word].append(tag)
'''

dictionary = json.load(open('./{}.lex{}.dictionary.json'.format(sys.argv[1], sys.argv[2])))
nt_word = np.zeros((len(dictionary['const'])+1, len(dictionary['word'])+1))
nt_rule = np.zeros((len(dictionary['const'])+1, len(dictionary['rule'])+1))
word_rule = np.zeros((len(dictionary['word'])+1, len(dictionary['rule'])+1))
word_word = np.zeros((len(dictionary['word'])+1, len(dictionary['word'])+1))
for split in ['valid', 'train', 'test']:
    data = pickle.load(open('data/{}.{}.lex{}.dat'.format(sys.argv[1], split, sys.argv[2]), 'rb'))
    #data = pickle.load(open('data/{}.{}.lex3.trg.dat'.format(sys.argv[1], split), 'rb'))
    for _, tree in enumerate(data):
        if _ % 10000 == 0:
            print(_)
        for node in PreOrderIter(tree):
            if node.is_leaf:
                nt, word, tag, _ = node.name.split('__')
                nt_word[dictionary['const'][nt], dictionary['word'][word]] += 1
            else:
                if split != 'test':
                    nt, word, tag, rule = node.name.split('__')
                    rule = rule[:rule.find('[')-1]
                    word_rule[dictionary['word'][word], dictionary['rule'][rule]] += 1
            if node.is_leaf and node.parent is not None and split != 'test':
                nt, word, tag, rule = node.name.split('__')
                p_nt, p_word, p_tag, _ = node.parent.name.split('__')
                if word != p_word:
                    word_word[dictionary['word'][p_word], dictionary['word'][word]] += 1

np.save('lex{}.{}_nt_word_count'.format(sys.argv[2], sys.argv[1]), nt_word)
np.save('lex{}.{}_nt_rule_count'.format(sys.argv[2], sys.argv[1]), nt_rule)
np.save('lex{}.{}_word_rule_count'.format(sys.argv[2], sys.argv[1]), word_rule)
np.save('lex{}.{}_word_word_count'.format(sys.argv[2], sys.argv[1]), word_word)
