import argparse
import json
import pickle
from collections import Counter
import os
from glob import glob
import pdb
import numpy as np
#from stanfordcorenlp import StanfordCoreNLP


def build_word2id(seq_paths, min_word_count):
    """Creates word2id dictionary.
    
    Args:
        seq_path: String; text file path
        min_word_count: Integer; minimum word count threshold
        
    Returns:
        word2id: Dictionary; word-to-id dictionary
    """
    parser = StanfordCoreNLP(r'../stanford-corenlp-full-2018-02-27')
    sequences = []
    parse_seqs = []
    num_seqs = 0
    counter = Counter()
    for seq in seq_paths:
        print(seq)
        if seq.find('parse') > -1 or seq.find('rules') > -1:
            f = open(seq, 'rb')
            f.seek(0)
            if seq.find('psn') > -1:
                sequence = [obj['parse'] for chat in pickle.load(f) for obj in chat]
            else:
                sequence = [obj['parse'] for obj in pickle.load(f)]
            parse_seqs.extend(sequence)
        elif seq.find('psn') > -1:
            sequence = open(seq).readlines()
            for persona in sequence:
                persona = persona.split('|')
                sequences.extend(persona)
                num_seqs += len(persona)
        else:
            sequence = open(seq).readlines()
            sequences.extend(sequence)
            num_seqs += len(sequence)
    
    for i, sequence in enumerate(sequences):
        tokens = sequence.strip().lower().split()
        counter.update(tokens)

        if i % 10000 == 0:
            print("[{}/{}] Tokenized the sequences.".format(i, num_seqs))

    for i, sequence in enumerate(parse_seqs):
        sequence = [t if t.find('GEN-') == -1 else t[4:].lower() for t in sequence]
        counter.update(sequence)

        if i % 10000 == 0:
            print("[{}/{}] Tokenized the sequences.".format(i, num_seqs))

    # create a dictionary and add special tokens
    word2id = {}
    word2id['<start>'] = 1
    word2id['__eou__'] = 2
    
    # add the words to the word2id dictionary
    ind = len(word2id) + 1
    for word, count in counter.items():
        if count >= min_word_count and word not in word2id:
            word2id[word] = ind
            ind += 1
    
    print(len(word2id))
    return word2id

def build_lex_word2id(seq_paths, min_word_count, config):
    """Creates word2id dictionary.
    
    Args:
        seq_path: String; text file path
        min_word_count: Integer; minimum word count threshold
        
    Returns:
        word2id: Dictionary; word-to-id dictionary
    """
    sequences = []
    parse_seqs = []
    num_seqs = 0
    counter = Counter()
    rule_dict = {}
    word_dict = {}
    nt_dict = {}
    word_dict['<start>'] = 1
    word_dict['__eou__'] = 2
    word_dict['<UNK>'] = 3
    rule_dict['<UNK>'] = 1
    rule_dict['RULE: EOD'] = 2
    word_count = Counter()
    rule_count = Counter()
    test_word_count = Counter()
    test_rule_count = Counter()

    def load(tree, rule_dict, word_dict, nt_dict, seq):
        nt, word, tag, rule = tree.name.split('__')
        word = word.lower()
        rule = rule[:rule.find('[')-1]
        '''
        if rule not in rule_dict:
            rule_dict[rule] = len(rule_dict) + 1
        '''
        if seq.find('test') > -1:
            test_rule_count[rule] += 1
        else:
            rule_count[rule] += 1
        if nt not in nt_dict:
            nt_dict[nt] = len(nt_dict) + 1
        '''
        if word not in word_dict:
            word_dict[word] = len(word_dict) + 1
        '''
        word_count[word] += 1
        for ch in tree.children:
            load(ch, rule_dict, word_dict, nt_dict, seq)

    for seq in seq_paths:
        if os.path.isdir(seq):
            continue
        if seq.split('/')[-1].find('lex') > -1:
            if seq.find('lex{}'.format(config.type)) > -1:
                print(seq)
                trees = np.load(seq)
                for tr in trees:
                    load(tr, rule_dict, word_dict, nt_dict, seq)
        else:
            print(seq)
            sequence = open(seq).readlines()
            if seq.find('psn') > 1:
                if (seq.find('rev') > -1 and config.rev) or ((seq.find('rev') == -1) and not config.rev):
                    sequence = [s.replace('|', ' ').replace('.', ' .') for s in sequence]
                else:
                    sequene = []
            sequences.extend(sequence)
            num_seqs += len(sequence)
    
    for i, sequence in enumerate(sequences):
        tokens = sequence.strip().lower().split()
        word_count.update(tokens)

        if i % 10000 == 0:
            print("[{}/{}] Tokenized the sequences.".format(i, num_seqs))

    for k, v in word_count.items():
        if v >= 1:
            if k not in word_dict:
                word_dict[k] = len(word_dict) + 1

    for k, v in rule_count.items():
        if v >= 1:
            if k not in rule_dict:
                rule_dict[k] = len(rule_dict) + 1

    print(len(word_dict), len(rule_dict), len(nt_dict))
    return {'word': word_dict, 'rule': rule_dict, 'const': nt_dict}



def main(config):
    
    if config.lex:
        dictionary = build_lex_word2id(glob(os.path.join('./{}'.format(config.data), '*')), config.min_count, config)
        #dictionary = build_lex_word2id(glob(os.path.join('./{}/'.format(config.data), '*')), config.min_count)
        json.dump(dictionary, open('{}.lex{}.dictionary.json'.format(config.data, config.type), 'w'))
    else:
        dictionary = build_word2id(glob(os.path.join(config.src_dir, '*')), config.min_count)
        
        # save word2id dictionaries
        with open(config.dict_path, 'w') as f:
            json.dump(dictionary, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default='./persona/Rules')
    parser.add_argument('--dict_path', type=str, default='./persona.rules.dictionary.json')
    parser.add_argument('--data', type=str, default='persona')
    parser.add_argument('--min_count', type=int, default=0)
    parser.add_argument('--type', type=int, default=2)
    parser.add_argument('--lex', action='store_true', default=False)
    parser.add_argument('--rev', action='store_true', default=False)
    config = parser.parse_args()
    print (config)
    main(config)
