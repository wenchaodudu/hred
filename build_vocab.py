import argparse
import json
from collections import Counter
import os
from glob import glob
import pdb


def build_word2id(seq_paths, min_word_count):
    """Creates word2id dictionary.
    
    Args:
        seq_path: String; text file path
        min_word_count: Integer; minimum word count threshold
        
    Returns:
        word2id: Dictionary; word-to-id dictionary
    """
    sequences = []
    num_seqs = 0
    for seq in seq_paths:
        sequence = open(seq).readlines()
        sequences.extend(sequence)
        num_seqs += len(sequence)

    counter = Counter()
    
    for i, sequence in enumerate(sequences):
        tokens = sequence.strip().split()
        counter.update(tokens)

        if i % 10000 == 0:
            print("[{}/{}] Tokenized the sequences.".format(i, num_seqs))

    # create a dictionary and add special tokens
    word2id = {}
    word2id['<PAD>'] = 0
    word2id['<GO>'] = 1
    
    # add the words to the word2id dictionary
    ind = 2
    for word, count in counter.items():
        if count >= min_word_count:
            word2id[word] = ind
            ind += 1
    
    return word2id


def main(config):
    
    # build word2id dictionaries for source and target sequences
    dictionary = build_word2id(glob(os.path.join(config.src_dir, '*')), config.min_count)
    
    # save word2id dictionaries
    with open(config.dict_path, 'w') as f:
        json.dump(dictionary, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default='./data')
    parser.add_argument('--dict_path', type=str, default='./data/dictionary.json')
    parser.add_argument('--min_count', type=int, default=1)
    config = parser.parse_args()
    print (config)
    main(config)
