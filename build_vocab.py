import argparse
import json
from collections import Counter


def build_word2id(seq_paths, min_word_count):
    """Creates word2id dictionary.
    
    Args:
        seq_path: String; text file path
        min_word_count: Integer; minimum word count threshold
        
    Returns:
        word2id: Dictionary; word-to-id dictionary
    """
    sequences = []
    num_seq = 0
    for seq in seq_paths:
        sequences += open(seq_path.readlines())
        num_seqs += len(sequences)

    counter = Counter()
    
    for i, sequence in enumerate(sequences):
        tokens = sequence.strip().split()
        counter.update(tokens)

        if i % 1000 == 0:
            print("[{}/{}] Tokenized the sequences.".format(i, num_seqs))

    # create a dictionary and add special tokens
    word2id = {}
    word2id['<start>'] = 0
    word2id['<end>'] = 1
    word2id['<unk>'] = 2
    
    # add the words to the word2id dictionary
    ind = len(word2id)
    for word, count in counter.items():
        if count >= min_word_count:
            word2id[word] = ind
            ind += 1
    
    return word2id


def main(config):
    
    # build word2id dictionaries for source and target sequences
    dictionary = build_word2id([config.src_path, config.trg_path], config.min_word_count)
    
    # save word2id dictionaries
    with open(config.dict_path, 'w') as f:
        json.dump(dictionary, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str)
    parser.add_argument('--trg_path', type=str)
    parser.add_argument('--dict_path', type=str, default='./data/dictionary.json')
    parser.add_argument('--min_count', type=int, default=4)
    config = parser.parse_args()
    print (config)
    main(config)
