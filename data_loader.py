import nltk
import json
import torch
import pickle
import torch.utils.data as data
import pdb
from string import punctuation
from anytree import Node, PreOrderIter
from nltk.corpus import stopwords


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, src_path, trg_path, psn_path, dictionary):
        """Reads source and target sequences from txt files."""
        self.src_seqs = open(src_path).readlines()
        self.num_total_seqs = len(self.src_seqs)
        parse_file = pickle.load(open(trg_path, 'rb'))
        #self.trg_seqs = open(trg_path).readlines()
        self.trg_seqs = [None for x in range(self.num_total_seqs)]
        self.pos_seqs = [None for x in range(self.num_total_seqs)]
        self.stopword = set(stopwords.words('english')) & set(['.'])
        if psn_path:
            self.psn_seqs = open(psn_path).readlines()
        else:
            self.psn_seqs = ['' for x in range(self.num_total_seqs)]
        self.max_len = 50
        self.word2id = dictionary['word']
        self.nt2id = dictionary['const']
        for x in range(self.num_total_seqs):
            if x % 10000 == 0:
                print(x)
            self.src_seqs[x] = self.preprocess(self.src_seqs[x], self.word2id, 'src')
            #self.trg_seqs[x] = self.preprocess(self.trg_seqs[x], self.word2id, 'trg')
            self.psn_seqs[x] = self.preprocess(self.psn_seqs[x], self.word2id, 'psn')
            self.src_seqs[x] = self.psn_seqs[x] + self.src_seqs[x]
            words, tags = self.preprocess_tree(parse_file[x])
            self.trg_seqs[x] = words
            self.pos_seqs[x] = tags

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        psn_seq = self.psn_seqs[index]
        pos_seq = self.pos_seqs[index]
        return src_seq, trg_seq, psn_seq, pos_seq

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, name):
        """Converts words to ids."""
        tokens = sequence.strip().lower()
        if name == 'src':
            if '__eou__' in tokens:
                tokens = ' '.join(tokens.split('__eou__')[-2:-1]).split()
            else:
                tokens = tokens.split()
        elif name == 'psn':
            if '.|' in tokens:
                tokens += '|'
                tokens = tokens.replace('.|', ' . ').split()
            else:
                tokens = tokens.split()
            #tokens = [w for w in tokens if w not in self.stopword]
        else:
            tokens = tokens.strip(punctuation).strip().split()
            tokens.insert(0, '<start>')
            tokens.append('__eou__')
        sequence = []
        #sequence.extend([word2id[token] if token in word2id else word2id['<UNK>'] for token in tokens])
        sequence.extend([word2id[token] for token in tokens])
        return sequence

    def preprocess_tree(self, tree):
        words = ['<start>']
        tags = []
        for node in PreOrderIter(tree):
            '''
            if node.is_leaf:
                tag, word, _, _ = node.name.split('__')
                words.append(word)
                tags.append(tag)
            '''
            nt, word, tag, rule = node.name.split('__')
            ind = int(rule.split()[-1][1:-1])
            node.ind = ind 
            if node.parent is None:
                words.append(word)
                tags.append(tag)
            elif node.parent.children.index(node) != node.parent.ind:
                words.append(word)
                tags.append(tag)
        words.append('__eou__')
        tags.append('.')
        return [self.word2id[w] for w in words], [self.nt2id[nt] for nt in tags]

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).

    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.

    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if end:
                padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lengths

    '''
    trg_data = list(zip(trg, [*range(len(trg))]))
    trg_data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    #src_seqs, trg_seqs = zip(*data)
    ctc_seqs, ctc_indices = zip(*ctc_data)
    trg_seqs, trg_indices = zip(*trg_data)
    '''

    src_seqs, trg_seqs, psn_seqs, pos_seqs = zip(*data)
    data = list(zip(src_seqs, trg_seqs, psn_seqs, pos_seqs, [*range(len(data))]))
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs, psn_seqs, pos_seqs, indices = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)
    psn_seqs, psn_lengths = merge(psn_seqs)
    pos_seqs, pos_lengths = merge(pos_seqs)

    return src_seqs, src_lengths, trg_seqs, trg_lengths, psn_seqs, psn_lengths, indices, pos_seqs


def get_loader(src_path, trg_path, psn_path, word2id, batch_size=100, shuffle=True):
    """Returns data loader for custom dataset.

    Args:
        src_path: txt file path for source domain.
        trg_path: txt file path for target domain.
        src_word2id: word-to-id dictionary (source domain).
        trg_word2id: word-to-id dictionary (target domain).
        batch_size: mini-batch size.

    Returns:
        data_loader: data loader for custom dataset.
    """
    # build a custom dataset
    dataset = Dataset(src_path, trg_path, psn_path, word2id)

    # data loader for custome dataset
    # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
    # please see collate_fn for details
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)

    return data_loader
