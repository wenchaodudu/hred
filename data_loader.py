import nltk
import json
import torch
import torch.utils.data as data
import pdb


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, src_path, trg_path, psn_path, word2id):
        """Reads source and target sequences from txt files."""
        self.src_seqs = open(src_path).readlines()
        self.trg_seqs = open(trg_path).readlines()
        self.psn_seqs = open(psn_path).readlines()
        self.num_total_seqs = len(self.src_seqs)
        self.max_len = 100
        self.word2id = word2id
        for x in range(self.num_total_seqs):
            if x % 100000 == 0:
                print(x)
            self.src_seqs[x] = self.preprocess(self.src_seqs[x], self.word2id)
            self.trg_seqs[x] = self.preprocess(self.trg_seqs[x], self.word2id)
            self.psn_seqs[x] = self.preprocess(self.psn_seqs[x], self.word2id)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        psn_seq = self.psn_seqs[index]
        return src_seq, trg_seq, psn_seq

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id):
        """Converts words to ids."""
        tokens = sequence.strip().lower().replace('|', ' ').replace('__eou__', ' ').replace('.', ' .')
        tokens = tokens.split()[-self.max_len:]
        sequence = []
        sequence.append(word2id['<start>'])
        sequence.extend([word2id[token] if token in word2id else word2id['<UNK>'] for token in tokens])
        sequence.append(word2id['__eou__'])
        return sequence


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

    src_seqs, trg_seqs, psn_seqs = zip(*data)
    data = list(zip(src_seqs, trg_seqs, psn_seqs, [*range(len(data))]))
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs, psn_seqs, indices = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)
    psn_seqs, psn_lengths = merge(psn_seqs)

    return src_seqs, src_lengths, trg_seqs, trg_lengths, psn_seqs, psn_lengths, indices


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
