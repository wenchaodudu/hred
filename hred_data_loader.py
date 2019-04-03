import json
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, src_path, trg_path, word2id):
        """Reads source and target sequences from txt files."""
        self.max_utt_len = 30
        self.max_ctc_len = 150
        self.max_turn = 5
        src_seqs = open(src_path).readlines()
        trg_seqs = open(trg_path).readlines()
        self.num_total_seqs = len(src_seqs)
        self.src_seqs = [None for x in range(self.num_total_seqs)]
        self.trg_seqs = [None for x in range(self.num_total_seqs)]
        self.ctc_seqs = [None for x in range(self.num_total_seqs)]
        print("Preprocessing contexts.")
        for _, line in enumerate(src_seqs):
            if _ % 100000 == 0:
                print(_)
            self.src_seqs[_] = self.preprocess_src(line, word2id)
            self.ctc_seqs[_] = self.preprocess_trg(line, word2id, self.max_ctc_len)
        print("Preprocessing responses.")
        for _, line in enumerate(trg_seqs):
            if _ % 100000 == 0:
                print(_)
            self.trg_seqs[_] = self.preprocess_trg(line, word2id, self.max_utt_len)
        self.word2id = word2id

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        ctc_seq = self.ctc_seqs[index]
        return src_seq, trg_seq, ctc_seq

    def __len__(self):
        return self.num_total_seqs

    def preprocess_src(self, text, word2id):
        """Converts words to ids."""
        utterances = text.split('__eou__')[:-1]
        context = []
        for seq in utterances:
            if seq.strip():
                tokens = seq.split()[:self.max_utt_len]
                sequence = []
                sequence.extend([word2id[token] if token in word2id else word2id['<UNK>'] for token in tokens])
                context.append(sequence)
        if len(context) > self.max_turn:
            return context[-self.max_turn:]
        else:
            context.extend([[]] * (self.max_turn - len(context)))
        return context

    def preprocess_trg(self, text, word2id, max_len):
        tokens = text.split()[:-1][:max_len-2]
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
    def merge(sequences, max_len):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = min(lengths[i], max_len)
            if end:
                padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lengths

    '''
    src, trg, ctc = zip(*data)
    ctc_len = [len(x) for x in src]
    utt_indices = [(x, y) for x in range(len(src)) for y in range(len(src[x]))]
    src_flatten = [utt for context in src for utt in context]
    src_data = list(zip(src_flatten, utt_indices))
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    src_data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    #src_seqs, trg_seqs = zip(*data)
    src_seqs, indices = zip(*src_data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg)

    return src_seqs, src_lengths, indices, trg_seqs, trg_lengths, ctc_len
    '''
    src, trg, ctc = zip(*data)
    turn_len = [sum(1 for y in x if y) for x in src]
    #ctc_utt_indices = [(x, y) for x in range(len(src)) for y in range(len(src[x]))]
    utt_flatten = [utt for context in src for utt in context]
    src_data = list(zip(utt_flatten, [*range(len(utt_flatten))]))
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    src_data.sort(key=lambda x: len(x[0]), reverse=True)

    ctc_data = list(zip(ctc, [*range(len(ctc))]))
    ctc_data.sort(key=lambda x: len(x[0]), reverse=True)
    trg_data = list(zip(trg, [*range(len(trg))]))
    trg_data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    #src_seqs, trg_seqs = zip(*data)
    ctc_seqs, ctc_indices = zip(*ctc_data)
    trg_seqs, trg_indices = zip(*trg_data)

    #src_seqs, trg_seqs = zip(*data)
    src_seqs, indices = zip(*src_data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs, 30)
    ctc_seqs, ctc_lengths = merge(ctc_seqs, 150)
    trg_seqs, trg_lengths = merge(trg_seqs, 30)

    return src_seqs, src_lengths, indices, ctc_seqs, ctc_lengths, ctc_indices, trg_seqs, trg_lengths, trg_indices, turn_len


def get_hr_loader(src_path, trg_path, word2id, batch_size=100, shuffle=False):
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
    dataset = Dataset(src_path, trg_path, word2id)

    # data loader for custome dataset
    # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
    # please see collate_fn for details
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)

    return data_loader
