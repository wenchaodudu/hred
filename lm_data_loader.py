import nltk
import json
import torch
import torch.utils.data as data
import pdb
from stanfordcorenlp import StanfordCoreNLP
import json
import pickle
import numpy as np
from ast import literal_eval
from copy import deepcopy


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, parse_path, dictionary):
        """Reads source and target sequences from txt files."""
        parse_file = open(parse_path, 'rb')
        parse_file.seek(0)
        parse_file = pickle.load(parse_file)
        self.num_total_seqs = len(parse_file)
        self.trg_seqs = [None for x in range(self.num_total_seqs)]
        self.parent_seqs = [None for x in range(self.num_total_seqs)]
        self.sibling_seqs = [None for x in range(self.num_total_seqs)]
        self.leaf_seqs = [None for x in range(self.num_total_seqs)]
        self.lex_seqs = [None for x in range(self.num_total_seqs)]
        self.leaf_indices = [None for x in range(self.num_total_seqs)]
        self.lex_indices = [None for x in range(self.num_total_seqs)]
        self.rule_seqs = [None for x in range(self.num_total_seqs)]
        self.word_mask = [None for x in range(self.num_total_seqs)]
        self.rule_mask = [None for x in range(self.num_total_seqs)]
        self.positions = [None for x in range(self.num_total_seqs)]
        self.ancestors = [None for x in range(self.num_total_seqs)]
        self.max_len = 50
        self.word_dict = dictionary['word']
        self.rule_dict = dictionary['rule']
        self.nt_dict = dictionary['const']
        for x in range(self.num_total_seqs):
            if x % 10000 == 0:
                print(x)
            trg_seqs = []
            parent_seqs = [[], []]
            sibling_seqs = [[], []]
            leaf_seqs = []
            rule_seqs = []
            word_mask = []
            rule_mask = []
            positions = []
            ancestors = []
            tree_dict = {}
            self.load(parse_file[x], trg_seqs, parent_seqs, sibling_seqs, leaf_seqs, rule_seqs, word_mask, rule_mask, tree_dict)
            self.load_positions(parse_file[x], np.cumsum(leaf_seqs), positions)
            self.load_ancestors(parse_file[x], ancestors, tree_dict)
            #self.src_seqs[x] = self.preprocess(self.src_seqs[x], self.word_dict)
            #self.psn_seqs[x] = self.preprocess(self.psn_seqs[x], self.word_dict)
            self.trg_seqs[x] = trg_seqs
            self.parent_seqs[x] = parent_seqs
            self.sibling_seqs[x] = sibling_seqs
            self.rule_seqs[x] = rule_seqs
            self.word_mask[x] = word_mask
            self.rule_mask[x] = rule_mask
            self.positions[x] = positions
            self.ancestors[x] = ancestors
            '''
            self.leaf_indices[x] = deepcopy(leaf_seqs)
            self.lex_indices[x] = word_mask
            '''
            self.leaf_indices[x] = np.cumsum(leaf_seqs)
            self.lex_indices[x] = np.cumsum(word_mask)
            self.leaf_seqs[x] = self.load_leaf(leaf_seqs, trg_seqs)
            self.lex_seqs[x] = self.load_leaf(deepcopy(word_mask), trg_seqs)
            
    def __getitem__(self, x):
        """Returns one data pair (source and target)."""
        return self.trg_seqs[x], \
               self.parent_seqs[x], self.sibling_seqs[x], self.leaf_seqs[x], self.lex_seqs[x], self.rule_seqs[x], \
               self.leaf_indices[x], self.lex_indices[x], \
               self.word_mask[x], self.rule_mask[x], \
               self.positions[x], self.ancestors[x]

    def __len__(self):
        return self.num_total_seqs

    def load_dict(self, tree, d):
        d[tree] = len(d)
        for ch in tree.children:
            self.load_dict(ch, d)

    def load_ancestors(self, tree, ancestors, tree_dict):
        if tree.is_leaf:
            anc = tree.ancestors
            anc_ind = [tree_dict[a] for a in anc]
            ancestors.append(anc_ind)
        for ch in tree.children:
            self.load_ancestors(ch, ancestors, tree_dict)

    def load_leaf(self, leaf_seqs, trg_seqs):
        res = [None for x in leaf_seqs]
        first = -1
        second = -1
        third = -1
        for x in range(len(leaf_seqs)):
            first_leaf = (0, 0, 0) if first == -1 else trg_seqs[first]
            second_leaf = (0, 0, 0) if second == -1 else trg_seqs[second]
            third_leaf = (0, 0, 0) if third == -1 else trg_seqs[third]
            leaf_seqs[x] = [first_leaf, second_leaf, third_leaf]
            if leaf_seqs[x] == 1:
                third = second
                second = first
                first = x
        return leaf_seqs

    def load_positions(self, tree, leaf_seqs, positions):
        if positions:
            positions.append((tree.depth, leaf_seqs[len(positions)-1].item()))
        else:
            positions.append((tree.depth, 0))
        for ch in tree.children:
            self.load_positions(ch, leaf_seqs, positions)

    def load(self, tree, trg_seqs, parent_seqs, sibling_seqs, leaf_seqs, rule_seqs, word_mask, rule_mask, par_dict):
        rule_ind = 0
        word_ind = 0
        if tree.is_leaf:
            nt, word, tag = tree.name.split('__')
            word = word.lower()
            rule_seqs.append(0)
            rule_mask.append(0)
            leaf_seqs.append(1)
        else:
            nt, word, tag, rule = tree.name.split('__')
            word = word.lower()
            word_ind = self.word_dict[word] if word in self.word_dict else self.word_dict['<UNK>']
            inh = literal_eval(rule[rule.find('['):rule.find(']')+1])[0]
            rule = rule[:rule.find('[')-1]
            tag = rule.split()[inh + 1]
            rule_ind = self.rule_dict[rule] if rule in self.rule_dict else self.rule_dict['<UNK>']
            rule_seqs.append(rule_ind)
            rule_mask.append(1)
            leaf_seqs.append(0)
        '''
        par_rule_ind = 0
        if tree.parent is not None:
            _, _, _, par_rule = tree.parent.name.split('__')
            par_rule = par_rule[:par_rule.find('[')-1]
            par_rule_ind = self.rule_dict[par_rule]
        '''
        trg_seqs.append((self.nt_dict[nt], word_ind, self.nt_dict[tag], rule_ind))
        '''
        if tree.parent is not None:
            parent_seqs[0].append(par_dict[tree.parent])
            if tree.parent.parent is not None:
                parent_seqs[1].append(par_dict[tree.parent.parent])
                pos = tree.parent.parent.children.index(tree.parent)
                if pos > 0:
                    sibling_seqs[1].append(par_dict[tree.parent.parent.children[pos-1]]) 
                else:
                    sibling_seqs[1].append((0, 0, 0))
            else:
                parent_seqs[1].append((0, 0, 0))
                sibling_seqs[1].append((0, 0, 0))
            pos = tree.parent.children.index(tree)
            if pos > 0:
                sibling_seqs[0].append(par_dict[tree.parent.children[pos-1]]) 
            else:
                sibling_seqs[0].append((0, 0, 0))
        else:
            parent_seqs[0].append((0, 0, 0))
            sibling_seqs[0].append((0, 0, 0))
            parent_seqs[1].append((0, 0, 0))
            sibling_seqs[1].append((0, 0, 0))
        '''
        anc = tree.ancestors
        '''
        if len(anc) >= 3:
            parent_seqs[0].append(par_dict[anc[1]])
            parent_seqs[1].append(par_dict[anc[2]])
        elif len(anc) == 2:
            parent_seqs[0].append(par_dict[anc[1]])
            parent_seqs[1].append((0, 0, 0))
        else:
            parent_seqs[0].append((0, 0, 0))
            parent_seqs[1].append((0, 0, 0))
        '''
        parent_seqs[0].append((0, 0, 0))
        parent_seqs[1].append((0, 0, 0))
        sibling_seqs[0].append((0, 0, 0))
        sibling_seqs[1].append((0, 0, 0))
        par_dict[tree] = trg_seqs[-1]
        if tree.parent:
            ind = tree.parent.children.index(tree)
            openb = tree.parent.name.rindex('[')
            closeb = tree.parent.name.rindex(']')
            inds = literal_eval(tree.parent.name[openb:closeb+1])
            if ind in inds:
                word_mask.append(0)
            else:
                word_mask.append(1)
        else:
            word_mask.append(1)
        for ch in tree.children:
            self.load(ch, trg_seqs, parent_seqs, sibling_seqs, leaf_seqs, rule_seqs, word_mask, rule_mask, par_dict)

    def preprocess(self, sequence, word2id):
        """Converts words to ids."""
        tokens = sequence.strip().lower().replace('__eou__', ' ').replace('|', ' ').replace('.', ' .').split()[-self.max_len:]
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
    def merge(sequences, lengths=None):
        if lengths is None:
            lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if end:
                padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lengths

    def extract(trg_seqs, x):
        return [[step[x] for step in trg] for trg in trg_seqs]

    '''
    trg_data = list(zip(trg, [*range(len(trg))]))
    trg_data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    #src_seqs, trg_seqs = zip(*data)
    ctc_seqs, ctc_indices = zip(*ctc_data)
    trg_seqs, trg_indices = zip(*trg_data)
    '''

    trg_seqs, par_seqs, sib_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, word_mask, rule_mask, positions, ancestors = zip(*data)
    data = list(zip(trg_seqs, par_seqs, sib_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, word_mask, rule_mask, positions, ancestors, [*range(len(data))]))
    data.sort(key=lambda x: len(x[0]), reverse=True)
    trg_seqs, par_seqs, sib_seqs, leaf_seqs, lex_seqs, rule_seqs, leaf_indices, lex_indices, word_mask, rule_mask, positions, ancestors, indices = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    rule_seqs, trg_lengths = merge(rule_seqs)
    word_mask, trg_lengths = merge(word_mask, trg_lengths)
    rule_mask, trg_lengths = merge(rule_mask, trg_lengths)
    ancestors = [anc for lst in ancestors for anc in lst]
    anc_lex = [[x[1] for x in anc] for anc in ancestors]
    anc_nt = [[x[0] for x in anc] for anc in ancestors]
    anc_rule = [[x[3] for x in anc] for anc in ancestors]
    anc_lex, anc_lengths = merge(anc_lex)
    anc_nt, anc_lengths = merge(anc_nt, anc_lengths)
    anc_rule, anc_lengths = merge(anc_rule, anc_lengths)
    ancestors = (anc_lex, anc_nt, anc_rule)
    _trg_seqs = [None, None, None]
    parent_seqs = [[None, None, None], [None, None, None]]
    leaves_seqs = [[None, None, None], [None, None, None], [None, None, None]]
    lexes_seqs = [[None, None, None], [None, None, None], [None, None, None]]
    sibling_seqs = [[None, None, None], [None, None, None]]
    leaf_seqs = [[[x[2] for x in seq], [x[1] for x in seq], [x[0] for x in seq]] for seq in leaf_seqs]
    lex_seqs = [[[x[2] for x in seq], [x[1] for x in seq], [x[0] for x in seq]] for seq in lex_seqs]
    for x in range(3):
        _trg_seqs[x], _ = merge(extract(trg_seqs, x), trg_lengths)
        parent_seqs[0][x], _ = merge(extract([par[0] for par in par_seqs], x), trg_lengths)
        parent_seqs[1][x], _ = merge(extract([par[1] for par in par_seqs], x), trg_lengths)
        leaves_seqs[0][x], _ = merge(extract([leaf[0] for leaf in leaf_seqs], x), trg_lengths)
        leaves_seqs[1][x], _ = merge(extract([leaf[1] for leaf in leaf_seqs], x), trg_lengths)
        leaves_seqs[2][x], _ = merge(extract([leaf[2] for leaf in leaf_seqs], x), trg_lengths)
        lexes_seqs[0][x], _ = merge(extract([leaf[0] for leaf in lex_seqs], x), trg_lengths)
        lexes_seqs[1][x], _ = merge(extract([leaf[1] for leaf in lex_seqs], x), trg_lengths)
        lexes_seqs[2][x], _ = merge(extract([leaf[2] for leaf in lex_seqs], x), trg_lengths)
        sibling_seqs[0][x], _ = merge(extract([sib[0] for sib in sib_seqs], x), trg_lengths)
        sibling_seqs[1][x], _ = merge(extract([sib[1] for sib in sib_seqs], x), trg_lengths)

    pos = [[], []]
    for p in positions:
        pos[0].append([i[0] for i in p])
        pos[1].append([i[1] for i in p])
    pos[0] = merge(pos[0])[0]
    pos[1] = merge(pos[1])[0]

    return indices, _trg_seqs, trg_lengths, \
           parent_seqs, sibling_seqs, leaves_seqs, lexes_seqs, rule_seqs, \
           leaf_indices, lex_indices, \
           word_mask, rule_mask, \
           pos, ancestors, anc_lengths


def get_loader(parse_path, word2id, batch_size=100, shuffle=True):
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
    dataset = Dataset(parse_path, word2id)

    # data loader for custome dataset
    # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
    # please see collate_fn for details
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)

    return data_loader
