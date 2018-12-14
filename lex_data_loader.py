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
from anytree import Node, PreOrderIter


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, src_path, parse_path, trg_path, psn_path, dictionary):
        """Reads source and target sequences from txt files."""
        parse_file = open(parse_path, 'rb')
        parse_file.seek(0)
        parse_file = pickle.load(parse_file)
        self.num_total_seqs = len(parse_file)
        if psn_path is not None:
            self.src_seqs = open(src_path).readlines()
            self.psn_seqs = open(psn_path).readlines()
            self.data = 'persona'
        else:
            src_parse_file = open(src_path, 'rb')
            src_parse_file.seek(0)
            src_parse_file = pickle.load(src_parse_file)
            self.src_seqs = [None for x in range(self.num_total_seqs)]
            self.psn_seqs = [None for x in range(self.num_total_seqs)]
            self.data = 'microsoft'
        #self.trg_seqs = [None for x in range(self.num_total_seqs)]
        self.trg_seqs = open(trg_path).readlines()
        self.parent_seqs = [None for x in range(self.num_total_seqs)]
        self.sibling_seqs = [None for x in range(self.num_total_seqs)]
        self.leaf_seqs = [None for x in range(self.num_total_seqs)]
        self.lex_seqs = [None for x in range(self.num_total_seqs)]
        self.leaf_indices = [None for x in range(self.num_total_seqs)]
        self.lex_indices = [None for x in range(self.num_total_seqs)]
        self.rule_seqs = [None for x in range(self.num_total_seqs)]
        self.word_mask = [None for x in range(self.num_total_seqs)]
        self.noun_mask = [None for x in range(self.num_total_seqs)]
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
            leaf_seqs = []
            lex_seqs = []
            rule_seqs = []
            word_mask = []
            rule_mask = []
            positions = []
            ancestors = []
            tree_dict = {}
            noun_mask = []
            #self.load(parse_file[x], trg_seqs, parent_seqs, sibling_seqs, leaf_seqs, rule_seqs, word_mask, rule_mask, tree_dict)
            self.load(parse_file[x], trg_seqs, leaf_seqs, rule_seqs, lex_seqs, word_mask, rule_mask, noun_mask, tree_dict)
            self.load_positions(parse_file[x], np.cumsum(leaf_seqs), positions)
            self.load_ancestors(parse_file[x], ancestors, tree_dict)
            if psn_path:
                self.src_seqs[x] = self.preprocess(self.src_seqs[x], self.word_dict, 'src')
                self.psn_seqs[x] = self.preprocess(self.psn_seqs[x], self.word_dict, 'psn')
                self.src_seqs[x] = self.psn_seqs[x] + self.src_seqs[x]
                words, tree = self.preprocess_tree(parse_file[x], self.word_dict, self.nt_dict, self.rule_dict)
                #words = self.preprocess(self.trg_seqs[x], self.word_dict, 'trg')
                words.append(self.word_dict['__eou__'])
                self.psn_seqs[x] = words
            else:
                words, tree = self.preprocess_tree(src_parse_file[x], self.word_dict, self.nt_dict, self.rule_dict)
                self.src_seqs[x] = words
                self.psn_seqs[x] = tree
            self.trg_seqs[x] = trg_seqs
            self.rule_seqs[x] = rule_seqs
            self.word_mask[x] = word_mask
            self.rule_mask[x] = rule_mask
            self.positions[x] = positions
            self.ancestors[x] = ancestors
            self.noun_mask[x] = noun_mask
            '''
            self.leaf_indices[x] = deepcopy(leaf_seqs)
            self.lex_indices[x] = word_mask
            '''
            self.leaf_indices[x] = np.cumsum(leaf_seqs)
            self.lex_indices[x] = np.cumsum(word_mask)
            '''
            self.leaf_seqs[x] = self.load_leaf(leaf_seqs, trg_seqs)
            self.lex_seqs[x] = self.load_leaf(deepcopy(word_mask), trg_seqs)
            '''
            self.lex_seqs[x] = lex_seqs
            
    def __getitem__(self, x):
        """Returns one data pair (source and target)."""
        return self.src_seqs[x], self.trg_seqs[x], self.psn_seqs[x], \
               self.rule_seqs[x], self.lex_seqs[x], \
               self.leaf_indices[x], self.lex_indices[x], \
               self.word_mask[x], self.rule_mask[x], self.noun_mask[x], \
               self.positions[x], self.ancestors[x]

    def __len__(self):
        return self.num_total_seqs

    def load_dict(self, tree, d):
        d[tree] = len(d)
        for ch in tree.children:
            self.load_dict(ch, d)

    def load_ancestors(self, tree, ancestors, tree_dict):
        anc = tree.ancestors
        anc_ind = []
        nt = tree_dict[tree][0]
        for a in reversed(anc):
            d = tree_dict[a]
            new_nt = d[0]
            # use constituent of current node
            d = (nt, d[1], d[2], d[3])
            if a.parent is None or a.parent.children.index(a) < len(a.parent.children) - 1:
                anc_ind.append(d)
            else:
                dd = (d[0], -1, d[2], d[3])
                anc_ind.append(dd)
            nt = new_nt
        ancestors.append(list(reversed(anc_ind)))
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

    def load(self, parse_tree, trg_seqs, leaf_seqs, rule_seqs, lex_seqs, word_mask, rule_mask, noun_mask, par_dict):
        last_word = None
        for tree in PreOrderIter(parse_tree):
            rule_ind = 0
            if tree.is_leaf:
                nt, word, tag, rule = tree.name.split('__')
                #rule_seqs.append(self.rule_dict['RULE: EOD'])
                rule_mask.append(1)
                leaf_seqs.append(1)
                rule = rule[:rule.find('[')-1]
                rule_ind = self.rule_dict[rule]
            else:
                nt, word, tag, rule = tree.name.split('__')
                inh = literal_eval(rule[rule.find('['):rule.find(']')+1])[0]
                rule = rule[:rule.find('[')-1]
                rule_ind = self.rule_dict[rule] if rule in self.rule_dict else self.rule_dict['<UNK>']
                tag = rule.split()[inh + 1]
                #rule_seqs.append(rule_ind)
                rule_mask.append(1)
                leaf_seqs.append(0)
            
            trg_seqs.append((self.nt_dict[nt], self.word_dict[word], self.nt_dict[nt], rule_ind))
            if tree.parent is None:
                rule_seqs.append(0)
                par_lex = 0
            else:
                par_nt, par_word, par_tag, par_rule = tree.parent.name.split('__')
                par_rule = par_rule[:par_rule.find('[')-1]
                par_lex = self.word_dict[par_word]
                if par_rule in self.rule_dict:
                    rule_seqs.append(self.rule_dict[par_rule])
                else:
                    rule_seqs.append(self.rule_dict['<UNK>'])
            if last_word is not None:
                lex_seqs.append((self.word_dict[last_word], par_lex))
            last_word = word
            
            anc = tree.ancestors
            par_dict[tree] = trg_seqs[-1]
            if tree.parent:
                ind = tree.parent.children.index(tree)
                openb = tree.parent.name.find('[')
                closeb = tree.parent.name.find(']')
                inds = literal_eval(tree.parent.name[openb:closeb+1])
                if ind in inds:
                    word_mask.append(0)
                    noun_mask.append(0)
                else:
                    word_mask.append(1)
                    if nt[:2] == 'NN':
                        noun_mask.append(1)
                    else:
                        noun_mask.append(0)
            else:
                word_mask.append(1)
                if nt[:2] == 'NN':
                    noun_mask.append(1)
                else:
                    noun_mask.append(0)

    def preprocess(self, sequence, word2id, name):
        """Converts words to ids."""
        if self.data == 'persona':
            tokens = sequence.strip().lower()
            if name == 'src':
                if '__eou__' in tokens:
                    tokens = tokens.split('__eou__')[-2].split()
                else:
                    tokens = tokens.split()
            elif name == 'trg':
                tokens = tokens.split()
            else:
                if '.|' in tokens:
                    tokens += '|'
                    tokens = tokens.replace('.|', ' . ').split()
                else:
                    tokens = tokens.split()
            sequence = []
            #sequence.extend([word2id[token] if token in word2id else word2id['<UNK>'] for token in tokens])
            sequence.extend([word2id[token] for token in tokens])
            return sequence
        else:
            tokens = sequence.strip().lower().split()
            sequence = []
            sequence.append(word2id['<start>'])
            sequence.extend([word2id[token] for token in tokens])
            sequence.append(word2id['__eou__'])
            return sequence

    def preprocess_tree(self, tree, word_dict, nt_dict, rule_dict):
        words = []
        nts = []
        rules = []
        wwords = []
        for node in PreOrderIter(tree):
            if node.is_leaf:
                nt, word, tag, rule = node.name.split('__')
                rule = rule[:rule.find('[')-1]
                rules.append(rule_dict[rule])
            else:
                nt, word, tag, rule = node.name.split('__')
                rule = rule[:rule.find('[')-1]
                if rule in rule_dict:
                    rules.append(rule_dict[rule])
                else:
                    rules.append(rule_dict['<UNK>'])
            nts.append(nt_dict[nt])
            if node.is_leaf:
                words.append(word_dict[word])
                wwords.append(word)
        return words, list(zip(nts, rules))


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

    src_seqs, trg_seqs, psn_seqs, rule_seqs, lex_seqs, leaf_indices, lex_indices, word_mask, rule_mask, node_mask, positions, ancestors = zip(*data)
    data = list(zip(src_seqs, trg_seqs, psn_seqs, rule_seqs, lex_seqs, leaf_indices, lex_indices, word_mask, rule_mask, node_mask, positions, ancestors, [*range(len(data))]))
    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_seqs, trg_seqs, psn_seqs, rule_seqs, lex_seqs, leaf_indices, lex_indices, word_mask, rule_mask, node_mask, positions, ancestors, indices = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    if isinstance(psn_seqs[0][0], tuple):
        psn_nt_seqs = extract(psn_seqs, 0)
        psn_rule_seqs = extract(psn_seqs, 1)
        psn_nt_seqs, psn_lengths = merge(psn_nt_seqs)
        psn_rule_seqs, psn_lengths = merge(psn_rule_seqs, psn_lengths)
        psn_seqs = (psn_nt_seqs, psn_rule_seqs)
    else:
        psn_seqs, psn_lengths = merge(psn_seqs)
    rule_seqs, trg_lengths = merge(rule_seqs)
    '''
    lex_seqs_1, _ = merge(lex_seqs[0])
    lex_seqs_2, _ = merge(lex_seqs[1])
    lex_seqs = (lex_seqs_1, lex_seqs_2)
    '''
    lex_seqs = None
    word_mask, trg_lengths = merge(word_mask, trg_lengths)
    rule_mask, trg_lengths = merge(rule_mask, trg_lengths)
    node_mask, trg_lengths = merge(node_mask, trg_lengths)
    ancestors = [anc for lst in ancestors for anc in lst]
    anc_lex = [[x[1] for x in anc if x[1] != -1] for anc in ancestors]
    anc_nt = [[x[0] for x in anc] for anc in ancestors]
    anc_rule = [[x[3] for x in anc] for anc in ancestors]
    anc_lex, anc_lengths = merge(anc_lex)
    anc_nt, anc_lengths = merge(anc_nt, anc_lengths)
    anc_rule, anc_lengths = merge(anc_rule, anc_lengths)
    ancestors = (anc_lex, anc_nt, anc_rule)
    _trg_seqs = [None, None, None, None]
    '''
    parent_seqs = [[None, None, None], [None, None, None]]
    leaves_seqs = [[None, None, None], [None, None, None], [None, None, None]]
    lexes_seqs = [[None, None, None], [None, None, None], [None, None, None]]
    sibling_seqs = [[None, None, None], [None, None, None]]
    leaf_seqs = [[[x[2] for x in seq], [x[1] for x in seq], [x[0] for x in seq]] for seq in leaf_seqs]
    lex_seqs = [[[x[2] for x in seq], [x[1] for x in seq], [x[0] for x in seq]] for seq in lex_seqs]
    '''
    for x in range(4):
        _trg_seqs[x], _ = merge(extract(trg_seqs, x), trg_lengths)
        '''
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
        '''

    pos = [[], []]
    for p in positions:
        pos[0].append([i[0] for i in p])
        pos[1].append([i[1] for i in p])
    pos[0] = merge(pos[0])[0]
    pos[1] = merge(pos[1])[0]

    return src_seqs, src_lengths, indices, _trg_seqs, trg_lengths, \
           psn_seqs, psn_lengths, \
           rule_seqs, lex_seqs, \
           leaf_indices, lex_indices, \
           word_mask, rule_mask, node_mask, \
           pos, ancestors, anc_lengths


def get_loader(src_path, parse_path, trg_path, psn_path, word2id, batch_size=100, shuffle=True):
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
    dataset = Dataset(src_path, parse_path, trg_path, psn_path, word2id)

    # data loader for custome dataset
    # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
    # please see collate_fn for details
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)

    return data_loader
