import pdb
import sys
import pickle
import json
import re
import numpy as np
from anytree import Node


count = 0
def convert(node):
    global count
    children = node.children
    if children:
        names = [ch.name.split('__') for ch in children]
        words = [n[1] for n in names]
        tags = [n[2] for n in names]
        nt, w, pt = node.name.split('__')
        ch_name = ' '.join([n[0] for n in names])
        inds = [x for x in range(len(names)) if words[x] == w and tags[x] == pt]
        if len(inds) >= 1:
            rule = '__RULE: {} {}'.format(ch_name, inds)
            node.name += rule
            if len(inds) >= 2:
                count += 1
        else:
            pdb.set_trace()
        for ch in children:
            convert(ch)

with open(sys.argv[1]) as infile:
    #with open(sys.argv[2], 'w') as outfile:
    data = []
    last = ''
    for _, line in enumerate(infile):
        if _ % 1000 == 0:
            print(_)
        if sys.argv[1].find('psn') == -1:
            words = []
            seq = []
            stack = []
            root = None
            ind = 0
            node2id = {}
            leaves = []
            tokens = []
            parse = line.strip('\n')
            for token in parse.split():
                if token[0] == '(':
                    nt, word, tag, __ = re.split(r'[\[/\]]', token[1:])
                    name = '{}__{}__{}'.format(nt, word, tag)
                    if stack:
                        node = Node(name, parent=stack[-1])
                    else:
                        node = Node(name)
                    stack.append(node)
                    if root is None:
                        root = node
                else:
                    first = token.find(')')
                    '''
                    seq.append('GEN-{}'.format(token[:first]))
                    if stack:
                        child = Node(token[:first], parent=stack[-1])
                    else:
                        child = Node(token[:first])
                    '''
                    leaves.append(stack[-1])
                    words.append(token[:first])
                    for x in range(len(token) - first): 
                        stack.pop()
            convert(root)
            data.append(root)
        else:
            if line == last:
                data.append(data[-1])
            else:
                pers = []
                persona = line.split('|')
                for s in persona:
                    words = []
                    seq = []
                    parse = nlp.parse(s)
                    for token in parse.split()[1:]:
                        if token[0] == '(':
                            seq.append('NT-{}'.format(token[1:]))
                        else:
                            first = token.find(')')
                            seq.append('GEN-{}'.format(token[:first]))
                            words.append(token[:first])
                            seq.extend(['REDUCE'] * (len(token) - first - 1))
                    seq.pop()
                    pers.append({'words': words, 'parse': seq})
                data.append(pers)
            last = line
    pickle.dump(data, open(sys.argv[2], 'wb'))


print(count)
