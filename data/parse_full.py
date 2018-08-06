from stanfordcorenlp import StanfordCoreNLP
import pdb
import sys
import pickle
import json
import nltk
from anytree import Node


nlp = StanfordCoreNLP(r'../../stanford-corenlp-full-2018-02-27')

def convert(node, l):
    ch = node.children
    if ch:
        if len(ch) == 1 and ch[0].name.find('NT') == -1:
            l.append(ch[0].name)
        else:
            names = ' '.join([n.name for n in ch])
            l.append('RULE: {} -> {}'.format(node.name, names))
            for c in ch:
                convert(c, l)

with open(sys.argv[1]) as infile:
    #with open(sys.argv[2], 'w') as outfile:
    data = []
    last = ''
    for _, line in enumerate(infile):
        if _ % 1000 == 0:
            print(_)
        if sys.argv[1].find('psn') == -1:
            sent = nltk.tokenize.sent_tokenize(line)
            words = []
            seq = []
            stack = []
            root = None
            ind = 0
            node2id = {}
            leaves = []
            tokens = []
            for s in sent:
                parse = nlp.parse(s)
                for token in parse.split()[1:]:
                    if token[0] == '(':
                        seq.append('NT-{}'.format(token[1:]))
                        if stack:
                            node = Node(seq[-1], parent=stack[-1])
                        else:
                            node = Node(seq[-1])
                        stack.append(node)
                        #node2id[node] = ind 
                        #ind += 1
                        if root is None:
                            root = node
                    else:
                        first = token.find(')')
                        #ind -= 1
                        #stack.pop()
                        seq.append('GEN-{}'.format(token[:first]))
                        if stack:
                            child = Node(token[:first], parent=stack[-1])
                        else:
                            child = Node(token[:first])
                        leaves.append(child)
                        #node2id[child] = ind 
                        #ind += 1
                        words.append(token[:first])
                        seq.extend(['REDUCE'] * (len(token) - first)) 
                        #ind += len(token) - first - 1 
                        for x in range(len(token) - first): 
                            if stack:
                                stack.pop()
                convert(root, tokens)
                root = None
            data.append({'words': words, 'parse': tokens})
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

