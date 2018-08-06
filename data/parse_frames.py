from stanfordcorenlp import StanfordCoreNLP
import pdb
import sys
import pickle
import json
import nltk
from anytree import Node


nlp = StanfordCoreNLP(r'../../stanford-corenlp-full-2018-02-27')

def common_ancestor(root, n1, n2):
    p1 = []
    p2 = []
    p = n1
    p1.append(p)
    while p.parent is not None:
        p = p.parent
        p1.append(p)
    p = n2
    p2.append(p)
    while p.parent is not None:
        p = p.parent
        p2.append(p)
    (l1, l2) = (p1, p2) if len(p1) > len(p2) else (p2, p1)
    for a in l1:
        if a in l2:
            return a
    return None

source = pickle.load(open(sys.argv[1], 'rb'))
data = []
for __, chat in enumerate(source):
    if __ % 1 == 0:
        print(__)
    post_chat = []
    for _, utt in enumerate(chat):
        words = []
        seq = []
        leaves = []
        if _ % 2 == 1:
            span = utt['span']
        start = 0
        change = []
        node2id = {}
        ind = 0

        #for s, sp in enumerate(span):
        for s in range(len(utt['acts'])):
            for name, val in utt['acts'][s]['vals']:
                if val is not None:
                    utt['text'] = utt['text'].replace(' {} '.format(val), ' {} '.format(name.upper()))

        sent = nltk.tokenize.sent_tokenize(utt['text'])
        for s in sent:
            parse = nlp.parse(s)
            stack = []
            root = None
            for token in parse.split()[1:]:
                if token[0] == '(':
                    seq.append('NT-{}'.format(token[1:]))
                    if stack:
                        node = Node(seq[-1], parent=stack[-1])
                    else:
                        node = Node(seq[-1])
                    stack.append(node)
                    node2id[node] = ind
                    ind += 1
                    if root is None:
                        root = node
                else:
                    seq.pop() # ignore POS tags
                    ind -= 1
                    stack.pop()
                    first = token.find(')')
                    seq.append('GEN-{}'.format(token[:first]))
                    if stack:
                        child = Node(token[:first], parent=stack[-1])
                    else:
                        child = Node(token[:first])
                    leaves.append(child)
                    node2id[child] = ind
                    ind += 1
                    words.append(token[:first])
                    seq.extend(['REDUCE'] * (len(token) - first - 1))
                    ind += len(token) - first - 1
                    for x in range(len(token) - first - 1):
                        if stack:
                            stack.pop()
            seq.pop()
            ind -= 1
            if _ % 2 == 1:
                end = len(words) - 1
                for s, sp in enumerate(span):
                    b = max(start, sp[0])
                    e = min(end, sp[1])
                    if e > b:
                        common = common_ancestor(root, leaves[b], leaves[e])
                        index = node2id[common]
                        change.append((index, utt['acts'][s]['name']))
                        assert seq[index] == common.name
                start = end + 1
        for i, a in change:
            seq[i] = 'NT-{}'.format(a)
        post_chat.append({'words': words, 'parse': seq})
        if _ % 2 == 1:
            post_chat[-1]['acts'] = utt['acts']
    data.append(post_chat)
        

def write(data, split):
    src_file = open('frames.{}.src'.format(split), 'w')
    trg_file = open('frames.{}.trg'.format(split), 'w')
    context = []
    trg = []
    for chat in data:
        if len(chat) % 2 > 0:
            chat = chat[:-1]
        for x, utt in enumerate(chat):
            context.append(' '.join(utt['words']).strip('\n'))
            if len(context) > 10:
                context = context[-9:]
            if x % 2 == 0:
                src_file.write(' __eou__ '.join(context) + ' __eou__')
                src_file.write('\n')
            else:
                trg_file.write(' '.join(utt['words']) + ' __eou__\n')
                trg.append(utt)
    pickle.dump(trg, open('frames.{}.parse.trg'.format(split), 'wb'))

write(data[:1000], 'train')
write(data[1000:1100], 'valid')
write(data[1100:], 'test')

