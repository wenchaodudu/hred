from stanfordcorenlp import StanfordCoreNLP
import pdb
import sys
import pickle
import json
import nltk


nlp = StanfordCoreNLP(r'../../stanford-corenlp-full-2018-02-27')

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
            for s in sent:
                parse = nlp.parse(s)
                pdb.set_trace()
                for token in parse.split()[1:]:
                    if token[0] == '(':
                        seq.append('NT-{}'.format(token[1:]))
                    else:
                        first = token.find(')')
                        seq.append('GEN-{}'.format(token[:first]))
                        words.append(token[:first])
                        seq.extend(['REDUCE'] * (len(token) - first - 1))
                seq.pop()
            data.append({'words': words, 'parse': seq})
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

