import nltk
import pdb
import json
import numpy as np
import sys
import pickle
from ast import literal_eval
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from stanfordcorenlp import StanfordCoreNLP
from zss import simple_distance, Node
from collections import Counter
from sklearn.utils.extmath import randomized_svd
from string import punctuation


#nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2018-02-27')
dictionary = json.load(open('./persona.lex3.dictionary.json'))['word']
word_vectors = np.zeros((len(dictionary)+1, 300))
word_prob = np.zeros(len(dictionary)+1)
word2vec_file = open('./glove.6B.300d.txt')
word_counter = pickle.load(open('./data/word_count.dat', 'rb'))
total = sum(word_counter.values())
next(word2vec_file)
for line in word2vec_file:
    word, vec = line.split(' ', 1)
    if word in dictionary:
        word_vectors[dictionary[word]] = np.fromstring(vec, dtype=np.float32, sep=' ')
        word_prob[dictionary[word]] = word_counter[word] / total

'''
attn_model = torch.load('attn.persona.pt')
grammar_model = torch.load('grammar.persona.pt')
'''

def build_tree(sentences):
    sent = nltk.tokenize.sent_tokenize(sentences)
    roots = []
    for s in sent:
        parse = nlp.parse(s)
        stack = []
        root = None
        for token in parse.split():
            if token[0] == '(':
                node = Node(token[1:])
                if stack:
                    stack[-1].addkid(node)
                stack.append(node)
                if root is None:
                    root = node
            else:
                first = token.find(')')
                child = Node(token[:first])
                if stack:
                    stack[-1].addkid(child)
                for x in range(len(token) - first):
                    if stack:
                        stack.pop()
        roots.append(root)

    R = Node('R')
    for node in roots:
        R.addkid(node)
    return (R)

#def word_embed_dist(trg_embed, trg_lengths, generations):
def word_embed_dist(hyp, ref, model):
    hyp_embed = model.embed(Variable(hyp.cuda()))
    hyp_embed = hyp_embed / hyp_embed.norm(dim=1).unsqueeze(1)
    ref_embed = model.embed(Variable(ref.cuda()))
    ref_embed = ref_embed / ref_embed.norm(dim=1).unsqueeze(1)
    dots = torch.matmul(hyp_embed, ref_embed.transpose(0, 1))
    dist = dots.sum() / (dots.size(0) * dots.size(1))
    return dist

data = open('generations.txt').readlines()
alt = open(sys.argv[1]).readlines()
#alt = ['i' for x in range(14056)]
unlex = open('constrained.txt').readlines()
sent_vec = np.zeros((3, len(alt), 300))
ind = 0
bleu_score = [0, 0, 0]
rouge_1 = [0, 0, 0]
rouge_2 = [0, 0, 0]
rouge_l = [0, 0, 0]
tree_ed = [0, 0, 0]
word_embed = [0, 0, 0]
lengths = [0, 0, 0, 0]
counters = [Counter(), Counter(), Counter(), Counter()]
bi_counters = [Counter(), Counter(), Counter(), Counter()]
tri_counters = [Counter(), Counter(), Counter(), Counter()]
count = 0
RGE = Rouge()
translator = str.maketrans('', '', punctuation)
while True:
    if ind // 4 >= 14055:
        break
    '''
    if data[ind + 5][0] != '[':
        ind += 5
        continue
    '''
    '''
    if data[ind+2].find('<start>') > -1 or data[ind+2] == '\n':
        ind += 2
        continue
    if data[ind+3].find('<start>') > -1 or data[ind+3] == '\n':
        ind += 2
        continue
    if data[ind+4].find('<start>') > -1 or data[ind+4] == '\n':
        ind += 2
        continue
    '''
    assert data[ind+1].find('<start>') > -1
    target = data[ind + 1]
    target = target.replace('<start>', '')
    #target = target.replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ')
    target = target.translate(translator)
    attn = data[ind + 2]
    #attn = attn.replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ')
    attn = attn.translate(translator)
    unconstrained = alt[ind // 4]
    #unconstrained = unconstrained.replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ')
    unconstrained = unconstrained.translate(translator)
    constrained = unlex[ind // 4]
    #sys.stdout.write(attn)
    '''
    unconstrained = data[ind + 3]
    constrained = data[ind + 4]
    attn = nltk.tokenize.sent_tokenize(attn)[:2]
    attn = ' '.join(attn)
    unconstrained = nltk.tokenize.sent_tokenize(unconstrained)[:2]
    unconstrained = ' '.join(unconstrained)
    constrained = nltk.tokenize.sent_tokenize(constrained)[:2]
    constrained = ' '.join(constrained)
    slots = literal_eval(data[ind + 5])
    for slot, val in slots:
        if val is None:
            continue
        elif val is True:
            constrained = constrained.replace(slot, slot[7:])
            attn += ' {}'.format(slot[7:])
            unconstrained += ' {}'.format(slot[7:])
        else:
            constrained = constrained.replace(slot, str(val))
            attn += ' {}'.format(str(val))
            unconstrained += ' {}'.format(str(val))
    '''
    if count % 1000 == 0:
        print(count)
    count += 1

    bleu_score[0] += sentence_bleu([target], attn)
    bleu_score[1] += sentence_bleu([target], unconstrained)
    bleu_score[2] += sentence_bleu([target], constrained)

    r_score = RGE.get_scores(attn, target)
    rouge_1[0] += r_score[0]['rouge-1']['f']
    rouge_2[0] += r_score[0]['rouge-2']['f']
    rouge_l[0] += r_score[0]['rouge-l']['f']
    r_score = RGE.get_scores(unconstrained, target)
    rouge_1[1] += r_score[0]['rouge-1']['f']
    rouge_2[1] += r_score[0]['rouge-2']['f']
    rouge_l[1] += r_score[0]['rouge-l']['f']
    r_score = RGE.get_scores(constrained, target)
    rouge_1[2] += r_score[0]['rouge-1']['f']
    rouge_2[2] += r_score[0]['rouge-2']['f']
    rouge_l[2] += r_score[0]['rouge-l']['f']

    '''
    target_tree = build_tree(target)
    attn_tree = build_tree(attn)
    unconstrained_tree = build_tree(unconstrained)
    constrained_tree = build_tree(constrained)
    tree_ed[0] += simple_distance(target_tree, attn_tree)
    tree_ed[1] += simple_distance(target_tree, unconstrained_tree)
    tree_ed[2] += simple_distance(target_tree, constrained_tree)
    '''

    '''
    target_word = nlp.word_tokenize(target)
    attn_word = nlp.word_tokenize(attn)
    unconstrained_word = nlp.word_tokenize(unconstrained)
    unlex_word = nlp.word_tokenize(constrained) 
    '''
    target_word = target.translate(translator).split()
    attn_word = attn.translate(translator).split()
    unconstrained_word = unconstrained.translate(translator).split()
    unlex_word = constrained.translate(translator).split()

    target_ind = [dictionary[w] for w in target_word]
    attn_ind = [dictionary[w] for w in attn_word]
    lex_ind = [dictionary[w] for w in unconstrained_word]
    
    a = .001
    sent_vec[0, count-1, :] = np.sum(word_vectors[target_ind] * (a / (a + word_prob[target_ind, np.newaxis])), axis=0) / len(target_word)
    sent_vec[1, count-1, :] = np.sum(word_vectors[attn_ind] * (a / (a + word_prob[attn_ind, np.newaxis])), axis=0) / len(attn_word)
    if len(unconstrained_word):
        sent_vec[2, count-1, :] = np.sum(word_vectors[lex_ind] * (a / (a + word_prob[lex_ind, np.newaxis])), axis=0) / len(unconstrained_word)

    lengths[0] += len(target_word)
    lengths[1] += len(attn_word)
    lengths[2] += len(unconstrained_word)
    lengths[3] += len(unlex_word)
    def n_gram(sent, n):
        return [' '.join(sent[x:x+n]) for x in range(len(sent)-n+1)]
    counters[0].update(target_word)
    counters[1].update(attn_word)
    counters[2].update(unconstrained_word)
    counters[3].update(unlex_word)
    bi_counters[0].update(n_gram(target_word, 2))
    bi_counters[1].update(n_gram(attn_word, 2))
    bi_counters[2].update(n_gram(unconstrained_word, 2))
    tri_counters[0].update(n_gram(target_word, 3))
    tri_counters[1].update(n_gram(attn_word, 3))
    tri_counters[2].update(n_gram(unconstrained_word, 3))
    '''
    target_attn_embed = torch.LongTensor([attn_model.dictionary[word] if word in attn_model.dictionary else 0 for word in nlp.word_tokenize(target)])
    target_grammar_embed = torch.LongTensor([grammar_model.dictionary[word] if word in grammar_model.dictionary else 0 for word in nlp.word_tokenize(target)])
    attn_embed = torch.LongTensor([attn_model.dictionary[word] if word in attn_model.dictionary else 0 for word in nlp.word_tokenize(attn)])
    unconstrained_embed = torch.LongTensor([grammar_model.dictionary[word] if word in grammar_model.dictionary else 0 for word in nlp.word_tokenize(attn)])
    constrained_embed = torch.LongTensor([grammar_model.dictionary[word] if word in grammar_model.dictionary else 0 for word in nlp.word_tokenize(attn)])
    word_embed[0] += word_embed_dist(attn_embed, target_attn_embed, attn_model).data[0]
    word_embed[1] += word_embed_dist(unconstrained_embed, target_grammar_embed, grammar_model).data[0]
    word_embed[2] += word_embed_dist(constrained_embed, target_grammar_embed, grammar_model).data[0]
    '''

    if ind >= len(data) - 3:
        break
    ind += 4

print(np.asarray(bleu_score) / count)
print(np.asarray(rouge_1) / count)
print(np.asarray(rouge_2) / count)
print(np.asarray(rouge_l) / count)
print(np.asarray(tree_ed) / count)
print(np.asarray(lengths) / count)
print(np.asarray([len(c) for c in counters]))
print(np.asarray([len(c) for c in bi_counters]))
print(np.asarray([len(c) for c in tri_counters]))
print(np.asarray(word_embed) / count)

#sent_vec = sent_vec.reshape(3 * sent_vec.shape[1], 300)
U, S, Vt = randomized_svd(sent_vec[0].T, n_components=1)
#sent_vec = sent_vec.reshape(3 * sent_vec.shape[1], 300)
#U = np.tile(U[:, 0], 3)[:, np.newaxis]
for x in range(3):
    sent_vec[x] -= (U * np.dot(U.T, sent_vec[x].T)).T
#sent_vec = sent_vec.reshape(3, len(alt), 300)

def cosine(sent_vec, x, y):
    return (np.sum(sent_vec[x] * sent_vec[y], axis=1) / (np.sqrt(np.sum(sent_vec[x]**2, axis=1)) * np.sqrt(np.sum(sent_vec[y]**2, axis=1)))).sum()
print(cosine(sent_vec, 0, 1) / count)
print(cosine(sent_vec, 0, 2) / count)
