import torch
import numpy as np
import json
from gensim.models import Word2Vec

from hred_data_loader import get_loader

def main():
    dictionary = json.load(open('./dictionary.json'))
    vocab_size = len(dictionary) + 1

    word_embedding_dim = 300
    print("Vocabulary size:", len(dictionary))
    word_vectors = np.random.uniform(low=-0.1, high=0.1, size=(vocab_size, word_embedding_dim))
    found = 0
    print("Loading word vecotrs.")
    word2vec = Word2Vec.load('./word2vec.vector')
    for word in word2vec.wv.vocab:
        if word in dictionary:
            word_vectors[dictionary[word]] = word2vec.wv[word]
            found += 1
    print(found)

    train_loader = get_loader('./data/train.src', './data/train.tgt', dictionary, 80)

    disc = torch.load('discriminator.pt')

    for _, (src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, trg_seqs, trg_lengths, trg_indices,
            turn_len) in enumerate(train_loader):
        print(trg_seqs)
        scores = disc.evaluate(src_seqs, src_lengths, src_indices, trg_seqs, trg_lengths, trg_indices)
        print(scores)

