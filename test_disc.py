import torch
import numpy as np
import json
from gensim.models import Word2Vec

from hred_data_loader import get_loader


def main():
    dictionary = json.load(open('./dictionary.json'))
    vocab_size = len(dictionary) + 1
    inverse_dict = {}
    for word, wid in dictionary.items():
        inverse_dict[wid] = word

    train_loader = get_loader('./data/train.src', './data/train.tgt', dictionary, 80)

    disc = torch.load('discriminator.pt')
    disc.flatten_parameters()

    for _, (src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, trg_seqs, trg_lengths,
            trg_indices, turn_len) in enumerate(train_loader):
        scores = disc.evaluate(ctc_seqs, ctc_lengths, ctc_indices, trg_seqs, trg_lengths, trg_indices)
        for i in range(scores.shape[0]):
            print(reconstruct_sent(trg_seqs[i], inverse_dict))
            print(scores[i])


def reconstruct_sent(seq, dictionary):
    return ' '.join([dictionary[wid] for wid in seq])


if __name__ == '__main__':
    main()
