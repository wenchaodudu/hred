import torch
import numpy as np
import json
from gensim.models import Word2Vec

from hred_data_loader import get_loader


def main():
    dictionary = json.load(open('./dictionary.json'))
    vocab_size = len(dictionary) + 1

    train_loader = get_loader('./data/train.src', './data/train.tgt', dictionary, 80)

    disc = torch.load('discriminator.pt')

    for _, (src_seqs, src_lengths, src_indices, ctc_seqs, ctc_lengths, ctc_indices, trg_seqs, trg_lengths, trg_indices,
            turn_len) in enumerate(train_loader):
        print(trg_seqs)
        scores = disc.evaluate(src_seqs, src_lengths, src_indices, trg_seqs, trg_lengths, trg_indices)
        print(scores)


if __name__ == '__main__':
    main()
