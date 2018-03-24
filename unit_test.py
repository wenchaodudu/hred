import torch
from torch.autograd import Variable
import numpy as np

from model import Embedding, UtteranceEncoder


def test_embedding():
    init_embedding = np.asarray([[0.1 * i] * 10 for i in range(5)])
    embedding = Embedding(5, 10, init_embedding)
    print(embedding)
    input_word = Variable(torch.LongTensor([0,1,1,2,4]))
    print(embedding(input_word))


def test_encoder():
    init_embedding = np.asarray([[0.1 * i] * 10 for i in range(5)])
    encoder = UtteranceEncoder(init_embedding, hidden_size=10)
    input_word = Variable(torch.LongTensor([0, 1, 2, 3, 4]))
    print(encoder(input_word))


if __name__ == '__main__':
    # test_embedding()
    test_encoder()
