import torch
from torch.autograd import Variable
import numpy as np
import random

from model import Embedding, UtteranceEncoder, ContextEncoder


def test_embedding():
    init_embedding = np.asarray([[0.1 * i] * 10 for i in range(5)])
    embedding = Embedding(5, 10, init_embedding)
    print(embedding)
    input_word = Variable(torch.LongTensor([0,1,1,2,4]))
    print(embedding(input_word))


def test_encoder():
    init_embedding = np.asarray([[0.1 * i] * 10 for i in range(5)])
    encoder = UtteranceEncoder(init_embedding, hidden_size=10)
    input_word = Variable(torch.LongTensor([[0, 1, 2, 3, 4], [1, 1, 2, 2, 3]]))
    output = encoder(input_word)
    print(output)

    cencoder = ContextEncoder(20, 10, 2)
    output = cencoder(output)
    print(output)


def get_dummy_train_data(D, N, l, V):
    data = []
    for i in range(D):
        datapoint = []
        for j in range(N):
            length = random.randint(1, l)
            uttr = np.zeros(length)
            for k in range(length):
                uttr[k] = random.randint(0, V-1)
            datapoint.append(uttr)
        data.append(datapoint)
    return data


if __name__ == '__main__':
    # test_embedding()
    test_encoder()
