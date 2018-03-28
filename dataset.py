import numpy as np
import random


class Dataset(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_next_batch(self):
        pass

    def start_iter(self):
        pass


class DummyDataset(Dataset):
    def __init__(self, batch_size, num_dialog, max_turn, max_len, vocab_size, embed_dim):
        super(DummyDataset, self).__init__(batch_size)
        self.num_dialog = num_dialog
        self.max_turn = max_turn
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.embedding = get_dummy_embedding(vocab_size, embed_dim)
        self.data = self.gen_data()
        self.iterator = None

    def gen_data(self):
        data = []
        for i in range(self.num_dialog):
            datapoint = []
            turns = random.randint(2, self.max_turn)
            for j in range(turns):
                length = random.randint(1, self.max_len)
                uttr = np.zeros(length)
                for k in range(length):
                    uttr[k] = random.randint(1, self.vocab_size)
                datapoint.append(uttr)
            datapoint.sort(key=len, reverse=True)
            data.append(datapoint)
        data.sort(key=len, reverse=True)

        padded = np.zeros((self.num_dialog, self.max_turn, self.max_len))
        for d in range(len(data)):
            dialog = data[d]
            for t in range(len(dialog)):
                turn = dialog[t]
                for w in range(len(turn)):
                    padded[d, t, w] = turn[w]

        return padded

    def get_embedding(self):
        return self.embedding

    def get_next_batch(self):
        if self.iterator:
            batch = self.iterator.pop()
            return self.data[batch * self.batch_size : (batch + 1) * self.batch_size]
        else:
            return None

    def start_iter(self):
        num_batch = int(self.data.shape[0] / self.batch_size)
        self.iterator = list(range(num_batch))
        random.shuffle(self.iterator)


def get_dummy_embedding(vocabulary_size, embedding_size):
    return np.random.random((vocabulary_size, embedding_size))


if __name__ == '__main__':
    data = DummyDataset(4, 16, 5, 10, 50, 20)
    data.start_iter()
    while True:
        batch = data.get_next_batch()
        if batch is None:
            break
        print(np.shape(batch))
