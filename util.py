import numpy as np


def get_length(data):
    len_turns = list(map(lambda x: np.shape(np.where(x[:,0]>0))[1], data))
    len_uttrs = []
    for d in data:
        len_uttrs += list(map(lambda x: np.shape(np.where(x>0))[1], d))
    return len_turns, len_uttrs