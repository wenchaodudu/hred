import pdb
import csv
import numpy as np


truth = csv.reader(open('../Holmes/test_answer.csv'))
next(truth)
labels = []
for id, choice in truth:
    labels.append(int(ord(choice) - ord('a')))
pred = open('test.txt').readlines()
scores = np.zeros((5200, 4))
for _, line in enumerate(pred):
    num = line.split()
    for x in range(4):
        scores[_, x] = float(num[x])
scores = scores.reshape(1040, 5, 4)
pred1 = np.argmin(scores[:, :, 0], axis=1)
pred2 = np.argmin(scores[:, :, 1], axis=1)
def weight(x):
    return np.argmin(scores[:, :, 1] * x + scores[:, :, 2] + scores[:, :, 3], axis=1)
pdb.set_trace()
