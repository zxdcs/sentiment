__author__ = 'zxd'

import numpy

from dnn.data_reader import read_data
from data_util.data import Data


def read_utterance_id(file):
    f = open(file)
    uid = []
    for line in f:
        tokens = line[:line.index('.')].split('_')
        uid.append((tokens[0], tokens[2]))
    return uid


def find_context(uid, idx):
    l_idx = -1
    if idx - 1 >= 0 and uid[idx][0] == uid[idx - 1][0]:
        l_idx = idx - 1
    ls_idx = -1
    p = idx - 1
    while p >= 0 and uid[idx][0] == uid[p][0]:
        if uid[idx][1] == uid[p][1]:
            ls_idx = p
            break
        p -= 1
    return l_idx, ls_idx


def array_copy(sa, ta, sx, tx, ty):
    for i in range(sa.shape[1]):
        ta[tx][ty + i] = sa[sx][i]


def context_feature(in_file, out_file):
    x, y = read_data(in_file)
    xc = numpy.ones((x.shape[0], x.shape[1] * 3)) / 2
    uid = read_utterance_id(r'..\data\raw_data_all\filelist.txt')
    for i in range(len(x)):
        l_idx, ls_idx = find_context(uid, i)
        if l_idx != -1:
            array_copy(x, xc, l_idx, i, x.shape[1])
        if ls_idx != -1:
            array_copy(x, xc, ls_idx, i, x.shape[1] * 2)
    f = open(out_file, 'w', encoding='utf-8')
    for ix, iy in zip(xc, y):
        f.write(Data(ix, iy).to_str() + '\n')


if __name__ == '__main__':
    context_feature(r'..\data\data_all\lexical_vec_avg.txt', r'..\data\data_context\lexical_vec_avg.txt')