__author__ = 'zxd'

import pynlpir
import numpy

from data_util import word_vec


WORD_VEC_DIM = 200


class Data(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_str(self):
        output = str(self.y)
        for idx, val in enumerate(self.x):
            if val:
                output += ' {0:d}:{1:g}'.format(idx + 1, val)
        return output


def read_lexical_datas(file, compose_func=None):
    pynlpir.open()
    f = open(file, 'r', encoding='utf-8')
    tokens_list = [pynlpir.segment(line.replace('幺', '一'), pos_tagging=False) for line in f]
    if compose_func is None:
        word_idx = {}
        for tokens in tokens_list:
            for token in tokens:
                if token not in word_idx:
                    word_idx[token] = len(word_idx)
        array = numpy.zeros([len(tokens_list), len(word_idx)])
        for i, tokens in enumerate(tokens_list):
            for token in tokens:
                array[i][word_idx[token]] = 1.0
    else:
        print('reading word vectors')
        word_vecs = word_vec.read_word_vec(r'../data/vectors_cbow')
        print('reading complete')
        array = numpy.asarray([compose_func(tokens, word_vecs) for tokens in tokens_list])
    return array


def compose_func_avg(tokens, word_vecs):
    token_vec = numpy.asarray([word_vecs[token] for token in tokens if token in word_vecs])
    if len(token_vec):
        avg_vec = numpy.average(token_vec, axis=0)
    else:
        avg_vec = numpy.zeros([WORD_VEC_DIM])
    return sigmoid(avg_vec)


def compose_func_bigram(tokens, word_vecs):
    token_vec = numpy.asarray([word_vecs[token] for token in tokens if token in word_vecs])
    if len(token_vec) > 1:
        bigram_vec = numpy.zeros([WORD_VEC_DIM])
        for i in range(1, len(token_vec)):
            bigram_vec += sigmoid(token_vec[i - 1] + token_vec[i])
        bigram_vec /= len(token_vec)
    elif len(token_vec) == 1:
        bigram_vec = sigmoid(token_vec[0])
    else:
        bigram_vec = sigmoid(numpy.zeros([WORD_VEC_DIM]))
    return bigram_vec


def read_acoustic_datas(file):
    f = open(file, 'r', encoding='utf-8')
    tokens_list = [[float(token) for token in line.split()] for line in f]
    return numpy.asarray(tokens_list)


def read_labels(file):
    f = open(file, 'r', encoding='utf-8')
    return [int(line) for line in f]


def sigmoid(x):
    return (numpy.tanh(x / 2) + 1) / 2


def process_raw_data_lexical():
    xs = read_lexical_datas(r'../data/raw_data_balanced/text.txt', compose_func=compose_func_bigram)
    ys = read_labels(r'../data/raw_data_balanced/label.txt')
    f = open(r'../data/data_balanced/lexical_vec_bigram.txt', 'w', encoding='utf-8')
    for x, y in zip(xs, ys):
        f.write(Data(x, y).to_str() + '\n')


def process_raw_data_acoustic():
    xs = read_acoustic_datas(r'../data/raw_data_balanced/acoustic_scale.txt')
    ys = read_labels(r'../data/raw_data_balanced/label.txt')
    f = open(r'../data/data_balanced/acoustic.txt', 'w', encoding='utf-8')
    for x, y in zip(xs, ys):
        f.write(Data(x, y).to_str() + '\n')


def process_raw_data_acous_lex():
    acous_xs = read_acoustic_datas(r'../data/raw_data_balanced/acoustic_scale.txt')
    lex_xs = read_lexical_datas(r'../data/raw_data_balanced/text.txt', compose_func=compose_func_bigram)
    xs = numpy.hstack((acous_xs, lex_xs))
    ys = read_labels(r'../data/raw_data_balanced/label.txt')
    f = open(r'../data/data_balanced/acous_lex_bigram.txt', 'w', encoding='utf-8')
    for x, y in zip(xs, ys):
        f.write(Data(x, y).to_str() + '\n')


if __name__ == '__main__':
    process_raw_data_lexical()