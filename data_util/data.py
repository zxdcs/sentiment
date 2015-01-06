__author__ = 'zxd'

import pynlpir


class Data(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def read_lexical_datas(file):
    pynlpir.open()
    f = open(file, 'r', encoding='utf-8')
    for line in f:
        line = line.replace('幺', '一')
        tokens = pynlpir.segment(line, pos_tagging=False)
        print(tokens)


if __name__ == '__main__':
    read_lexical_datas(r'../data/raw_data_balanced/text.txt')