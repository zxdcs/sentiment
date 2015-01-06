__author__ = 'zxd'

from sklearn import preprocessing as pp
import numpy as np


def read_data(file):
    f = open(file, 'r', encoding='utf-8')
    data = [[float(tok) for tok in line.split(' ')] for line in f]
    return np.asarray(data)


def write_data(file, data_scaled):
    f = open(file, 'w', encoding='utf-8')
    data_str = [' '.join(map('{0:.6f}'.format, row)) + '\n' for row in data_scaled]
    f.writelines(data_str)


def scale_min_max(data, min=0, max=1):
    min_max_scaler = pp.MinMaxScaler((min, max), copy='False')
    data_scaled = min_max_scaler.fit_transform(data)
    return data_scaled


if __name__ == '__main__':
    data_un = read_data(r'../data/raw_data_balanced/acoustic.txt')
    data_sc = scale_min_max(data_un)
    write_data(r'../data/raw_data_balanced/acoustic_scale.txt', data_sc)