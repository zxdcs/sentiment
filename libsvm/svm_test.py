__author__ = 'zxd'

from libsvm.svmutil import *
from exp.evaluation import *


def cross_validation(file, k=10):
    param = '-t 2 -c 1'
    y, x = svm_read_problem(file)
    total_len = len(y)
    fold_len = total_len // k
    avg_score = 0
    for i in range(10):
        sp1 = i * fold_len
        sp2 = (i + 1) * fold_len
        m = svm_train(y[:sp1] + y[sp2:], x[:sp1] + x[sp2:], param)
        p_label, p_acc, p_val = svm_predict(y[sp1:sp2], x[sp1:sp2], m)
        fs = f_score(y[sp1:sp2], p_label)
        print('f-score: ' + str(fs))
        avg_score += fs
    avg_score /= k
    return avg_score


def test(file):
    param = '-t 2 -c 10'
    y, x = svm_read_problem(file)
    sp_idx = 3701
    m = svm_train(y[:sp_idx], x[:sp_idx], param)
    p_label, p_acc, p_val = svm_predict(y[sp_idx:], x[sp_idx:], m)
    fscore, precison, recall = f_score(y[sp_idx:], p_label)
    print(p_label)
    print('fscore {0:f}  precision {1:f}  recall {2:f}'.format(fscore, precison, recall))


def test2(file):
    param = '-t 2 -c 25 -w0 0.2'
    y, x = svm_read_problem(file)
    sp_idx = 23993
    m = svm_train(y[:sp_idx], x[:sp_idx], param)
    p_label, p_acc, p_val = svm_predict(y[sp_idx:], x[sp_idx:], m)
    fscore, precison, recall = f_score(y[sp_idx:], p_label)
    print(p_label)
    print('fscore {0:f}  precision {1:f}  recall {2:f}'.format(fscore, precison, recall))


if __name__ == '__main__':
    # score = cross_validation(r'..\data\data_balanced\lexical_vec_avg.txt')
    # test(r'..\data\data_balanced\acoustic.txt')
    test2(r'..\data\data_all\acoustic.txt')