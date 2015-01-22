__author__ = 'zxd'

import numpy

from libsvm.svmutil import *


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
    param = '-t 2 -c 10 -w0 0.2'
    y, x = svm_read_problem(file)
    sp_idx = 23993
    m = svm_train(y[:sp_idx], x[:sp_idx], param)
    p_label, p_acc, p_val = svm_predict(y[sp_idx:], x[sp_idx:], m)
    fscore, precison, recall = f_score(y[sp_idx:], p_label)
    print(p_label)
    print('fscore {0:f}  precision {1:f}  recall {2:f}'.format(fscore, precison, recall))


def f_score(y_real, y_pred, target=1, label_num=2):
    if len(y_real) != len(y_pred):
        raise TypeError(
            'y_real should have the same shape as y_pred',
            ('y_real ', len(y_real), 'y_pred', len(y_pred))
        )
    count = numpy.zeros([label_num, label_num])
    for real, pred in zip(y_real, y_pred):
        count[real][pred] += 1
    precison = count[target][target] / numpy.sum(count, axis=0)[target]
    recall = count[target][target] / numpy.sum(count, axis=1)[target]
    fscore = 2 * precison * recall / (precison + recall)
    return fscore, precison, recall


if __name__ == '__main__':
    # score = cross_validation(r'..\data\data_balanced\lexical_vec_avg.txt')
    # test(r'..\data\data_balanced\acous_lex_avg.txt')
    test2(r'..\data\data_all\lexical_vec_avg.txt')