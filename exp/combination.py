__author__ = 'zxd'

from libsvm.svmutil import *
from exp.evaluation import *


def svm_classify(file, sp_idx=3701, param='-t 2 -c 10'):
    y, x = svm_read_problem(file)
    m = svm_train(y[:sp_idx], x[:sp_idx], param)
    p_label, p_acc, p_val = svm_predict(y[sp_idx:], x[sp_idx:], m)
    p_val_array = numpy.asarray([v[0] for v in p_val])
    p_val_array /= numpy.max(numpy.fabs(p_val_array))
    return p_val_array


def read_real(file, sp_idx=3701):
    y, x = svm_read_problem(file)
    return y[sp_idx:]


def combine_svm():
    sp_idx = 3701
    text_predict = svm_classify(r'..\data\data_balanced\lexical_vec_avg.txt', sp_idx, '-t 2 -c 5')
    acous_predict = svm_classify(r'..\data\data_balanced\acoustic.txt', sp_idx, '-t 2 -c 25')
    combine_predict = text_predict + acous_predict
    y_pred = [0 if v < 0 else 1 for v in combine_predict]
    y_real = read_real(r'..\data\data_balanced\lexical_vec_avg.txt', sp_idx)
    fscore, precison, recall = f_score(y_real, y_pred)
    print(y_pred)
    print('fscore {0:f}  precision {1:f}  recall {2:f}'.format(fscore, precison, recall))


def combine_svm_imbalance():
    sp_idx = 23993
    text_predict = svm_classify(r'..\data\data_all\lexical_vec_avg.txt', sp_idx, '-t 2 -c 5 -w0 0.2')
    acous_predict = svm_classify(r'..\data\data_all\acoustic.txt', sp_idx, '-t 2 -c 25 -w0 0.2')
    combine_predict = text_predict + acous_predict
    y_pred = [0 if v < 0 else 1 for v in combine_predict]
    y_real = read_real(r'..\data\data_all\lexical_vec_avg.txt', sp_idx)
    fscore, precison, recall = f_score(y_real, y_pred)
    print(y_pred)
    print('fscore {0:f}  precision {1:f}  recall {2:f}'.format(fscore, precison, recall))


if __name__ == '__main__':
    combine_svm_imbalance()