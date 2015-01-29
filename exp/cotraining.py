__author__ = 'zxd'

import math

from libsvm.svmutil import *
from exp.evaluation import *


def svm_classify(x_train, y_train, x_test, y_test, param='-t 2 -c 10'):
    m = svm_train(y_train, x_train, param)
    p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
    p_val_list = [v[0] for v in p_val]
    return p_label, p_val_list


def best_trust(p_label, p_val_list, ids, pc, nc):
    id_val = list(zip(ids, p_val_list))
    p_id_val = []
    n_id_val = []
    for i in range(len(p_label)):
        if p_label[i] == 1:
            p_id_val.append(id_val[i])
        else:
            n_id_val.append(id_val[i])
    p_id_val.sort(key=lambda x: math.fabs(x[1]), reverse=True)
    n_id_val.sort(key=lambda x: math.fabs(-x[1]), reverse=True)
    return p_id_val[:pc] + n_id_val[:nc]


def agree(text_trust, acous_trust):
    trust = {t[0]: t[1] for t in text_trust}
    for t in acous_trust:
        if t[0] in trust:
            if math.fabs(t[1]) > math.fabs(trust[t[0]]):
                trust[t[0]] = t[1]
        else:
            trust[t[0]] = t[1]
    return trust


def cotrain_svm():
    pc, nc = 200, 200
    ty, tx = svm_read_problem(r'..\data\data_balanced\lexical_vec_avg.txt')
    ay, ax = svm_read_problem(r'..\data\data_balanced\acoustic.txt')
    sp_idx = 3701
    y_train, y_test = ty[:sp_idx], ty[sp_idx:]
    x_trains, x_tests = (tx[:sp_idx], ax[:sp_idx]), (tx[sp_idx:], ax[sp_idx:])
    test_ids = list(range(len(y_test)))
    y_pred = [None] * len(y_test)
    y_real = list(y_test)
    it = 1
    while x_tests[0]:
        print('iter: ' + str(it))
        it += 1
        text_label, text_p_val_list = svm_classify(x_trains[0], y_train, x_tests[0], y_test, '-t 2 -c 5')
        text_trust = best_trust(text_label, text_p_val_list, test_ids, pc, nc)
        acous_label, acous_p_val_list = svm_classify(x_trains[1], y_train, x_tests[1], y_test, '-t 2 -c 25')
        acous_trust = best_trust(acous_label, acous_p_val_list, test_ids, pc, nc)
        trust = agree(text_trust, acous_trust)
        for tid in trust:
            idx = test_ids.index(tid)
            x_trains[0].append(x_tests[0][idx])
            x_trains[1].append(x_tests[1][idx])
            # ugly code because of the way libsvm treat labels
            if y_train[0] == 1:
                y_pred[tid] = 0 if trust[tid] < 0 else 1
            else:
                y_pred[tid] = 0 if trust[tid] >= 0 else 1
            y_train.append(y_pred[tid])
            del x_tests[0][idx]
            del x_tests[1][idx]
            del test_ids[idx]
            del y_test[idx]
    fscore, precison, recall = f_score(y_real, y_pred)
    print(y_pred)
    print('fscore {0:f}  precision {1:f}  recall {2:f}'.format(fscore, precison, recall))


def cotrain_svm_imbalance():
    pc, nc = 200, 2500
    ty, tx = svm_read_problem(r'..\data\data_all\lexical_vec_avg.txt')
    ay, ax = svm_read_problem(r'..\data\data_all\acoustic.txt')
    sp_idx = 23993
    y_train, y_test = ty[:sp_idx], ty[sp_idx:]
    x_trains, x_tests = (tx[:sp_idx], ax[:sp_idx]), (tx[sp_idx:], ax[sp_idx:])
    test_ids = list(range(len(y_test)))
    y_pred = [None] * len(y_test)
    y_real = list(y_test)
    it = 1
    while x_tests[0]:
        print('iter: ' + str(it))
        it += 1
        text_label, text_p_val_list = svm_classify(x_trains[0], y_train, x_tests[0], y_test, '-t 2 -c 5 -w0 0.2')
        text_trust = best_trust(text_label, text_p_val_list, test_ids, pc, nc)
        acous_label, acous_p_val_list = svm_classify(x_trains[1], y_train, x_tests[1], y_test, '-t 2 -c 25 -w0 0.2')
        acous_trust = best_trust(acous_label, acous_p_val_list, test_ids, pc, nc)
        trust = agree(text_trust, acous_trust)
        for tid in trust:
            idx = test_ids.index(tid)
            x_trains[0].append(x_tests[0][idx])
            x_trains[1].append(x_tests[1][idx])
            # ugly code because of the way libsvm treat labels
            if y_train[0] == 1:
                y_pred[tid] = 0 if trust[tid] < 0 else 1
            else:
                y_pred[tid] = 0 if trust[tid] >= 0 else 1
            y_train.append(y_pred[tid])
            del x_tests[0][idx]
            del x_tests[1][idx]
            del test_ids[idx]
            del y_test[idx]
    fscore, precison, recall = f_score(y_real, y_pred)
    print(y_pred)
    print('fscore {0:f}  precision {1:f}  recall {2:f}'.format(fscore, precison, recall))


if __name__ == '__main__':
    cotrain_svm_imbalance()