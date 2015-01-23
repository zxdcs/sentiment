__author__ = 'zxd'

import math

from libsvm.svmutil import *
from exp.evaluation import *


def svm_classify(x_train, y_train, x_test, y_test, param='-t 2 -c 10'):
    m = svm_train(y_train, x_train, param)
    p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
    p_val_list = [v[0] for v in p_val]
    return p_val_list


def best_trust(p_val_list, ids, pc, nc):
    id_val = list(zip(ids, p_val_list))
    p_id_val = [x for x in id_val if x[1] >= 0]
    n_id_val = [x for x in id_val if x[1] < 0]
    p_id_val.sort(key=lambda x: x[1], reverse=True)
    n_id_val.sort(key=lambda x: -x[1], reverse=True)
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
    pc, nc = 100, 100
    ty, tx = svm_read_problem(r'..\data\data_balanced\lexical_vec_avg.txt')
    ay, ax = svm_read_problem(r'..\data\data_balanced\acoustic.txt')
    sp_idx = 3701
    y_train, y_test = ty[:sp_idx], ty[sp_idx:]
    x_trains, x_tests = (tx[:sp_idx], ax[:sp_idx]), (tx[sp_idx:], ax[sp_idx:])
    test_ids = list(range(len(y_test)))
    y_pred = [None] * len(y_test)
    y_real = list(y_test)
    iter = 1
    while x_tests[0]:
        print('iter: ' + str(iter))
        iter += 1
        text_p_val_list = svm_classify(x_trains[0], y_train, x_tests[0], y_test)
        text_trust = best_trust(text_p_val_list, test_ids, pc, nc)
        acous_p_val_list = svm_classify(x_trains[1], y_train, x_tests[1], y_test)
        acous_trust = best_trust(acous_p_val_list, test_ids, pc, nc)
        trust = agree(text_trust, acous_trust)
        for tid in trust:
            idx = test_ids.index(tid)
            x_trains[0].append(x_tests[0][idx])
            x_trains[1].append(x_tests[1][idx])
            y_pred[tid] = 0 if trust[tid] < 0 else 1
            y_train.append(y_pred[tid])
            del x_tests[0][idx]
            del x_tests[1][idx]
            del test_ids[idx]
            del y_test[idx]
    fscore, precison, recall = f_score(y_real, y_pred)
    print(y_pred)
    print('fscore {0:f}  precision {1:f}  recall {2:f}'.format(fscore, precison, recall))


if __name__ == '__main__':
    cotrain_svm()