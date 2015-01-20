__author__ = 'zxd'

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
        fs = fscore(p_label, y[sp1:sp2])
        print('f-score: ' + str(fs))
        avg_score += fs
    avg_score /= k
    return avg_score


def test(file):
    param = '-t 2 -c 1'
    y, x = svm_read_problem(file)
    total_len = len(y)
    sp1 = int(0.7 * total_len)
    sp2 = int(1 * total_len)
    m = svm_train(y[:sp1] + y[sp2:], x[:sp1] + x[sp2:], param)
    p_label, p_acc, p_val = svm_predict(y[sp1:sp2], x[sp1:sp2], m)
    fs = fscore(p_label, y[sp1:sp2])
    return fs


def test2(file):
    param = '-t 2 -c 1'
    y, x = svm_read_problem(file)
    sp_idx = 3725
    m = svm_train(y[:sp_idx], x[:sp_idx], param)
    p_label, p_acc, p_val = svm_predict(y[sp_idx:], x[sp_idx:], m)
    fs = fscore(p_label, y[sp_idx:])
    return fs


def test_unbalance(file_b, file_u):
    param = '-t 2 -c 1 -w0 0.12'
    yb, xb = svm_read_problem(file_b)
    total_len = len(yb)
    sp = int(0.6 * total_len)
    m = svm_train(yb[:sp], xb[:sp], param)

    yu, xu = svm_read_problem(file_u)
    total_len = len(yu)
    sp = int(0.7 * total_len)
    p_label, p_acc, p_val = svm_predict(yu[sp:], xu[sp:], m)
    fs = fscore(p_label, yu[sp:])
    return fs


def fscore(calc_list, real_list, target_label=1):
    print(calc_list)
    tp = fp = tn = fn = 0
    for calc, real in zip(calc_list, real_list):
        if calc == target_label and real == target_label:
            tp += 1
        elif calc == target_label and real != target_label:
            fp += 1
        elif calc != target_label and real != target_label:
            tn += 1
        elif calc != target_label and real == target_label:
            fn += 1
        else:
            raise Exception('label wrong!')
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = 2 * p * r / (p + r)
    print('p:{0:f} r:{1:f} f:{2:f}'.format(p, r, f))
    return f


if __name__ == '__main__':
    # score = cross_validation(r'..\data\data_balanced\lexical_vec_avg.txt')
    score = test2(r'..\data\data_balanced\lexical_vec_avg.txt')
    # score = test_unbalance(r'..\data\data_all\lexical_vec_avg.txt',
    # r'..\data\data_balanced\lexical_vec_avg.txt')
    print(score)