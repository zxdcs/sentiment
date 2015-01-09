__author__ = 'zxd'

from libsvm.svmutil import *


def cross_validation(file, k=10):
    param = '-t 2 -c 1 -g 0.05'
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


def fscore(calc_list, real_list, target_label=1):
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
    return f


if __name__ == '__main__':
    score = cross_validation(r'..\data\data_balanced\lexical_vec_avg01.txt')
    print(score)