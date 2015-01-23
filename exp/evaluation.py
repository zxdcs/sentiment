__author__ = 'zxd'

import numpy


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