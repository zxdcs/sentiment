import numpy
import theano

__author__ = 'zxd'

'''
def load_data(dataset):
    """ Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    """

    # Load the dataset
    print('... loading data')
    f = open(dataset, 'r')

    all_datas = f.readlines()
    n = len(all_datas)
    dim = 0
    for line in all_datas:
        for token in line.split()[1:]:
            idx = int(token[:token.index(':')])
            dim = max(dim, idx)

    x = numpy.zeros([n, dim])
    y = numpy.zeros(n)
    for i in range(n):
        tokens = all_datas[i].split(' ')
        y[i] = int(tokens[0])
        for j in range(1, len(tokens)):
            idx = tokens[j].find(':')
            if idx != -1:
                feature = int(tokens[j][:idx]) - 1
                d = float(tokens[j][idx + 1:])
                x[i][feature] = d
    st = int(0.7 * n)
    end = int(1 * n)
    train_set = (numpy.append(x[:st], x[end:], axis=0), numpy.append(y[:st], y[end:], axis=0))
    valid_set = (x[st:end], y[st:end])
    f.close()

    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    return rval
'''


def load_data(dataset, sp_idx=3701):
    # balance sp: 3701  imbalance sp: 23993
    x, y = read_data(dataset)
    return split_train_valid(x, y, sp_idx)


def read_data(dataset):
    """ Loads the dataset (all & not shared var)

    :type dataset: string
    :param dataset: the path to the dataset
    """

    # Load the dataset
    print('... reading data')
    f = open(dataset, 'r')

    all_datas = f.readlines()
    n = len(all_datas)
    dim = 0
    for line in all_datas:
        for token in line.split()[1:]:
            idx = int(token[:token.index(':')])
            dim = max(dim, idx)

    x = numpy.zeros([n, dim])
    y = numpy.zeros(n)
    for i in range(n):
        tokens = all_datas[i].split(' ')
        y[i] = int(tokens[0])
        for j in range(1, len(tokens)):
            idx = tokens[j].find(':')
            if idx != -1:
                feature = int(tokens[j][:idx]) - 1
                d = float(tokens[j][idx + 1:])
                x[i][feature] = d
    f.close()
    return x, y


def split_train_valid(x, y, sp_idx):
    train_set = (x[:sp_idx], y[:sp_idx])
    valid_set = (x[sp_idx:], y[sp_idx:])
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    return rval


def split_data(x, y, p_st, p_en):
    n = len(x)
    st = int(p_st * n)
    end = int(p_en * n)
    train_set = (numpy.append(x[:st], x[end:], axis=0), numpy.append(y[:st], y[end:], axis=0))
    valid_set = (x[st:end], y[st:end])

    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    return rval


def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype='int32'), borrow=borrow)
    return shared_x, shared_y