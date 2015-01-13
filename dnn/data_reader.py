import numpy
import theano

__author__ = 'zxd'


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
    st = int(0.3 * n)
    end = int(0.4 * n)
    train_set = (numpy.append(x[:st], x[end:], axis=0), numpy.append(y[:st], y[end:], axis=0))
    valid_set = (x[st:end], y[st:end])

    f.close()
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # witch row's correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. It should give the target
    # target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype='int32'), borrow=borrow)

        return shared_x, shared_y

    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    return rval


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


def split_data(x, y, p_st, p_en):
    n = len(x)
    st = int(p_st * n)
    end = int(p_en * n)
    train_set = (numpy.append(x[:st], x[end:], axis=0), numpy.append(y[:st], y[end:], axis=0))
    valid_set = (x[st:end], y[st:end])

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype='int32'), borrow=borrow)
        return shared_x, shared_y

    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    return rval