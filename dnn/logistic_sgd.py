"""
Logistic regression using stochastic gradient descent for sentiment classification.
"""

__docformat__ = 'restructedtext en'

import os
import time
import random

import theano
import numpy
import theano.tensor as T

from dnn.data_reader import split_data, read_data, load_data


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        self.n_out = n_out

        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y, penalty=None):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        if penalty is None:
            return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        else:
            return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
                           * penalty[T.arange(y.shape[0]), y])


def sgd_optimization(datasets, learning_rate=0.13, n_epochs=1000, batch_size=30):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type datasets: string
    :param datasets: datasets

    """

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    # compute number of minibatches for training, validation
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size + 1

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    dim = train_set_x.get_value(borrow=True).shape[1]
    classifier = LogisticRegression(input=x, n_in=dim, n_out=2)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    penalty = [0.2, 1]
    penalty_sh = theano.shared(numpy.asarray([penalty] * batch_size, dtype=theano.config.floatX), borrow=True)
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.y_pred,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')

    best_fscore = 0
    start_time = time.clock()

    epoch = 0
    batches_idx = list(range(n_train_batches))
    while epoch < n_epochs:
        epoch += 1
        random.shuffle(batches_idx)
        for minibatch_index in batches_idx:
            minibatch_avg_cost = train_model(minibatch_index)

        # compute f-score on validation set
        y_preds = [validate_model(i) for i in range(n_valid_batches)]
        y_pred = [pij for pi in y_preds for pij in pi]
        y_real = valid_set_y.get_value(borrow=True)
        fscore = f_score(y_real, y_pred)
        print(
            'epoch {0:d}, minibatch {1:d}/{2:d}, fscore {3:f} %'
            .format(epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    fscore * 100.)
        )

        # if we got the best validation score until now
        if fscore > best_fscore:
            best_fscore = fscore
            print('-----Best score: {0:f}-----'.format(best_fscore))

    end_time = time.clock()
    print('Optimization complete with best validation score of {0:.1f} %,'
          .format(best_fscore * 100.))
    print('The code run for {0:d} epochs, with {1:f} epochs/sec'
          .format(epoch, 1. * epoch / (end_time - start_time)))
    print('The code for file ' + os.path.split(__file__)[1] +
          ' ran for {0:.1f}s'.format(end_time - start_time))

    return best_fscore


def errors(y_real, y_pred):
    if len(y_real) != len(y_pred):
        raise TypeError(
            'y_real should have the same shape as y_pred',
            ('y_real ', len(y_real), 'y_pred', len(y_pred))
        )
    neq = 0
    for real, pred in zip(y_real, y_pred):
        if real != pred:
            neq += 1
    loss = 1.0 * neq / len(y_pred)
    return loss


def f_score(y_real, y_pred, target=1, label_num=2):
    print(y_pred)
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
    print('p:{0:f} r:{1:f} f:{2:f}'.format(precison, recall, fscore))
    return fscore


def corss_validation():
    k = 4
    avg_score = 0
    x, y = read_data(r'..\data\data_balanced\lexical_vec_avg.txt')
    for p_st in [x / k for x in range(0, k)]:
        p_en = p_st + 1 / k
        datas = split_data(x, y, p_st, p_en)
        score = sgd_optimization(datas, n_epochs=100)
        avg_score += score
    avg_score /= k
    print('Average score is: {0:f}'.format(avg_score))


if __name__ == '__main__':
    sgd_optimization(load_data(r'..\data\data_balanced\lexical_vec_avg.txt', sp_idx=3701),
                     n_epochs=1000, batch_size=100)
    # corss_validation()