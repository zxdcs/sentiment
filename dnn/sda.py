"""
The stacked denoising auto-encoders (SdA) using Theano.
"""
import os
import time
import random

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from dnn.logistic_sgd import LogisticRegression
from dnn.data_reader import load_data
from dnn.mlp import HiddenLayer
from dnn.da import DA
from exp.evaluation import *


class SDA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            n_ins=784,
            hidden_layers_sizes=[500, 500],
            n_outs=10,
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = DA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
        # end-snippet-2
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y

        # predict y
        self.y_pred = self.logLayer.y_pred

    def pretraining_functions(self, train_set_x, batch_size):
        """ Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        """

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for da in self.dA_layers:
            # get the cost and the updates list
            cost, updates = da.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        """Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        """

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            },
            name='train'
        )

        valid_score_i = theano.function(
            [index],
            self.y_pred,
            givens={
                self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            },
            name='valid'
        )

        return train_fn, valid_score_i


def test_SdA(datasets, finetune_lr=0.1, pretraining_epochs=15,
             pretrain_lr=0.001, training_epochs=1000, batch_size=1):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    :type finetune_lr: float
    :param finetune_lr: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer

    :type datasets: string
    :param datasets: datasets

    """

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    # compute number of minibatches for training, validation
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size + 1

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print('... building the model')
    # construct the stacked denoising autoencoder class
    dim = int(train_set_x.get_value(borrow=True).shape[1])
    sda = SDA(
        numpy_rng=numpy_rng,
        n_ins=dim,
        hidden_layers_sizes=[900, 600, 300],
        n_outs=2
    )
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)
    print('... pre-training the model')
    start_time = time.clock()
    # Pre-train layer-wise
    corruption_levels = [.5, .5, .5]
    for i in range(sda.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            corruption=corruption_levels[i],
                                            lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost ' % (i, epoch), end=' ')
            print(numpy.mean(c))

    end_time = time.clock()

    print('The pretraining code for file ' +
          os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # imbalanced data
    penalty = [0.2, 1]
    penalty_sh = theano.shared(numpy.asarray([penalty] * batch_size, dtype=theano.config.floatX), borrow=True)
    sda.finetune_cost = sda.logLayer.negative_log_likelihood(sda.y, penalty_sh)

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn, validate_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('... finetunning the model')
    best_fscore = (0, 0, 0)
    start_time = time.clock()

    epoch = 0
    batches_idx = list(range(n_train_batches))
    while epoch < training_epochs:
        epoch += 1
        random.shuffle(batches_idx)
        for minibatch_index in batches_idx:
            minibatch_avg_cost = train_fn(minibatch_index)

        # compute f-score on validation set
        y_preds = [validate_model(i) for i in range(n_valid_batches)]
        y_pred = [pij for pi in y_preds for pij in pi]
        y_real = valid_set_y.get_value(borrow=True)
        print(y_pred)
        fscore, precison, recall = f_score(y_real, y_pred)
        print('epoch {0:d}, fscore {1:f}  precision {2:f}  recall {3:f}'.format(epoch, fscore, precison, recall))

        # if we got the best validation score until now
        if fscore > best_fscore[0]:
            best_fscore = (fscore, precison, recall)
            print('-----Best score: {0:f}-----'.format(fscore))

    end_time = time.clock()
    print('Optimization complete with best validation score: fscore {0:f}  precision {1:f}  recall {2:f},'
          .format(best_fscore[0], best_fscore[1], best_fscore[2]))
    print('The training code for file ' +
          os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    # test_SdA(load_data(r'..\data\data_balanced\acous_lex_avg.txt', sp_idx=3701),
    # pretraining_epochs=50, training_epochs=500, batch_size=50)
    test_SdA(load_data(r'..\data\data_all\acous_lex_avg.txt', sp_idx=23993),
             pretraining_epochs=50, training_epochs=500, batch_size=50)