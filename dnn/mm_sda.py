__author__ = 'zxd'

"""
The multi-modal stacked denoising auto-encoders
"""
import os
import time

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from dnn.logistic_sgd import LogisticRegression
from dnn.data_reader import load_data
from dnn.mlp import HiddenLayer
from dnn.da import DA


class MMSDA(object):
    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            n_ins=[784, 784],
            hidden_layers_sizes=[[500, 500], [500, 500], [500]],
            n_outs=10,
    ):
        # last modal is fusion layers
        self.n_modals = len(hidden_layers_sizes)
        self.n_layers = [len(size) for size in hidden_layers_sizes]

        self.sigmoid_layers = []
        self.da_layers = []
        for i in range(self.n_modals):
            self.sigmoid_layers.append([])
            self.da_layers.append([])
        self.params = []

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = [T.matrix('x' + str(i)) for i in range(self.n_modals - 1)]
        self.y = T.ivector('y')

        for k in range(self.n_modals):
            for i in range(self.n_layers[k]):
                # construct the sigmoidal layer
                # the input to this layer is either the activation of the hidden
                # layer below or the input of the SdA if you are on the first layer

                if i == 0:
                    if k != self.n_modals - 1:  # individual modal
                        input_size = n_ins[k]
                        layer_input = self.x[k]
                    else:  # fusion layers
                        input_size = numpy.sum([size[-1]] for size in hidden_layers_sizes)
                        # TODO
                        layer_input =
                else:
                    input_size = hidden_layers_sizes[k][i - 1]
                    layer_input = self.sigmoid_layers[k][-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[k][i],
                                        activation=T.nnet.sigmoid)  # add the layer to our list of layers
            self.sigmoid_layers[k].append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this layer
            da_layer = DA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[k][i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.da_layers[k].append(da_layer)

    # Construct fusion layer


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
    for da in self.da_layers:
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
        hidden_layers_sizes=[400, 400, 400],
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

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn, validate_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('... finetunning the model')
    best_fscore = 0
    start_time = time.clock()

    epoch = 0

    while epoch < training_epochs:
        epoch += 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)

        # compute f-score on validation set
        y_preds = [validate_model(i) for i in range(n_valid_batches)]
        y_pred = [pij for pi in y_preds for pij in pi]
        y_real = valid_set_y.get_value(borrow=True)
        fscore, precison, recall = f_score(y_real, y_pred)
        print('epoch {0:d}, fscore {1:f}  precision {2:f}  recall {3:f}'.format(epoch, fscore, precison, recall))

        # if we got the best validation score until now
        if fscore > best_fscore:
            best_fscore = fscore
            print('-----Best score: {0:f}-----'.format(best_fscore))

    end_time = time.clock()
    print('Optimization complete with best validation score of {0:.1f} %,'
          .format(best_fscore * 100.))
    print('The training code for file ' +
          os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.))


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
    return fscore, precison, recall


if __name__ == '__main__':
    test_SdA(load_data(r'..\data\data_balanced\lexical_vec_bigram.txt'), pretraining_epochs=50, training_epochs=300,
             batch_size=50)
