__author__ = 'zxd'

"""
The multi-modal stacked denoising auto-encoders
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


class MSDA(object):
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
                        input_size = numpy.sum([hidden_layers_sizes[m][-1] for m in range(self.n_modals - 1)])
                        layer_input = T.concatenate(
                            [self.sigmoid_layers[m][-1].output for m in range(self.n_modals - 1)],
                            axis=1)
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

        # Construct logistic layer
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1][-1].output,
            n_in=hidden_layers_sizes[-1][-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # predict y
        self.y_pred = self.logLayer.y_pred

    def pretraining_functions(self, datasets_modals, batch_size):
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        givens = [(self.x[k], datasets_modals[k][0][0][batch_begin: batch_end]) for k in range(self.n_modals - 1)]
        for da_layers_modal in self.da_layers:
            for da in da_layers_modal:
                # get the cost and the updates list
                cost, updates = da.get_cost_updates(corruption_level, learning_rate)
                # compile the theano function
                fn = theano.function(
                    inputs=[
                        index,
                        theano.Param(corruption_level, default=0.2),
                        theano.Param(learning_rate, default=0.1)
                    ],
                    outputs=cost,
                    updates=updates,
                    givens=givens,
                    on_unused_input='warn'
                )
                # append `fn` to the list of functions
                pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets_modals, batch_size, learning_rate):
        index = T.lscalar('index')  # index to a [mini]batch
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_givens = [(self.x[k], datasets_modals[k][0][0][batch_begin: batch_end]) for k in range(self.n_modals - 1)]
        train_givens.append((self.y, datasets_modals[0][0][1][batch_begin: batch_end]))
        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens=train_givens,
            name='train'
        )

        valid_givens = [(self.x[k], datasets_modals[k][1][0][batch_begin: batch_end]) for k in range(self.n_modals - 1)]
        valid_score_i = theano.function(
            [index],
            self.y_pred,
            givens=valid_givens,
            name='valid'
        )

        return train_fn, valid_score_i

    def build_seq_output_functions(self, datasets_modals):
        train_givens = [(self.x[k], datasets_modals[k][0][0]) for k in range(self.n_modals - 1)]
        train_fn = theano.function(
            [], [self.sigmoid_layers[-1][-1].output, self.y_pred], givens=train_givens,
        )

        valid_givens = [(self.x[k], datasets_modals[k][1][0]) for k in range(self.n_modals - 1)]
        valid_fn = theano.function(
            [], [self.sigmoid_layers[-1][-1].output, self.y_pred], givens=valid_givens,
        )

        return train_fn, valid_fn


def test_mmsda(datasets_modals, finetune_lr=0.1, pretraining_epochs=15,
               pretrain_lr=0.001, training_epochs=1000, batch_size=1, seq_output=False):
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

    """

    # compute number of minibatches for training, validation
    n_train_batches = datasets_modals[0][0][0].get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = datasets_modals[0][1][0].get_value(borrow=True).shape[0] // batch_size + 1
    numpy_rng = numpy.random.RandomState(89677)
    print('... building the model')

    # construct the stacked denoising autoencoder class
    dim = [int(dataset[0][0].get_value(borrow=True).shape[1]) for dataset in datasets_modals]
    msda = MSDA(
        numpy_rng=numpy_rng,
        n_ins=dim,
        hidden_layers_sizes=[[600, 200], [150, 100], [200, 100]],
        n_outs=2
    )

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = msda.pretraining_functions(datasets_modals=datasets_modals,
                                                 batch_size=batch_size)
    print('... pre-training the model')
    start_time = time.clock()
    # Pre-train layer-wise
    corruption_levels = [.2, .2, .2, .2, .2, .2]
    for i, pt_fn in enumerate(pretraining_fns):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            corruption=corruption_levels[i],
                                            lr=pretrain_lr))
            print('Pre-training layer {0:d}, epoch {1:d}, cost {2:f}'.format(i, epoch, numpy.mean(c)))
    end_time = time.clock()
    print('The pretraining code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # imbalanced data
    penalty = [0.2, 1]
    penalty_sh = theano.shared(numpy.asarray([penalty] * batch_size, dtype=theano.config.floatX), borrow=True)
    msda.finetune_cost = msda.logLayer.negative_log_likelihood(msda.y, penalty_sh)

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn, validate_model = msda.build_finetune_functions(
        datasets_modals=datasets_modals,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )
    if seq_output:
        seq_output_train_fn, seq_output_valid_fn = msda.build_seq_output_functions(datasets_modals)

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
        y_real = datasets_modals[0][1][1].get_value(borrow=True)
        print(y_pred)
        fscore, precison, recall = f_score(y_real, y_pred)
        print('epoch {0:d}, fscore {1:f}  precision {2:f}  recall {3:f}'.format(epoch, fscore, precison, recall))

        # if we got the best validation score until now
        if fscore > best_fscore[0]:
            best_fscore = (fscore, precison, recall)
            print('-----Best score: {0:f}-----'.format(fscore))
            if seq_output:
                seq_output_train = seq_output_train_fn()
                seq_output_valid = seq_output_valid_fn()

    end_time = time.clock()
    print('Optimization complete with best validation score: fscore {0:f}  precision {1:f}  recall {2:f},'
          .format(best_fscore[0], best_fscore[1], best_fscore[2]))
    print('The training code for file ' +
          os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    if seq_output:
        print('writing sequence output')
        seq_output_file(r'..\data\data_seq\seq_train_raw.txt', seq_output_train,
                        datasets_modals[0][0][1].get_value(borrow=True))
        seq_output_file(r'..\data\data_seq\seq_test_raw.txt', seq_output_valid,
                        datasets_modals[0][1][1].get_value(borrow=True))


def seq_output_file(file, output_data, real_labels):
    f = open(file, 'w', encoding='utf-8')
    for feature, predict, real in zip(output_data[0], output_data[1], real_labels):
        f.write(' '.join(map(lambda i: '{0:.0f}'.format(i), feature)))
        f.write(' {0:.0f} {1:.0f}\n'.format(predict, real))
    f.close()


if __name__ == '__main__':
    # acoustic_data = load_data(r'..\data\data_balanced\acoustic.txt', sp_idx=3701)
    # text_data = load_data(r'..\data\data_balanced\lexical_vec_avg.txt', sp_idx=3701)
    # test_mmsda([acoustic_data, text_data], pretraining_epochs=50, training_epochs=500, batch_size=50)
    acoustic_data = load_data(r'..\data\data_all\acoustic.txt', sp_idx=23993)
    text_data = load_data(r'..\data\data_all\lexical_vec_avg.txt', sp_idx=23993)
    test_mmsda([acoustic_data, text_data], pretraining_epochs=50, training_epochs=500,
               batch_size=50, seq_output=False)
