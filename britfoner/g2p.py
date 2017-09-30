'''

Sequence to sequence model building and training

'''
from typing import Tuple, Callable

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from numpy import ndarray, argmax

from britfoner import Seq, _symbols, Inv_Alphabet
from .seq2seq.models import AttentionSeq2Seq


def attention_g2p_model_from(input_dim: int,
                             input_length: int,
                             output_dim: int,
                             output_length: int,
                             hidden_n: int = 256,
                             dropout=.1,
                             depth = 1) \
        -> AttentionSeq2Seq:
    '''
    Creates a sequence to sequence model with attention

    :param input_dim: number of symbols in input alphabet (including end, start and padding)
    :param input_length: length of longest input sequence
    :param output_dim: number of symbols in output alphabet (including end, start and padding)
    :param output_length: length of longest output sequence
    :param hidden_n: number of hidden units
    :param dropout: dropout rate
    :param depth: depth of rnn stack
    :return: the created, compiled model
    '''
    model = AttentionSeq2Seq(output_dim=output_dim,
                             output_length=output_length,
                             hidden_dim=hidden_n,
                             input_dim=input_dim,
                             input_length=input_length,
                             unroll= dropout == 0.,
                             dropout= dropout,
                             depth=depth)

    model.compile(loss='mse', optimizer=Adam(lr= 1e-3, decay=1e-6))

    return model


def train_g2p(model: AttentionSeq2Seq,
              train_set: Tuple[ndarray, ndarray],
              val_set: Tuple[ndarray, ndarray],
              batch_n: int = 128,
              epochs: int = 100,
              callbacks: Callable = None) -> AttentionSeq2Seq:
    '''
    Trains given model

    :param model: sequence to sequence model
    :param train_set: training data as X, Y tuple of tensors
    :param val_set: validation data as X, Y tuple of tensors
    :param batch_n: batch size
    :param epochs: number of epochs
    :param callbacks: callbacks for keras training
    :return: trained model
    '''
    train_X, train_Y = train_set
    val_X, val_Y = val_set

    model.fit(train_X, train_Y,
              validation_data=(val_X, val_Y),
              batch_size=batch_n,
              epochs=epochs,
              callbacks=callbacks,
              verbose=0)

    return model


def most_likely_sequence(y_hat: ndarray, inv_alphabet: Inv_Alphabet) -> Seq:
    '''
    Returns the most likely sequence for the given prediced output vector. The decoding
    algorithm is greedy, picking the highest scored symbol in the output

    Any padding symbols are removed

    :param y_hat: the predicted output vector
    :param inv_alphabet: a sorted list of symbols representing the output alphabet
    :return: the predicted output sequence
    '''

    return tuple(inv_alphabet[argmax(t)] for t in y_hat if inv_alphabet[argmax(t)] not in _symbols)


class WER_ModelCheckpoint(ModelCheckpoint):
    '''
    Ensures saving of training model is done every time the Word Error Rate (WER)
    improves. This allows the final saved model to be the best one according to the
    WER metric
    '''
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, callback=None):

        super().__init__(filepath=filepath, monitor=monitor, verbose=verbose,
                         save_best_only=save_best_only, save_weights_only=save_weights_only,
                         mode=mode, period=period)

        self.callback = callback

    #hack to ensure the monitored quantity is the WER rather than the loss/metric
    def on_epoch_end(self, epoch, logs=None):
        WER = self.callback(epoch, logs)
        logs['WER'] = WER

        super().on_epoch_end(epoch, logs)
