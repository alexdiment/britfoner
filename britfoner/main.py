#!/usr/bin/env python3
'''

script to train model

'''


from os.path import join
from typing import Dict, Any, Tuple
import logging
from keras.callbacks import EarlyStopping
from keras.models import Model

from britfoner import _UNSTRESSED_BRITFONE, _MODEL_OUT
from britfoner.IO import decoded, dataset_from
from britfoner.g2p import train_g2p, most_likely_sequence, \
    attention_g2p_model_from, WER_ModelCheckpoint


def main_seq_2_seq(data_src: str = _UNSTRESSED_BRITFONE, model_src: str = None) -> Tuple[Model, str]:
    '''
    Creates, trains and saves a sequence to sequence model

    :param data_src: file containing data
    :param model_src: file containing previously trained model, to start the training from
    :return: the trained model together withe file name it has been saved to
    '''
    (train_X, val_X, train_Y, val_Y), index = dataset_from(data_src, val_size=.01)

    model = attention_g2p_model_from(index.x_dim, index.x_n, index.y_dim, index.y_n, dropout=.15)

    if model_src is not None:
        model.load_weights(join(_MODEL_OUT, model_src))

    on_epoch_end = epoch_publishing_fn_from(val_X, model, index)

    name = model_name_from(model)
    callbacks = [
        EarlyStopping(patience=35),
        WER_ModelCheckpoint(filepath=join(_MODEL_OUT, name),
                            verbose=0,
                            monitor='WER',
                            save_best_only=True,
                            callback=on_epoch_end)]

    logging.info(f'starting training with a [{len(train_X)}/{len(val_X)}] training/validation split...')
    model = train_g2p(model, (train_X, train_Y), (val_X, val_Y), epochs=5000, callbacks=callbacks)
    logging.info('finished training.')

    model.load_weights(join(_MODEL_OUT, name))

    end_publishing_fn_from(val_X, model, index)(None)

    return model, name


def model_name_from(model: Any) -> str:
    '''
    Creates file name to save model to

    :param model: the sequence to sequence model
    :return: the file name
    '''
    k = len(model.layers)

    input, hidden, output = model.layers[0].output_shape, model.layers[1].output_shape, model.layers[k - 1].output_shape

    return f'{input[1]}x{input[2]}x{hidden[2]}x{output[1]}x{output[2]}x{int((k-3)/2)}.h5'


def epoch_publishing_fn_from(val_X, model, index, period=10):
    '''
    Creates a function to publish state of model during training

    :param val_X: validation set input sequences
    :param model: the training model
    :param index: the dataset index
    :param period: the frequency to publish training data at
    :return: the function
    '''
    fraction = 100 / len(val_X)

    def on_epoch_end(epoch: int, logs: Dict[str, Any]):

        errors = 0.
        for x, y_hat in zip(val_X, model.predict(val_X)):
            sound_hat = most_likely_sequence(y_hat, index.inv_phone)

            word = decoded(x, index.inv_letter, reverse=True)

            sounds = index.word_to_sounds[word]
            error = +(sound_hat not in sounds)

            errors += error

        if epoch % period == 0:
            logging.info(f'[{epoch:04d}] WER [{fraction * errors:6.2f}], val. loss [{logs["val_loss"]:1.5f}]')

        return fraction * errors

    return on_epoch_end


def end_publishing_fn_from(val_X, model, index):
    '''
    Creates a function to publish state of model after training is finished

    :param val_X: validation set input sequences
    :param model: the training model
    :param index: the dataset index
    :return: the function
    '''
    fraction = 100 / len(val_X)

    def on_train_end(logs: Dict[str, Any]):

        logging.info('errors:')

        errors = 0.
        for x, y_hat in zip(val_X, model.predict(val_X)):

            sound_hat = most_likely_sequence(y_hat, index.inv_phone)

            word = decoded(x, index.inv_letter, reverse=True)

            sounds = index.word_to_sounds[word]
            error = +(sound_hat not in sounds)

            errors += error

            if error:  logging.info(f'\t{"".join(word)}\t{" ".join(sound_hat)}')

        logging.info(f'WER [{fraction*errors:0.2f}]')

    return on_train_end


if __name__ == '__main__':
    main_seq_2_seq()
