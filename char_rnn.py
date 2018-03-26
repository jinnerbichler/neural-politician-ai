# coding: utf-8
import logging
import sys
import os
from functools import partial
from pathlib import Path

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback, Callback
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from tensorflow.python.client import device_lib
import reader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('rnn')
logger.setLevel(logging.DEBUG)

SEQUENCE_LENGTH = 100
TENSORBOARD_LOGS_DIR = os.getenv('TENSORBOARD_LOGS_DIR', '/tmp/logs')
MODELS_DIR = os.getenv('MODELS_DIR', './models/')


# noinspection PyBroadException
def create_model(input_shape, output_shape, weights_file=None):
    model = Sequential()
    # model.add(LSTM(256, input_shape=input_shape, return_sequences=False))  # ToDo: remove
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    # model.add(Dropout(rate=0.2))
    model.add(LSTM(128))
    # model.add(Dropout(rate=0.2))
    model.add(Dense(output_shape, activation='softmax'))

    # try load existing weights
    try:
        if weights_file and Path(weights_file).exists():
            logger.info('Reading weights from %s', weights_file)
            model.load_weights(filepath=weights_file)
            logger.info('Successfully read weights from %s', weights_file)
        else:
            logger.info('No stored weights found.')
    except:
        logger.exception('Cannot not read stored weights!')

    optimizer = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


def train_generic():
    raw_text = reader.merge()
    raw_text = raw_text.lower()

    # raw_text = raw_text[:500000]  # ToDO: remove

    # create mapping of unique chars to integers
    logger.info('Creating vocabulary...')
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # summarize the loaded data
    n_chars = len(chars)
    len_corpus = len(raw_text)
    logger.info('Total characters: %d', n_chars)
    logger.info('Total corpus: %d', len_corpus)

    # prepare the dataset of input to output pairs encoded as integers
    step = 1
    sequences = []
    next_chars = []
    logger.info('Creating sequences...')
    for i in range(0, len_corpus - SEQUENCE_LENGTH, step):
        sequences.append(raw_text[i: i + SEQUENCE_LENGTH])
        next_chars.append(raw_text[i + SEQUENCE_LENGTH])
    n_sequences = len(sequences)
    logger.info('Total sequences: %d', n_sequences)

    # vectorize characters to one-hot-encoding
    logger.info('Vectorization...')
    x = np.zeros((n_sequences, SEQUENCE_LENGTH, n_chars), dtype=np.bool)
    y = np.zeros((n_sequences, n_chars), dtype=np.bool)
    for i, sentence in enumerate(sequences):
        for t, char in enumerate(sentence):
            x[i, t, char_to_int[char]] = 1
        y[i, char_to_int[next_chars[i]]] = 1
    logger.info('Finished vectorization.')

    # create the RNN model
    stored_weights = str(Path(MODELS_DIR).joinpath('char_generic.hdf5').absolute())
    model = create_model(input_shape=(SEQUENCE_LENGTH, n_chars), output_shape=n_chars,
                         weights_file=stored_weights)
    model.summary()

    # define the checkpoint
    logger.info('Storing weights in %s', stored_weights)
    checkpoint_cb = ModelCheckpoint(stored_weights, monitor='loss', verbose=1,
                                    save_best_only=True, mode='min')

    # Tensorboard callback
    tensorboard_cb = TensorBoard(log_dir=TENSORBOARD_LOGS_DIR, write_graph=True)
    tensorboard_cb.set_model(model)

    # prediction/validation callback
    predict_cb = partial(epoch_end_prediction, model=model, text=raw_text,
                         n_chars=n_chars, char_to_int=char_to_int,
                         int_to_char=int_to_char)
    predict_cb = LambdaCallback(on_epoch_end=predict_cb)

    # learning rate decay
    # learning_rate_decay_db = LearningRateReducer(reduce_rate=0.1)

    # fit the model
    callbacks = [checkpoint_cb, tensorboard_cb, predict_cb]
    # callbacks = [checkpoint_cb, learning_rate_decay_db, tensorboard_cb, predict_cb]
    # model.fit(x=x, y=y, epochs=50, batch_size=64, callbacks=callbacks, verbose=1)
    model.fit(x=x, y=y, epochs=50, batch_size=256, callbacks=callbacks,
              verbose=1)  # ToDO: enable


# class LearningRateReducer(Callback):
#     def __init__(self, patience=0, reduce_rate=0.5, reduce_nb=10, verbose=1):
#         super(Callback, self).__init__()
#         self.patience = patience
#         self.wait = 0
#         self.best_loss = float(sys.maxsize)
#         self.reduce_rate = reduce_rate
#         self.current_reduce_nb = 0
#         self.reduce_nb = reduce_nb
#         self.verbose = verbose
#
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         current_loss = logs.get('loss')
#         if current_loss < self.best_loss:
#             self.best_loss = current_loss
#             self.wait = 0
#             if self.verbose > 0:
#                 logger.info('LearningRateReducer: current best loss: %.3f', current_loss)
#         else:
#             if self.wait >= self.patience:
#                 self.current_reduce_nb += 1
#                 if self.current_reduce_nb <= 10:
#                     lr = float(K.get_value(self.model.optimizer.lr))
#                     lr *= (1.0 - self.reduce_rate)
#                     logger.info('LearningRateReducer: decaying learning rate to %f', lr)
#                     K.set_value(self.model.optimizer.lr, lr)
#                 else:
#                     if self.verbose > 0:
#                         logger.info(
#                             "LearningRateReducer: epoch %d: early stopping" % epoch)
#                     self.model.stop_training = True
#             self.wait += 1


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def epoch_end_prediction(epoch, logs, model, text, char_to_int, int_to_char, n_chars):
    # Function invoked at end of each epoch. Prints generated text.

    # start_index = np.random.randint(0, len(text) - SEQUENCE_LENGTH - 1)  # ToDO: enable
    start_index = 0
    start_sequence = text[start_index: start_index + SEQUENCE_LENGTH]
    logger.info('---- Generating text after Epoch: %d Seed: %s' % (epoch, start_sequence))
    for diversity in [0.2, 0.5, 1.0, 1.2]:

        sentence = start_sequence
        generated = sentence

        for i in range(400):
            x_pred = np.zeros((1, SEQUENCE_LENGTH, n_chars))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_int[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = int_to_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

        logger.info('---- Generated text (diversity: %s): %s', diversity, generated)
        sys.stdout.flush()


if __name__ == '__main__':
    local_device_protos = device_lib.list_local_devices()
    logger.info('Detected devices: {}'.format([d.name for d in local_device_protos]))
    train_generic()
