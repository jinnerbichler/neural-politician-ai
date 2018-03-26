import codecs
import logging
import os
import sys
from functools import partial
from pathlib import Path

from keras import Sequential, Model
from keras.callbacks import TensorBoard, LambdaCallback, ModelCheckpoint
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam
from tensorflow.python.client import device_lib

import speech_data
import numpy as np
from speech_data import SpeechSequence

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('word_rnn')
logger.setLevel(logging.DEBUG)

TENSORBOARD_LOGS_DIR = os.getenv('TENSORBOARD_LOGS_DIR', './graph/')
MODELS_DIR = os.getenv('MODELS_DIR', './models/')

SEQUENCE_LENGTH = 15
VOCAB_SIZE = 5000
BATCH_SIZE = 2


# noinspection PyBroadException
def create_model(vocab_size, sequence_length, weights_file=None):
    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 10, input_length=sequence_length))
    model.add(LSTM(256, return_sequences=True))
    # model.add(Dropout(rate=0.2))
    model.add(LSTM(256))
    model.add(Dense(vocab_size, activation='softmax'))

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

    # compile network
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def train_generic():
    # get extracted sentences
    sentences = speech_data.extract_sentences(try_cached=True)
    sentences = sentences[:40]
    dataset = SpeechSequence(sentences=sentences, batch_size=BATCH_SIZE,
                             vocab_size=VOCAB_SIZE, sequence_length=SEQUENCE_LENGTH)

    # storing metadata for TensorBoard
    embeddings_path = Path(TENSORBOARD_LOGS_DIR).joinpath('embeddings_meta.txt')
    embeddings_path.parent.mkdir(exist_ok=True)
    with codecs.open(str(embeddings_path), 'w', "utf-8") as embeddings_file:
        lines = ['Word\tIndex\tCount']
        lines += ['{}\t{}\t{}'.format(w, i, dataset.tokenizer.word_counts.get(w, 0))
                  for w, i in dataset.tokenizer.word_index.items() if i <= VOCAB_SIZE]
        # fix issue with different base indexes
        # lines += ['dummy\t{}\t0'.format(VOCAB_SIZE + 1)]
        embeddings_file.write('\n'.join(lines))

    # preparate data
    logger.info('Prepared data with length %d', len(dataset))

    # create the RNN model
    stored_weights = str(Path(MODELS_DIR).joinpath('word_generic.hdf5').absolute())
    model = create_model(vocab_size=VOCAB_SIZE, sequence_length=SEQUENCE_LENGTH,
                         weights_file=stored_weights)
    model.summary()

    # define the checkpoint
    logger.info('Storing weights in %s', stored_weights)
    checkpoint_cb = ModelCheckpoint(stored_weights, monitor='loss', verbose=1,
                                    save_best_only=True, mode='min')

    # Tensorboard callack
    tensorboard_cb = TensorBoard(log_dir=TENSORBOARD_LOGS_DIR, write_graph=True,
                                 embeddings_metadata=str(embeddings_path.absolute()),
                                 embeddings_freq=1)
    tensorboard_cb.set_model(model)

    # prediction/validation callback
    predict_cb = partial(epoch_end_prediction, model=model, dataset=dataset)
    predict_cb = LambdaCallback(on_epoch_end=predict_cb)

    # learning rate decay
    # learning_rate_decay_db = LearningRateReducer(reduce_rate=0.1)

    # fit the model
    callbacks = [checkpoint_cb, tensorboard_cb, predict_cb]

    model.fit_generator(generator=dataset, epochs=500, verbose=1, callbacks=callbacks,
                        shuffle=True)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# noinspection PyUnusedLocal
def epoch_end_prediction(epoch, logs, model, dataset):
    # type: (int, dict, Model, SpeechSequence) -> None
    # Function invoked at end of each epoch. Prints generated text.

    # start_index = np.random.randint(0, len(text) - SEQUENCE_LENGTH - 1)  # ToDO: enable
    start_index = 0
    start_sequence = dataset.encoded[start_index: start_index + SEQUENCE_LENGTH]
    decoded_string = dataset.decode_string(start_sequence)
    logger.info('---- Generating text after Epoch: %d Seed: %s' % (epoch, decoded_string))
    for diversity in [0.2, 0.5, 1.0, 1.2]:

        current_sequence = start_sequence.copy()
        generated = start_sequence.copy()

        for i in range(400):
            x_pred = np.array([current_sequence])
            preds = model.predict(x_pred, verbose=0)[0]
            next_word_id = sample(preds, diversity)

            generated += [next_word_id]
            current_sequence = current_sequence[1:] + [next_word_id]

        decoded_string = dataset.decode_string(generated)
        logger.info('---- Generated text (diversity: %s): %s', diversity, decoded_string)
        sys.stdout.flush()


if __name__ == '__main__':
    local_device_protos = device_lib.list_local_devices()
    logger.info('Detected devices: {}'.format([d.name for d in local_device_protos]))
    train_generic()
