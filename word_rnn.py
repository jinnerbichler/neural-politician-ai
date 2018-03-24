import logging
import sys
from pathlib import Path

import numpy as np
import os
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import speech_data

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('word_rnn')
logger.setLevel(logging.DEBUG)

TENSORBOARD_LOGS_DIR = os.getenv('TENSORBOARD_LOGS_DIR', './models/')
SEQUENCE_LENGTH = 15


def train_generic():

    # get extracted sentences
    sentences = speech_data.extract_sentences(try_cached=True)

    # tokenize words
    logger.info('Tokenizing words...')
    joined_sentences = ' '.join([w.lower() for sent in sentences for w in sent.words])
    tokenizer = Tokenizer(filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n', oov_token='UNK')
    tokenizer.fit_on_texts([joined_sentences])
    encoded = tokenizer.texts_to_sequences([joined_sentences])[0]
    vocab_size = (len(tokenizer.word_index))
    logger.info('Tokenizied words. Len voc: %d', vocab_size)

    # storing metadata for TensorBoard
    embeddings_file = Path(TENSORBOARD_LOGS_DIR).joinpath('embeddings_meta.txt')
    with open(embeddings_file, 'w') as embeddings_file:
        lines = ['Word\tIndex\tCount']
        lines += ['{}\t{}\t{}'.format(w, i, tokenizer.word_counts.get(w, 0))
                  for w, i in tokenizer.word_index.items()]
        lines += 'dummy\t{}'.format(vocab_size)  # fix issue with different base indexes
        embeddings_file.write('\n'.join(lines))

    # create word sequences
    sequences = list()
    logger.info('Creating training sequences...')
    for i in range(SEQUENCE_LENGTH, len(encoded)):
        sequence = encoded[i - SEQUENCE_LENGTH:i + 1]
        sequences.append(sequence)
    logger.info('Created sequences. Total Sequences: %d' % len(sequences))

    # split into x and y elements
    sequences = np.array(sequences)
    x, y = sequences[:, :-1], sequences[:, -1]

    # one hot encode outputs
    y = to_categorical(y, num_classes=vocab_size)

    # preparate data
    logger.info('Prepared data. Shape x: {}, y: {}'.format(x.shape, y.shape))

    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 10, input_length=SEQUENCE_LENGTH))
    model.add(LSTM(256, return_sequences=True))
    # model.add(Dropout(rate=0.2))
    model.add(LSTM(256))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())

    # compile network
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(x, y, epochs=500, verbose=1, batch_size=1, callbacks=[])


if __name__ == '__main__':
    train_generic()
