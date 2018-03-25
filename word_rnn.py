import codecs
import logging
import os
import sys
from pathlib import Path

from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam

import speech_data
from speech_data import SpeechSequence

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('word_rnn')
logger.setLevel(logging.DEBUG)

TENSORBOARD_LOGS_DIR = os.getenv('TENSORBOARD_LOGS_DIR', './models/')

SEQUENCE_LENGTH = 15
VOCAB_SIZE = 5000
BATCH_SIZE = 2


def train_generic():
    # get extracted sentences
    sentences = speech_data.extract_sentences(try_cached=True)
    dataset = SpeechSequence(sentences=sentences, batch_size=BATCH_SIZE,
                             num_words=VOCAB_SIZE, sequence_length=SEQUENCE_LENGTH)

    # storing metadata for TensorBoard
    embeddings_file = Path(TENSORBOARD_LOGS_DIR).joinpath('embeddings_meta.txt')
    with codecs.open(str(embeddings_file), 'w', "utf-8") as embeddings_file:
        lines = ['Word\tIndex\tCount']
        lines += ['{}\t{}\t{}'.format(w, i, dataset.tokenizer.word_counts.get(w, 0))
                  for w, i in dataset.tokenizer.word_index.items() if i <= VOCAB_SIZE]
        # fix issue with different base indexes
        # lines += ['dummy\t{}\t0'.format(VOCAB_SIZE + 1)]
        embeddings_file.write('\n'.join(lines))

    # preparate data
    logger.info('Prepared data with length %d', len(dataset))

    # define model
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, 10, input_length=SEQUENCE_LENGTH))
    model.add(LSTM(256, return_sequences=True))
    # model.add(Dropout(rate=0.2))
    model.add(LSTM(256))
    model.add(Dense(VOCAB_SIZE, activation='softmax'))
    print(model.summary())

    # compile network
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit_generator(generator=dataset, epochs=500, verbose=1, callbacks=[])


if __name__ == '__main__':
    train_generic()
