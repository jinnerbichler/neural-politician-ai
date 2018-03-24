import logging
import pickle
from pathlib import Path

import spacy
import numpy as np

import sys
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from spacy.tokens import Doc, Span

import reader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('word_rnn')
logger.setLevel(logging.DEBUG)

SEQUENCE_LENGTH = 15


def train_generic():
    raw_text = reader.merge()
    raw_text = raw_text.lower()

    # raw_text = raw_text[:1000]

    # extract single sentences
    sents = []
    cached_sents = Path('./data/sents.pickle')
    if Path(cached_sents).exists():
        logger.info('Loading sentences from cache %s', cached_sents)
        with open(cached_sents, 'rb') as pickle_file:
            sents = pickle.load(pickle_file)
        logger.info('Loaded %d sentences from cache', len(sents))
    else:
        logger.info('Loading Spacy...')
        nlp = spacy.load('de')
        logger.info('Creating Spacy document...')
        doc = nlp(raw_text)  # type: Doc
        logger.info('Extracting sentences...')
        for sent in doc.sents:  # type: Span
            if sent.text.startswith('-') or len(sent) < 3:
                continue
            sents.append([word.text.lower() for word in sent])
        with open(cached_sents, 'wb') as pickle_file:
            pickle.dump(sents, pickle_file)
        logger.info('Extracted %d sentences', len(sents))

    # tokenize words
    logger.info('Tokenizing words...')
    joined_sentences = ' '.join([w for sent in sents for w in sent])
    tokenizer = Tokenizer(filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n', oov_token='UNK')
    tokenizer.fit_on_texts([joined_sentences])
    encoded = tokenizer.texts_to_sequences([joined_sentences])[0]
    vocab_size = (len(tokenizer.word_index))
    with open('./data/tokenizer.pickle', 'wb') as pickle_file:
        pickle.dump(tokenizer, pickle_file)
    logger.info('Tokenizied words. Len voc: %d', vocab_size)

    # create word sequences
    sequences = list()
    logger.info('Creating training sequences...')
    for i in range(SEQUENCE_LENGTH, len(encoded)):
        sequence = encoded[i - SEQUENCE_LENGTH:i + 1]
        sequences.append(sequence)
    with open('./data/sequences.pickle', 'wb') as pickle_file:
        pickle.dump(sequences, pickle_file)
    print('Created sequences. Total Sequences: %d' % len(sequences))

    return

    # split into x and y elements
    sequences = np.array(sequences)
    x, y = sequences[:, :-1], sequences[:, -1]

    # one hot encode outputs
    y = to_categorical(y, num_classes=vocab_size)

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
