import glob
import itertools
import logging
import pickle
import re
from pathlib import Path

from django.http import JsonResponse
import tensorflow as tf
import numpy as np
from keras import Sequential
from keras import backend as K
from keras.models import load_model
import spacy

logger = logging.getLogger(__name__)

MODELS = {}
GRAPHS = {}
SESSIONS = {}
VOCAB = None  # type: SpeechVocabulary
UNK = '<UNK>'
SEQUENCE_LEN = 15
MIN_NUM_GENERATED = 100


class SpeechVocabulary:

    def __init__(self, vocab):
        self.input_vocab = vocab['input']
        self.input_word_ids = {v: k for k, v in self.input_vocab.items()}
        self.input_unk_id = len(self.input_vocab) - 1
        self.output_vocab = vocab['output']
        self.output_word_ids = {v: k for k, v in self.output_vocab.items()}
        self.output_unk_id = len(self.output_vocab) - 1

    def encode_input(self, words):
        return [self.input_vocab.get(w.lower(), self.input_unk_id) for w in words]

    def decode_input(self, encoded):
        return [self.input_word_ids[e] for e in encoded]

    def encode_output(self, words):
        return [self.output_vocab.get(w.lower(), self.output_unk_id) for w in words]

    def decode_output(self, encoded):
        return [self.output_word_ids[e] for e in encoded]

    def out_to_in(self, word_id):
        word = self.output_word_ids.get(word_id, UNK)
        return self.input_vocab.get(word, self.input_unk_id)


def init_models():
    global MODELS, VOCAB

    # load pretrained models
    for filepath in glob.iglob('./backend/trained_models/*.h5'):
        politician = ''.join(Path(filepath).name.split('.')[:-1])
        logger.info('Loading model {}'.format(filepath))

        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph).as_default() as session:
                MODELS[politician] = load_model(filepath)
                GRAPHS[politician] = graph
                SESSIONS[politician] = session
        K.clear_session()

    # load applied vocabulary
    with open('./backend/trained_models/raw_vocab.pickle', 'rb') as pickle_file:
        vocab_data = pickle.load(pickle_file)
        VOCAB = SpeechVocabulary(vocab_data)


def sample_word(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_speech(request):
    politician = request.GET['politician']
    start_text = request.GET['start_text']
    logger.info('Generating speech for {}'.format(politician))

    # pre-process input text
    for s in [',', '.', '!']:
        start_text = start_text.replace(s, ' ' + s)
    input_sequence = start_text.lower().split()
    current_input = VOCAB.encode_input(input_sequence)

    # adpat size of input sequence
    current_input = current_input[-SEQUENCE_LEN:]
    num_padding = SEQUENCE_LEN - len(current_input)
    current_input = [VOCAB.input_unk_id] * num_padding + current_input

    generated_text = input_sequence.copy()

    model = MODELS[politician]  # type: Sequential
    with GRAPHS[politician].as_default():
        with SESSIONS[politician].as_default():
            # generate words until dot after minimum number of words was passed.
            for word_iter in itertools.count():

                # predict id of next word
                x_pred = np.array([current_input])
                preds = model.predict(x_pred, verbose=0)[0]
                preds = preds[:-1]  # remove last entry, which represents unkown words
                next_output_word_id = sample_word(preds, 0.2)

                # convert id to word
                next_word = VOCAB.output_word_ids[next_output_word_id]
                generated_text.append(next_word)

                # adapt new input sequence
                next_input_word_id = VOCAB.out_to_in(word_id=next_output_word_id)
                current_input = current_input[1:] + [next_input_word_id]

                # check if sentence has finished
                if word_iter > MIN_NUM_GENERATED and next_word == '.':
                    break

    # merge words
    speech = ' '.join(generated_text)
    for match in re.findall(r'( ([,|.|!]))', speech):
        speech = speech.replace(match[0], match[1])
    return JsonResponse({'speech': speech, 'politician': politician})
