import glob
import itertools
import logging
import re
import sys
from pathlib import Path

from django.http import JsonResponse
import tensorflow as tf
import numpy as np
from keras import Sequential
from keras import backend as K
from keras.models import load_model

from intelligence.speech_data import SpeechSequence
import intelligence.speech_data as speech_data
from intelligence.word_rnn import sample_word

logger = logging.getLogger(__name__)

MODELS = {}
GRAPHS = {}
SESSIONS = {}
VOCAB = None  # type: SpeechSequence
SEQUENCE_LEN = 15
MIN_NUM_GENERATED = 250


def init_models():
    global MODELS, VOCAB

    # load applied vocabulary
    sys.modules['speech_data'] = speech_data
    VOCAB = SpeechSequence.load(path='./intelligence/data/dataset.pickle')

    # load pretrained models
    for filepath in glob.iglob('./intelligence/models/*.h5'):
        politician = ''.join(Path(filepath).name.split('.')[:-1])
        # if politician != 'kurz':
        #     continue
        logger.info('Loading model {}'.format(filepath))

        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph).as_default() as session:
                MODELS[politician] = load_model(filepath)
                GRAPHS[politician] = graph
                SESSIONS[politician] = session
        K.clear_session()


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
