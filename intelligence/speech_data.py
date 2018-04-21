# coding: utf-8
import pickle
import time
import urllib
from pathlib import Path
from typing import List

import spacy
import dropbox
import feedparser
import re
import unicodedata
import itertools
import logging
from datetime import datetime
from time import mktime
import requests
import sys
from bs4 import BeautifulSoup
from collections import defaultdict, namedtuple, OrderedDict, Counter
import os

from spacy.tokens import Doc
import numpy as np
from keras.utils import to_categorical, Sequence

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger('scraper')
logger.setLevel(logging.DEBUG)

DROPBOX_TOKEN = os.getenv('DROPBOX_TOKEN', '')
DROPBOX_SESSION_PATH = Path('/neural-politician') \
    .joinpath(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

PARLAMENT_BASE_URL = 'https://www.parlament.gv.at'
POLITICIANS = ['kurz', 'kern', 'strache', 'strolz']
SPEECHES_PICKLE = './data/speeches.pickle'
VOCAB_VECTORS = './data/word_vectors.pickle'
DATASET_FILE = './data/dataset.pickle'

PERIOD_FEEDS = {
    'XXIV': 'https://www.parlament.gv.at/PAKT/PLENAR/filter.psp?view=RSS&RSS=RSS&jsMode=RSS&xdocumentUri=%2FPAKT%2FPLENAR%2Findex.shtml&view=RSS&NRBRBV=NR&GP=XXIV&R_SISTEI=SI&LISTE=Anzeigen&listeId=1070&FBEZ=FP_007',
    'XXV': 'https://www.parlament.gv.at/PAKT/PLENAR/filter.psp?view=RSS&RSS=RSS&jsMode=RSS&xdocumentUri=%2FPAKT%2FPLENAR%2Findex.shtml&view=RSS&NRBRBV=NR&GP=XXV&R_SISTEI=SI&LISTE=Anzeigen&listeId=1070&FBEZ=FP_007',
    'XXVI': 'https://www.parlament.gv.at/PAKT/PLENAR/filter.psp?view=RSS&RSS=RSS&jsMode=RSS&xdocumentUri=%2FPAKT%2FPLENAR%2Findex.shtml&view=RSS&NRBRBV=NR&GP=XXVI&R_SISTEI=SI&LISTE=Anzeigen&listeId=1070&FBEZ=FP_007',
}

Sentence = namedtuple('Sentence', ['words', 'politician', 'speech_id', 'sent_id'])
WordVector = namedtuple('WordVector', ['id', 'word', 'vector'])


def collect():
    """
    Fetches the RSS feed for each period and extracts protocols of executed sessions.
    Collected speeches a stored in a pickle file for later usage.
    """
    all_speeches = defaultdict(list)
    for period, feed_url in PERIOD_FEEDS.items():

        logger.info('Processing period {} ({})'.format(period, feed_url))

        feed = feedparser.parse(feed_url)
        fetched_sessions = []  # avoid fetching a session twice
        for session_iter, session in enumerate(reversed(feed['items']), start=1):

            # extract session information
            title = session['title']
            published = datetime.fromtimestamp(mktime(session['published_parsed']))
            session_url = session['link']

            # check if sessions has already been fetched
            if title in fetched_sessions:
                continue
            fetched_sessions.append(title)

            logger.info('Fetching session "{}" {}/{} ({})'.format(
                title, session_iter, len(feed['items']), session_url))

            # fetch description of session
            response = requests.get(session_url)
            soup = BeautifulSoup(response.text, 'html5lib')
            lists = soup.find_all('li')

            # check if protocol is available
            for li in [l for l in lists if 'Stenographisches Protokoll' in str(l)]:
                for a in [a for a in li.find_all('a') if 'html' in str(a).lower()]:
                    # parse protocol
                    protocol_url = PARLAMENT_BASE_URL + a.attrs['href']
                    logger.debug('Fetching protocol {}'.format(protocol_url))
                    sessions_speeches = parse_protocol(protocol_url)

                    # enrich extracted speeches with session information
                    for politication, speeches in sessions_speeches.items():
                        for speech in speeches:
                            speech.update({'session': {'period': period,
                                                       'title': title,
                                                       'published': published,
                                                       'url': session_url}})
                        all_speeches[politication].extend(speeches)

            num_speeches = sum([len(s) for s in all_speeches.values()])
            logger.info('Current speech count: {}'.format(num_speeches))

        # store speeches
        with open(SPEECHES_PICKLE, 'wb') as pickle_file:
            pickle.dump(all_speeches, pickle_file)

        # convert to separate text file
        split()

    num_speeches = sum([len(s) for s in all_speeches.values()])
    logger.info('Total speech count: {}'.format(num_speeches))


def parse_protocol(url):
    """
    :param url: Url of created protocol
    :return: dictionary, which maps a politician to a list of speeches
    """

    # fetch protocol
    response = requests.get(url)
    response_text = response.text.replace('&shy;', '')  # remove hyphens
    soup = BeautifulSoup(response_text, 'html5lib')

    speeches = defaultdict(list)
    # first n sections a part of the table of content
    for section_iter in itertools.count(start=3):
        # extract relevant pargraphs
        section = soup.find('div', class_='WordSection{}'.format(section_iter))
        if not section:
            break
        first_paragraph = section.find('p', class_='StandardRB')
        other_paragraphs = section.find_all('p', class_='MsoNormal')

        # extract speech
        speech = first_paragraph.get_text() if first_paragraph else ''
        for paragraph in other_paragraphs:
            speech = '{} {}'.format(speech, paragraph.get_text())
        speech = unicodedata.normalize('NFKC', speech)
        speech = speech.replace('\n', ' ')

        # extract name and role
        prefix = re.match(r'^(.*?): ', speech)
        prefix = prefix.group() if prefix else ''
        speech = speech.replace(prefix, '')
        prefix = prefix.strip()
        match = re.match(r'^([\w\-]+) (.*?)[(|:]', prefix)
        role = name = None
        if match:
            role = match.group(1).strip()
            name = match.group(2).strip()
        party = re.search(r'\((.*?)\)', prefix)
        party = party.group(1) if party else None

        # remove parenthesis in speech
        speech = re.sub(r'\([^)]*\)', '', speech)
        speech = re.sub(' +', ' ', speech)  # remove double spaces

        section_iter += 1
        if not role or not name or not speech:
            continue

        # collect speeches of targeted politicians
        for politician in POLITICIANS:
            # 'Kurzmann' collides with 'Kurz'
            if politician in name.lower() and 'Kurzmann' not in name:
                logger.debug('Found speech (name: {}, role: {}, party: {})'.format(
                    name, role, party))
                speeches[politician].append({'name': name, 'role': role,
                                             'party': party, 'speech': speech})
    return speeches


def split():
    """
    Loads pickleds speeches and splits them in to separate
    textfiles (i.e. on per politician).
    """
    with open(SPEECHES_PICKLE, 'rb') as pickle_file:
        speeches = pickle.load(pickle_file)

    for politician, speeches in speeches.items():
        filename = './data/{}.txt'.format(politician)
        with open(filename, 'wt', encoding='utf8') as speeches_file:
            num_char = 0
            num_words = 0
            for speech in speeches:
                # write header and text of speech to file
                session = speech['session']
                header = '# {period:} - {title:} am {published:} ({url:})\n'.format(
                    **session)
                speeches_file.write(header)

                # write speech
                speech_text = speech['speech'].replace('- - ', '')  # parenthesis artifcat
                speeches_file.write(speech_text + '\n\n')

                # count metrics
                num_char += len(speech['speech'])
                num_words += len(speech['speech'].split())

            logger.info('Metrics of {}: chars: {}, words: {}'.format(
                politician, num_char, num_words))


def read_speeches(politician):
    # type: (str) -> List(str)
    single_speeches = []
    with open('./data/{}.txt'.format(politician), 'rt', encoding='utf8') as speeches_file:
        for line in speeches_file.readlines():
            # ignore comments and empty lines
            if line.startswith('#') or len(line) < 2:
                continue

            # clean speech text
            speech = re.sub(r'\[[^)]*\]', '', line)  # remove []
            speech = speech.replace('\\', '')  # replace @ sign
            speech = speech.replace('@', 'at')  # replace @ sign
            speech = speech.replace('&', 'und')  # replace and sigh
            # speech = speech.replace('?', '.')  # replace question mark
            # speech = speech.replace('!', '.')  # replace exlamation mark
            speech = speech.replace('\n', '')  # remove new line
            speech = speech.replace('(', '').replace(')', '')  # remove last parenthesis
            speech = speech.replace('%', 'Prozent')  # replace percentage sign
            speech = speech.replace('_i', 'I')  # replace gender-related underscores
            speech = speech.replace('*', '')  # remove invalid star
            speech = speech.replace('+', '')  # remove invalid plus
            speech = speech.replace('’', '')  # replace appostrove
            speech = speech.replace('‘', '')  # replace appostrove
            speech = speech.replace('`', '')  # replace appostrove
            speech = speech.replace('“', '\'')  # replace appostrove
            speech = speech.replace('„', '\'')  # replace appostrove
            speech = speech.replace('–', '-')  # replace proper hyphen
            speech = speech.replace('‐', '-')  # replace proper hyphen
            speech = speech.replace('§', '')  # remove paragrap sign
            speech = speech.replace('‚', ',')  # replace poper comma
            speech = speech.replace(';', ',')  # replace poper semi colon
            speech = speech.replace('ê', 'e')  # remove invalid derivative of e
            speech = speech.replace('é', 'e')  # remove invalid derivative of e
            speech = speech.replace('à', 'a')  # remove invalid derivative of a
            speech = speech.replace('á', 'a')  # remove invalid derivative of a
            speech = speech.replace('í', 'i')  # remove invalid derivative of i
            speech = speech.replace('ć', 'c')  # remove invalid derivative of c
            speech = speech.replace('ğ', 'g')  # remove invalid derivative of g
            speech = speech.replace('ń', 'n')  # remove invalid derivative of c
            speech = speech.replace('š', 's')  # remove invalid derivative of s
            speech = speech.replace('ž', 'z')  # remove invalid derivative of z
            speech = re.sub(' +', ' ', speech)  # remove consecutive spaces

            single_speeches.append(speech)
    return single_speeches


def extract_sentences(try_cached=True):
    # type: (bool) -> List(Sentence)

    sentences = []
    sents_file = Path('./data/sentences.pickle')
    if Path(sents_file).exists() and try_cached:
        logger.info('Loading sentences from cache %s', sents_file)
        with open(str(sents_file), 'rb') as pickle_file:
            sentences = pickle.load(pickle_file)
        logger.info('Loaded %d sentences from cache', len(sentences))
    else:
        nlp = spacy.load('de')
        for politician in POLITICIANS:
            logger.info('Extracting sentences of %s...', politician)
            speeches = read_speeches(politician=politician)
            for speech_id, speech in enumerate(speeches):
                doc = nlp(speech)  # type: Doc
                sent_id = 0
                for sent in doc.sents:
                    # check if valid sentence
                    if sent.text.startswith('-') or len(sent) < 3:
                        continue

                    sent_id += 1
                    words = [e.text for e in sent]
                    sentences.append(Sentence(words=words, politician=politician,
                                              sent_id=sent_id, speech_id=speech_id))
            logger.info('Extracted sentences. Current count %d', len(sentences))
        with open(str(sents_file), 'wb') as pickle_file:
            pickle.dump(sentences, pickle_file)
            logger.info('Saved extracted sentences to %s', str(sents_file))

    return sentences


def merge():
    # extract all single speeches
    all_speeches = []
    for politician in POLITICIANS:
        speeches = read_speeches(politician=politician)
        all_speeches.extend(speeches)

    # merge speeches
    logger.info('Merging %d speeches', len(all_speeches))
    merged_speeches = ' '.join(all_speeches)

    # write to file
    with open('./data/merged.txt', 'wt', encoding='utf8') as speeches_file:
        speeches_file.write(merged_speeches)

    return merged_speeches


def db_upload(file_path):
    # type: (Path) -> None
    try:
        dbx = dropbox.Dropbox(DROPBOX_TOKEN)
        mode = dropbox.files.WriteMode.overwrite
        db_path = str(DROPBOX_SESSION_PATH.joinpath(file_path))
        with open(str(file_path), 'rb') as f:
            data = f.read()
        dbx.files_upload(data, db_path, mode, mute=False)
        logger.info('Dropbox: Uploaded %s to %s', file_path, db_path)
    except dropbox.exceptions.ApiError as err:
        logger.error('Dropbox: API error', err)


def extract_word_vectors(sentences, try_cached=True):
    # check if already extracted
    if Path(VOCAB_VECTORS).exists() and try_cached:
        with open(VOCAB_VECTORS, 'rb') as pickle_file:
            return pickle.load(pickle_file)

    # creating dictionary
    words_speeches = {w.lower() for s in sentences for w in s.words}

    # download word vectors if necessary
    local_vec_file = Path('wiki.de.vec').absolute()
    if not local_vec_file.exists():
        logger.info('Downloading word vectors. This might take a while... ')
        download_word_vectors(str(local_vec_file))

    # extract necessary word vectors
    word_vectors = OrderedDict()
    vec_iter = 0
    with open(str(local_vec_file), 'r') as vector_file:
        for line in vector_file:
            columns = line.split()
            if len(columns) == 301:  # word plus vector
                word = columns[0]
                if word in words_speeches:
                    if word in word_vectors:  # word may appear twice
                        continue
                    vector = [float(v) for v in columns[-300:]]
                    word_vec = WordVector(id=len(word_vectors), word=word, vector=vector)
                    word_vectors[word] = word_vec

                vec_iter += 1
                if vec_iter % 50000 == 0:
                    logger.info('Checked {} words. Matches: {}'.format(
                        vec_iter, len(word_vectors)))

    logger.info('Matches: {}. Not found: {}'.format(
        len(word_vectors), len(words_speeches) - len(word_vectors)))

    # store extracted vectors
    with open(VOCAB_VECTORS, 'wb') as pickle_file:
        pickle.dump(word_vectors, pickle_file)

    logger.info('Wrote {} word vectors to {}'.format(len(word_vectors), VOCAB_VECTORS))

    return word_vectors


def download_word_vectors(local_file):
    def reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                         (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()

    logger.info('Downloading word vectors. This might take a while... ')
    wordvec_url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec'
    urllib.request.urlretrieve(wordvec_url, local_file, reporthook)


class SpeechSequence(Sequence):

    def __init__(self, sentences, output_size, batch_size, word_vectors, sequence_len,
                 oov_token=None):
        self.batch_size = batch_size
        self.sequence_length = sequence_len
        self.oov_token = oov_token or '<UNK>'
        self.raw_sentences = None
        self.input_encoded = None
        self.sequences = None
        self.next_words = None
        self.corpus_size = None
        self.input_words = None
        self.output_encoded = None
        self.words_raw = [w.lower() for sent in sentences for w in sent.words]

        # build input vocabulary
        self.word_vectors = word_vectors.copy()
        self.word_vectors[self.oov_token] = WordVector(word=self.oov_token,
                                                       id=len(word_vectors),
                                                       vector=[0.0] * 300)
        self.input_vocab = {w: wv.id for w, wv in self.word_vectors.items()}
        self.input_word_ids = {v: k for k, v in self.input_vocab.items()}
        self.input_vocab_size = len(self.word_vectors)
        self.input_unk_id = len(word_vectors)

        # tokenize and build OUTPUT vocabulary
        logger.debug('Building output vocablary...')
        word_counts_raw = Counter(self.words_raw)
        most_com = word_counts_raw.most_common(output_size - 1)  # oov token will be added
        output_w = sorted([tup[0] for tup in most_com])
        self.output_vocab = {w: i for i, w in enumerate(output_w) if i < output_size}
        self.output_vocab[self.oov_token] = len(self.output_vocab)  # last element is oov
        self.output_word_ids = {v: k for k, v in self.output_vocab.items()}
        self.output_unk_id = self.output_vocab[self.oov_token]
        self.output_vocab_size = len(self.output_vocab)
        self.output_word_counts = {w: c for w, c in most_com}
        output_unks = sum([v for k, v in word_counts_raw.items()
                           if k not in self.output_vocab])
        self.output_word_counts[self.oov_token] = output_unks
        logger.debug('Tokenizied OUTPUT words. Vocab size: %d, Corpus size: %d, Unks %d',
                     self.output_vocab_size, len(self.words_raw), output_unks)

        # encoding words
        logger.debug('Encoding words...')
        input_words = []
        for word in self.words_raw:
            input_word = word if word in self.word_vectors else self.oov_token
            input_words.append(input_word)

        # count words in vocabulary
        self.input_word_counts = Counter(input_words)
        self.input_vocab_size = len(self.input_word_counts)
        input_corpus_size = len(input_words)
        logger.debug('Tokenizied INPUT words. Vocab size: %d, Corpus size: %d, Unks %d',
                     self.input_vocab_size, input_corpus_size,
                     self.input_word_counts[self.oov_token])

    def save(self, path=None):
        path = path or DATASET_FILE
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    @staticmethod
    def load(path):
        path = path or DATASET_FILE
        with open(path, 'rb') as pickle_file:
            return pickle.load(pickle_file)

    def adapt(self, sentences):

        # encode words
        words = [w.lower() for sent in sentences for w in sent.words]
        logger.info('Adapting {} words'.format(len(words)))
        self.input_encoded = [self.input_vocab.get(w, self.input_unk_id)
                              for w in words]
        self.output_encoded = [self.output_vocab.get(w, self.output_unk_id)
                               for w in words]

        # create word sequences
        input_sequences = list()
        output_sequences = list()
        logger.debug('Creating training sequences...')
        for i in range(self.sequence_length, len(self.input_encoded)):
            input_sequence = self.input_encoded[i - self.sequence_length:i + 1]
            input_sequences.append(input_sequence)

            output_sequence = self.output_encoded[i - self.sequence_length:i + 1]
            output_sequences.append(output_sequence)
        logger.debug('Created sequences. Total Sequences: %d' % len(input_sequences))

        # split into x and y elements
        input_sequences = np.array(input_sequences)
        output_sequences = np.array(output_sequences)
        self.sequences, self.next_words = input_sequences[:, :-1], output_sequences[:, -1]

    def in_to_out(self, word_id):
        word = self.input_word_ids.get(word_id, self.oov_token)
        return self.output_vocab.get(word, self.output_unk_id)

    def out_to_in(self, word_id):
        word = self.output_word_ids.get(word_id, self.oov_token)
        return self.input_vocab.get(word, self.input_unk_id)

    def decode_input(self, encoded):
        return [self.input_word_ids[e] for e in encoded]

    def decode_input_string(self, encoded):
        return ' '.join(self.decode_input(encoded))

    def decode_output(self, encoded):
        return [self.output_word_ids[e] for e in encoded]

    def decode_output_string(self, encoded):
        return ' '.join(self.decode_output(encoded))

    def encode_output(self, words):
        return [self.output_vocab.get(w.lower(), self.output_unk_id) for w in words]

    def encode_input(self, words):
        return [self.input_vocab.get(w.lower(), self.input_unk_id) for w in words]

    def __len__(self):
        return int(np.ceil(len(self.sequences) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.sequences[idx * self.batch_size:(idx + 1) * self.batch_size]
        next_words = self.next_words[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = to_categorical(next_words, num_classes=self.output_vocab_size)

        return batch_x, batch_y


def convert_vocab():
    with open('./data/dataset.pickle', 'rb') as pickle_file:
        dataset = pickle.load(pickle_file)  # type: SpeechSequence

    vocab = {'input': dataset.input_vocab, 'output': dataset.output_vocab}
    with open('./data/raw_vocab.pickle', 'wb') as pickle_file:
        pickle.dump(vocab, pickle_file)

    logger.info('Converted vobabulary')


if __name__ == '__main__':
    # collect data from open data portal
    # collect()

    # splits data into text files for each politician
    # pre_process()

    # merges all speeches into one text file
    # merge()

    # extracts sentences and assign them to politicians
    # sentences = extract_sentences(try_cached=True)
    # word_vecs = extract_word_vectors(sentences)
    #
    # dataset = SpeechSequence(sentences=sentences, output_size=5000, batch_size=50,
    #                          word_vectors=word_vecs, sequence_len=15, oov_token='<UNK>')
    # dataset.adapt(sentences=sentences)

    convert_vocab()
