import pickle
import feedparser
import re
import unicodedata
import itertools
import logging
from datetime import datetime
from time import mktime, sleep

import requests
import sys
from bs4 import BeautifulSoup
from collections import defaultdict

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger('scraper')
logger.setLevel(logging.DEBUG)

PARLAMENT_BASE_URL = 'https://www.parlament.gv.at'
POLITICIANS = ['kurz', 'kern', 'strache']
SPEECHES_PICKLE = './data/speeches.pickle'

PERIOD_FEEDS = {
    'XXVI': 'https://www.parlament.gv.at/PAKT/PLENAR/filter.psp?view=RSS&RSS=RSS&jsMode=RSS&xdocumentUri=%2FPAKT%2FPLENAR%2Findex.shtml&view=RSS&NRBRBV=NR&GP=XXVI&R_SISTEI=SI&LISTE=Anzeigen&listeId=1070&FBEZ=FP_007',
    'XXV': 'https://www.parlament.gv.at/PAKT/PLENAR/filter.psp?view=RSS&RSS=RSS&jsMode=RSS&xdocumentUri=%2FPAKT%2FPLENAR%2Findex.shtml&view=RSS&NRBRBV=NR&GP=XXV&R_SISTEI=SI&LISTE=Anzeigen&listeId=1070&FBEZ=FP_007'
}


def collect():
    all_speeches = defaultdict(list)
    for period, feed_url in PERIOD_FEEDS.items():

        logger.info('Processing period {} ({})'.format(period, feed_url))

        feed = feedparser.parse(feed_url)
        fetched_sessions = []  # avoid fetching a session twice
        for session in reversed(feed['items']):

            # extract session information
            title = session['title']
            published = datetime.fromtimestamp(mktime(session['published_parsed']))
            session_url = session['link']

            # check if sessions has already been fetched
            if title in fetched_sessions:
                continue
            fetched_sessions.append(title)

            logger.info('Fetching "{}" ({})'.format(title, session_url))

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
                    sleep(0.5)

            num_speeches = sum([len(s) for s in all_speeches.values()])
            logger.info('Current speech count: {}'.format(num_speeches))

    num_speeches = sum([len(s) for s in all_speeches.values()])
    logger.info('Total speech count: {}'.format(num_speeches))

    # store speeches
    with open(SPEECHES_PICKLE, 'wb') as pickle_file:
        pickle.dump(all_speeches, pickle_file)


def parse_protocol(url):
    speeches = defaultdict(list)

    # fetch protocol
    response = requests.get(url)
    response_text = response.text.replace('&shy;', '')  # remove hyphens
    soup = BeautifulSoup(response_text, 'html5lib')

    for section_counter in itertools.count(start=1):
        # extract relevant pargraphs
        section = soup.find('div', class_='WordSection{}'.format(section_counter))
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

        section_counter += 1
        if not role or not name:
            continue

        # collect speeches of targeted politicians
        for politician in POLITICIANS:
            if politician in name.lower():
                logger.debug('Found speech: name: {}, role: {}, party: {}'.format(
                    name, role, party))
                speeches[politician].append({'name': name, 'role': role,
                                             'party': party, 'speech': speech})
    return speeches


def pre_process():
    with open(SPEECHES_PICKLE, 'rb') as pickle_file:
        speeches = pickle.load(pickle_file)

    for politician, speeches in speeches.items():
        with open('./data/{}.txt'.format(politician), 'w') as speeches_file:
            num_char = 0
            num_words = 0
            for speech in speeches:
                # write header and text of speech to file
                sess = speech['session']
                header = '# {period:} - {title:} {published:} ({url:})\n'.format(**sess)
                speeches_file.write(header)
                speeches_file.write(speech['speech'] + '\n\n')

                # count metrics
                num_char += len(speech['speech'])
                num_words += len(speech['speech'].split())

            logger.info('Metrics of {}: chars: {}, words: {}'.format(
                politician, num_char, num_words))


if __name__ == '__main__':
    collect()
    pre_process()
