FROM jinnerbichler/neural-politician:latest

EXPOSE 6006

RUN pip3 --no-cache-dir install \
        dropbox==8.7.1

RUN mkdir -p /cache/

COPY . .

ENTRYPOINT ["/usr/bin/python3", "word_rnn.py"]