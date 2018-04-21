FROM jinnerbichler/neural-politician:latest

EXPOSE 6006

RUN pip3 --no-cache-dir install \
        dropbox==8.7.1 \
        tensorflow-gpu==1.7.0

COPY . .

ENTRYPOINT ["/usr/bin/python3", "word_rnn.py"]