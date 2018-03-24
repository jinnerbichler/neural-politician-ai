FROM jinnerbichler/neural-politician:latest

EXPOSE 6006

COPY . .

ENTRYPOINT ["/usr/bin/python3", "char_rnn.py"]