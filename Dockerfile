FROM python:3
RUN git clone https://github.com/spacy-io/sense2vec.git
RUN pip install -r sense2vec/requirements.txt
RUN pip install -e sense2vec/
RUN pip install toolz
RUN pip install joblib
RUN python -m spacy.en.download
