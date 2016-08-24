from __future__ import print_function, unicode_literals, division
import io
import bz2
import logging
from toolz import partition
from os import path
import os
import re
import fileinput

import spacy.en
from preshed.counter import PreshCounter
from spacy.tokens.doc import Doc

from joblib import Parallel, delayed
import plac
try:
    import ujson as json
except ImportError:
    import json


LABELS = {
    'ENT': 'ENT',
    'PERSON': 'ENT',
    'NORP': 'ENT',
    'FAC': 'ENT',
    'ORG': 'ENT',
    'GPE': 'ENT',
    'LOC': 'ENT',
    'LAW': 'ENT',
    'PRODUCT': 'ENT',
    'EVENT': 'ENT',
    'WORK_OF_ART': 'ENT',
    'LANGUAGE': 'ENT',
    'DATE': 'DATE',
    'TIME': 'TIME',
    'PERCENT': 'PERCENT',
    'MONEY': 'MONEY',
    'QUANTITY': 'QUANTITY',
    'ORDINAL': 'ORDINAL',
    'CARDINAL': 'CARDINAL'
}

def iter_lines(loc):
    with open(loc,'r',encoding='utf-8') as file_:
        for line in file_:
            yield line

pre_format_re = re.compile(r'^[\`\*\~]')
post_format_re = re.compile(r'[\`\*\~]$')
url_re = re.compile(r'\[([^]]+)\]\(%%URL\)')
link_re = re.compile(r'\[([^]]+)\]\(https?://[^\)]+\)')
clean_re =re.compile('<.*?>')

def strip_meta(text):
    text = link_re.sub(r'\1', text)
    # Strip all html-tags
    text = re.sub(clean_re, '!!', text)
    text = text.replace('&gt;', '>').replace('&lt;', '<').replace('&nbsp;',' ')
    text = pre_format_re.sub('', text)
    text = post_format_re.sub('', text)
    return text


def parse_and_transform(batch_id, input_, out_dir,n_threads,batch_size):
    out_loc = path.join(out_dir, os.path.split(input_)[1])
    if path.exists(out_loc):
        return None
    print('Batch', batch_id)
    nlp = spacy.en.English()
    nlp.matcher = None

    with open(out_loc, 'w', encoding='utf8') as file_:
        texts = (strip_meta(text) for text in iter_lines(input_))
        #texts = strip_meta(infile_.read())
        texts = (text for text in texts if text.strip())
        for doc in nlp.pipe(texts, batch_size=batch_size, n_threads=n_threads):
            file_.write(transform_doc(doc))

def transform_doc(doc):
    for ent in doc.ents:
        ent.merge(ent.root.tag_, ent.text, LABELS[ent.label_])
    #for np in doc.noun_chunks:
    #    while len(np) > 1 and np[0].dep_ not in ('advmod', 'amod', 'compound'):
    #        np = np[1:]
    #    np.merge(np.root.tag_, np.text, np.root.ent_type_)
    strings = []
    for sent in doc.sents:
        if sent.text.strip():
            strings.append(' '.join(represent_word(w) for w in sent if not w.is_space))
    if strings:
        return '\n'.join(strings) + '\n'
    else:
        return ''


def represent_word(word):
    if word.like_url:
        return '%%URL|X'
    text = re.sub(r'\s', '_', word.text)
    tag = LABELS.get(word.ent_type_, word.pos_)
    if not tag:
        tag = '?'
    return text + '|' + tag

@plac.annotations(
    in_loc=("Location of input file"),
    out_dir=("Location of input file"),
)
def main(in_loc,out_dir):
    parse_and_transform(0,in_loc,out_dir,n_threads=1,batch_size=10000)

if __name__ == '__main__':
    plac.call(main)