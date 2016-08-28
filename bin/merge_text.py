from __future__ import print_function, unicode_literals, division
# -*- coding: utf-8 -*-
#import io
#import bz2
#import logging
#from toolz import partition
from os import path
import os
import re
import time

import spacy.en


from joblib import Parallel, delayed
import plac
try:
    import ujson as json
except ImportError:
    import json


#LABELS = {
#    'ENT': 'ENT',
#    'PERSON': 'ENT',
#    'NORP': 'ENT',
#    'FAC': 'ENT',
#    'ORG': 'ENT',
#    'GPE': 'ENT',
#    'LOC': 'ENT',
#    'LAW': 'ENT',
#    'PRODUCT': 'ENT',
#    'EVENT': 'ENT',
#    'WORK_OF_ART': 'ENT',
#    'LANGUAGE': 'ENT',
#    'DATE': 'DATE',
#    'TIME': 'TIME',
#    'PERCENT': 'PERCENT',
#    'MONEY': 'MONEY',
#    'QUANTITY': 'QUANTITY',
#    'ORDINAL': 'ORDINAL',
#    'CARDINAL': 'CARDINAL'
#}

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
    #'DATE': 'DATE',
    #'TIME': 'TIME',
    #'PERCENT': 'PERCENT',
    #'MONEY': 'MONEY',
    #'QUANTITY': 'QUANTITY',
    #'ORDINAL': 'ORDINAL',
    #'CARDINAL': 'CARDINAL'
}

def parallelize(func, iterator, n_jobs, extra, backend='multiprocessing'):
    extra = tuple(extra)
    return Parallel(n_jobs=n_jobs, backend=backend)(delayed(func)(*(item + extra))
                    for item in iterator)

def iter_lines(loc):
    with open(loc,'r',encoding='utf-8') as file_:
        for line in file_:
            yield line

pre_format_re = re.compile(r'^[\`\*\~]')
post_format_re = re.compile(r'[\`\*\~]$')
url_re = re.compile(r'\[([^]]+)\]\(%%URL\)')
link_re = re.compile(r'\[([^]]+)\]\(https?://[^\)]+\)')

def strip_meta(text):
    text = link_re.sub(r'\1', text)
    text = text.replace('&gt;', '>').replace('&lt;', '<').replace('&nbsp;',' ')
    text = pre_format_re.sub('', text)
    text = post_format_re.sub('', text)
    return text


def parse_and_transform(batch_id, input_, out_dir,n_threads,batch_size,noun_chunker):
    out_loc = path.join(out_dir, os.path.split(input_)[1])
    if path.exists(out_loc):
        return None
    print('Batch', batch_id)
    nlp = spacy.en.English()
    nlp.matcher = None
    timer = time.time()
    tokens = 0
    with open(out_loc, 'w', encoding='utf-8') as file_:
        texts = (strip_meta(text) for text in iter_lines(input_))
        texts = (text for text in texts if text.strip())
        for doc in nlp.pipe(texts, batch_size=batch_size, n_threads=n_threads):
            file_.write(transform_doc(doc,noun_chunker))
            tokens+=len(doc)
    deltat=tokens/(time.time()-timer)
    print('tokenized, merged and wrote at {:2f} tok/sec'.format(deltat))

def transform_doc(doc,noun_chunker):

    for ent in doc.ents:
        if ent in LABELS.keys():
            ent.merge(ent.root.tag_, ent.text, LABELS[ent.label_])

    if noun_chunker:
        for np in list(doc.noun_chunks):
            while len(np) > 1 and np[0].dep_ not in ('advmod', 'amod', 'compound'):
                np = np[1:]
            np.merge(np.root.tag_, np.text, np.root.ent_type_)

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
    in_loc=("Location of input directory or input file"),
    out_dir=("Location of output directory"),
    n_workers=("Number of workers", "option", "n", int),
    n_threads=("Number of threads per process", "option", "t", int),
    batch_size=("Number of texts to accumulate in a buffer", "option", "b", int),
    noun_chunker=("Flag if only noun chunks should be merged","flag","s")
)
def main(in_loc, out_dir, n_workers=4, n_threads=1, batch_size=10000,noun_chunker=False):
    if not path.exists(out_dir):
        path.join(out_dir)
    # if input is only one file, do single processing TODO: enable multiprocessing for single files
    if path.isfile(in_loc):
        parse_and_transform(0,in_loc,out_dir,n_threads=n_threads,batch_size=10000,noun_chunker=noun_chunker)
    else:
        textfiles = [path.join(in_loc, fn) for fn in os.listdir(in_loc)]
        if n_workers >= 2:
            do_work = parse_and_transform
            parallelize(do_work, enumerate(textfiles), n_workers, [out_dir, n_threads, batch_size,noun_chunker],backend='multiprocessing')
        else:
            [parse_and_transform(0, file, out_dir, n_threads, batch_size, noun_chunker) for file in textfiles]


if __name__ == '__main__':
    plac.call(main)