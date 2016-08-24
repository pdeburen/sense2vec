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
#clean_re =re.compile('<.*?>')

def strip_meta(text):
    text = link_re.sub(r'\1', text)
    #text = re.sub(clean_re, '!!', text)
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
    if list(doc.noun_chunks):
        for np in doc.noun_chunks:
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


#def transform_docs_blog(doc):
#    # Iterate over base NPs, e.g. "all their good ideas"
#    for np in doc.noun_chunks:
#        # Only keep adjectives and nouns, e.g. "good ideas"
#        while len(np) > 1 and np[0].dep_ not in ('amod', 'compound'):
#            np = np[1:]
#        if len(np) > 1:
#            # Merge the tokens, e.g. good_ideas
#            np.merge(np.root.tag_, np.text, np.root.ent_type_)
#    # Iterate over named entities
#    for ent in doc.ents:
#        if len(ent) > 1:
#            # Merge them into single tokens
#            ent.merge(ent.root.tag_, ent.text, ent.label_)
#    token_strings = []
#    for token in doc:
#        text = token.text.replace(' ', '_')
#        tag = token.ent_type_ or token.pos_
#        token_strings.append('%s|%s' % (text, tag))
#    return ' '.join(token_strings)

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
    batch_size=("Number of texts to accumulate in a buffer", "option", "b", int)
)
def main(in_loc, out_dir, n_workers=4, n_threads=1, batch_size=10000):
    if not path.exists(out_dir):
        path.join(out_dir)
    if path.isfile(in_loc):
        parse_and_transform(0,in_loc,out_dir,n_threads=1,batch_size=10000)
    else:
        textfiles = [path.join(in_loc, fn) for fn in os.listdir(in_loc)]
        if n_workers >= 2:
            #jobs = partition(200000, textfiles)
            do_work = parse_and_transform
            parallelize(do_work, enumerate(textfiles), n_workers, [out_dir, n_threads, batch_size],backend='multiprocessing')
        else:
            [parse_and_transform(0, file, out_dir, n_threads, batch_size) for file in textfiles]


if __name__ == '__main__':
    plac.call(main)