#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""Measuring text coherence using sentence vectors comparison as described by
Foltz [1998].

Coherence is calculated by summing cosine similarity scores between sentence
transitions [1->2, 2->3, 3->4].
Cosine similary is given by centroid vectors which are computed from the word
vectors of each sentences.
This script uses word2vec vectors.

Requires:
    csv file with "word, embedding" format

"""

from __future__ import division
import csv
import numpy as np
from nltk.corpus import stopwords
from string import punctuation


# set .t7 file
WORD2VEC_FILE = 'word2vec_reduced.csv'
STOP = stopwords.words('english')


def load_word2vec_file():
    # create dict {word: np.array}
    word2vec = {}
    with open(WORD2VEC_FILE, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            word2vec[line[0]] = np.fromstring(line[1], dtype=float, sep=' ')
    return word2vec


def centroid(sent):
    """Compute sentence vector mean"""
    word2vec = load_word2vec_file()
    s = set([word for word in sent.split()
            if word not in STOP and word not in punctuation])
    # slen = len(s)
    mean = np.zeros(300)
    for word in s:
        if word2vec.get(word) is None:
            continue
        mean = np.add(mean, word2vec[word])

    dotmean = np.sqrt(mean.dot(mean))
    if dotmean == 0:
        print 'Foltz> inf detected!'
        print 'MISBEHAVING: ', sent
        print ''
        return mean * np.array(0)
    return mean / dotmean  # norm sent vec


def sim(sent1, sent2):
    """Compute cosine similarity of vector means between given sentences"""
    # cosine similarity is simply a dot product due to norm in centroid()
    return np.dot(centroid(sent1), centroid(sent2))


def coherence(text, norm=True):
    """Compute text coherence by sentence vectors cosine similarity.

    Args:
        text (string) -- new line delimited sentences
        norm (bool) -- normalize by sent length

    Returns:
        cosine similarity (float)

    """
    sents = [s for s in text.split('\n') if s]
    sims = []
    for n in xrange(len(sents)-1):
        # check if sent2 has similar beginning
        if sents[n].startswith(' '.join(sents[n+1].split()[:3])):
            s = 0
        elif sents[n].endswith(' '.join(sents[n+1].split()[-4:])):
            s = 0
        else:
            s = sim(sents[n], sents[n+1])

        # normalize by sent lengths
        if norm and s != 0:
            s = s / (len(sents[n].split()) + len(sents[n+1].split()))

        print "%s --> %s: %f" % (sents[n], sents[n+1], s)
        sims.append(s)

    return sum(sims) / (n + 1)


if __name__ == '__main__':
    pass
