#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""Simple text coherence evaluation using words overlap.
Overal local text coherence is measured by evaluation similarity between
consequent sentences.

    coherence(T) = sum_{n-1} sim(Si, Si+1) / (n - 1)

where T - is some arbitrary text, n - number of sentence pairs, sim(Si, Si+1)
- is a measure of similarity between corresponding sentences Si and Si+1.

    sim(S1, S2) = 2 (|words(S1) ∩ words(S2)|) / (|words(S1)| + |words(S2)|)

We are filtering out all functional words and stem the sentence words before
constructing sets.

"""

from __future__ import division
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation


STOP = stopwords.words('english')
STEM = SnowballStemmer('english')


def normalize(word):
    """Perform stopword check and stemming."""
    if word not in STOP and word not in punctuation:
        return STEM.stem(word)


def sim(sent1, sent2):
    """Compute overlap similarity score."""
    s1 = set(filter(lambda x: x is not None,
                    [normalize(word) for word in sent1.split()]))
    s2 = set(filter(lambda x: x is not None,
                    [normalize(word) for word in sent2.split()]))
    return 2 * len(s1 & s2) / (len(s1) + len(s2))


def coherence(text, norm=True):
    """Evaluate text coherence.

    Args:
        text (string) -- text with sentences separated by new lines.
        norm (bool) -- normalize by sent length

    Returns:
        coherence value (float)

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
