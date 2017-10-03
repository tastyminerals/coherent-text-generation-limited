#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Measuring sentence coherence following Barzilay-Lapata [2005] wordnet based
measures.

We are using pywsd library and wordnet since pywsd wordnet API is buggy:
    pywsd.lesk available algorithms:
        original_lesk, simple_lesk, adapted_lesk, cosine_lesk

    Original Lesk (Lesk, 1986)
    Adapted/Extended Lesk (Banerjee and Pederson, 2002/2003)
    Simple Lesk (with definition, example(s) and hyper+hyponyms)
    Cosine Lesk (use cosines to calculate overlaps instead of using raw counts)

    pywsd.similarity function options:
        path,
        lch (Leacock-Chodorow),
        wup (Wu-Palmer),
        jcn (Jiang-Conrath),
        res (Resnik),
        lin (Lin)

path [0, 1]:
    Return a score denoting how similar two word senses are, based on the
    shortest path that connects the senses in the is-a (hypernym/hypnoym)
    taxonomy. The score is in the range 0 to 1. By default, there is now a
    fake root node added to verbs so for cases where previously a path could
    not be found---and None was returned---it should return a value. The old
    behavior can be achieved by setting simulate_root to be False. A score of
    1 represents identity i.e. comparing a sense with itself will return 1.

lch [0, 3.6 depending on taxonomy depth]:
    Return a score denoting how similar two word senses are, based on the
    shortest path that connects the senses (as above) and the maximum depth of
    the taxonomy in which the senses occur. The relationship is given as
    -log(p/2d) where p is the shortest path length and d the taxonomy depth.

wup [0, 1]:
    Return a score denoting how similar two word senses are, based on the depth
    of the two senses in the taxonomy and that of their Least Common Subsumer
    (most specific ancestor node). Note that at this time the scores given
    do _not_ always agree with those given by Pedersen's Perl implementation
    of Wordnet Similarity.

    The LCS does not necessarily feature in the shortest path connecting the
    two senses, as it is by definition the common ancestor deepest in the
    taxonomy, not closest to the two senses. Typically, however, it will so
    feature. Where multiple candidates for the LCS exist, that whose shortest
    path to the root node is the longest will be selected. Where the LCS has
    multiple paths to the root, the longer path is used for the purposes of
    the calculation.

res [-0.0, variable]:
    Return a score denoting how similar two word senses are, based on the
    Information Content (IC) of the Least Common Subsumer (most specific
    ancestor node). Note that for any similarity measure that uses information
    content, the result is dependent on the corpus used to generate the
    information content and the specifics of how the information content was
    created.

jcn [0?, 1e+300]:
    Return a score denoting how similar two word senses are, based on the
    Information Content (IC) of the Least Common Subsumer (most specific
    ancestor node) and that of the two input Synsets. The relationship is
    given by the equation: 1 / (IC(s1) + IC(s2) - 2 * IC(lcs)).

lin [0, 1]:
    Return a score denoting how similar two word senses are, based on the
    Information Content (IC) of the Least Common Subsumer (most specific
    ancestor node) and that of the two input Synsets. The relationship is
    given by the equation: 2 * IC(lcs) / (IC(s1) + IC(s2)).
"""

from __future__ import division
from itertools import permutations as permute
from pywsd import disambiguate  # does preprocessing under the hood
from pywsd.lesk import original_lesk, simple_lesk, adapted_lesk, cosine_lesk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wic
from nltk.corpus.reader.wordnet import WordNetError


def dis(sent, lesk, simopt):
    """Assign sense to all words in a given sentence.

    Args:
        sent (str) -- sentence string
        lesk (str) -- type of pywsd lesk algorithm [orig, simple, adapted, cos]
        simopt (str) -- type of pywsd similarity option [path, wupa, lch,
        jcn, res, lin]
    Return:
        a tuple of disambiguated words
    """
    if lesk in ['orig', 'original']:
        return disambiguate(sent, algorithm=original_lesk,
                            similarity_option=simopt,
                            keepLemmas=False)
    elif lesk == 'simple':
        return disambiguate(sent, algorithm=simple_lesk,
                            similarity_option=simopt,
                            keepLemmas=False)
    elif lesk == 'adapted':
        return disambiguate(sent, algorithm=adapted_lesk,
                            similarity_option=simopt,
                            keepLemmas=False)
    elif lesk == 'cos':
        return disambiguate(sent, algorithm=cosine_lesk,
                            similarity_option=simopt,
                            keepLemmas=False)


def sim(opt, lesk_algorithm, *args):
    """Compute semantic similarity between two sentences

    Information content based similarity measures:
        wn.jcn_similarity
        wn.res_similarity
        wn.lin_similarity

    Path length based similarity measures:
        wn.path_similarity
        wn.lch_similarity
        wn.wup_similarity

    """
    # set similarity algs
    dsims = {'jcn': wn.jcn_similarity,
             'res': wn.res_similarity,
             'lin': wn.lin_similarity,
             'path': wn.path_similarity,
             'lch': wn.lch_similarity,
             'wup': wn.wup_similarity}

    # information content based similarity measures
    bnc = None
    if opt in ['jcn', 'lin']:
        bnc = wic.ic('ic-bnc-add1.dat')
    elif opt in ['res', 'resnik']:
        bnc = wic.ic('ic-bnc-resnik-add1.dat')

    # disambiguate each sentence
    dissents = [(n, dis(sent, lesk_algorithm, simopt=opt)) for n, sent
                in enumerate(args)]

    if len(dissents) == 2 and dissents[0][1] is None and dissents[1][1] is None:
        print "WARNING: Lesk algorithm failed to disambiguate any words!"
        return 0

    # filter out words without found synsets
    senses = [(sent[0], word) for sent in dissents for word in sent[1]
              if word[1] is not None]

    # create permuations of word pairs from each sentence
    perms0 = [p for p in permute(senses, 2)]

    # filter out synsets from the same sentence
    perms1 = [p for p in perms0 if p[0][0] != p[1][0]]

    # filter out AB == BA cases
    perms2 = set()
    pos = ['n']  # set pos filter
    for p in perms1:
        # we also filter out pairs with adjectives
        if p[0][1][1].pos() not in pos or p[1][1][1].pos() not in pos:
            continue
        if p[::-1] not in perms2:
            perms2.add(p)

    # compute similarity scores
    scores = []
    for p in perms2:
        try:
            score = dsims[opt](p[0][1][1], p[1][1][1], bnc)
            print p[0][1][1], p[1][1][1], score
            if score is None:
                continue
            if opt == 'jcn' and score > 1:
                score = 1.0
            scores.append(score)
        except WordNetError as err:
            print err

    # apply Barzilay-Lapata similarity measure, no argmax due to wsd
    # S1 = len(dissents[0][1])
    # S2 = len(dissents[1][1])

    # S1 = len(args[0].split())  # n sentence length
    # S2 = len(args[1].split())  # n+1 sentence length
    return sum(scores)  # / (S1 * S2)


def coherence(text, opt="wup", lesk_algorithm="adapted", norm=True):
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
            s = sim(opt, lesk_algorithm, sents[n], sents[n+1])

        # normalize by sent lengths
        if norm and s != 0:
            s = s / (len(sents[n].split()) + len(sents[n+1].split()))

        print "\n%s --> %s: %f" % (sents[n], sents[n+1], s)
        print '-' * 100
        sims.append(s)

    return sum(sims) / (n + 1)


if __name__ == '__main__':
    pass
