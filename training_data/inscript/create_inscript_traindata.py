#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Parse InScript corpus and extract its text content alogside annotations.
This script works with InScript corpus and provides a number of useful methods
that can be used to extract various information from the corpus.

List of corpus  dirs:
    "bath", "bicycle", "bus", "cake", "flight", "grocery", "haircut",
    "library", "train", "tree"

Brief InScript xml structure:
<story>
    <text>
        <content>
        </content>
        <sentences>
            <sentence>
                <token>
                    <pos/>
                    <lemma/>
                </token>
            </sentence>
        </sentences>
    </text>
    <annotations>
        <participants>
        </participants>
        <events>
        </events>
        <chains>
            <chain/>
        </chains>
    </annotations>
</story>

"""
from __future__ import division
import sys
import os
import re
import math
import shutil
import xml.etree.ElementTree as et
from collections import defaultdict, Counter
from INSCRIPT_EV_MAP import LABEL2ID


# set corpus path
corp_path = os.path.join("InScript", "corpus")
# list of InScript scenario paths
SPATHS = [os.path.join(corp_path, d) for d in os.listdir(corp_path)
          if os.path.isdir(os.path.join(corp_path, d))]
TOPICS = ["bath", "bicycle", "bus", "cake", "flight", "grocery",
          "haircut", "library", "train", "tree"]
TOPIC2ID = {
                "bath": 1,
                "bicycle": 2,
                "bus": 3,
                "cake": 4,
                "flight": 5,
                "grocery": 6,
                "haircut": 7,
                "library": 8,
                "train": 9,
                "tree": 10
            }


def build_corpus_paths():
    """Create and return a map of all corpus xml file paths.
    {"bath": [<xml file paths>], "bicycle": [<xml file paths>], ...}
    """
    corpus = {}
    for spath in SPATHS:
        sdir = os.path.basename(spath)
        corpus[sdir] = map(lambda x: os.path.join(spath, x), os.listdir(spath))
    return corpus


def extract_token_labels_map(events=False, participants=False, blacklist=None):
    """
    Parse all files in a given scenario dir and return a new dict where
    {fpath: ([[sent1], [sent2], ...],
        {(from, to, token): event/participant type})}

    You can pick which event/participant labels to filter out by providing a
    list of label names, e.g. ["UnrelEv", "NPart"]

    Args:
        events (bool) -- extract event labels
        participants (bool) -- extract participant labels
        blacklist (set) -- a set of event/participant labels to exclude

    """
    scenario_map = {}  # {story path: ([sents], {from: (to, token, label)})}
    pathmap = build_corpus_paths()

    for topic in TOPICS:
        for fpath in pathmap[topic]:
            root = et.parse(fpath).getroot()
            # tokenizing
            content = [sent.split() for sent in root[0][0].text.split('\n')]
            event_map = {}  # event labels map
            part_map = {}  # participant labels map

            if events:
                # accessing event tag level
                for label in root[1][1]:
                    # filter out blacklist labels
                    if label.attrib["name"] in blacklist:
                        continue
                    event_map[label.attrib["from"]] = (label.attrib.get("to"),
                                                       label.attrib["text"],
                                                       label.attrib["name"])

            if participants:
                # and participant tag level
                for label in root[1][0]:
                    # filter out blacklist labels
                    if label.attrib["name"] in blacklist:
                        continue
                    part_map[label.attrib["from"]] = (label.attrib.get("to"),
                                                      label.attrib["text"],
                                                      label.attrib["name"])

            labels_map = {}
            labels_map.update(event_map)
            labels_map.update(part_map)
            scenario_map[fpath] = (content, labels_map)

    return scenario_map


def create_train_data(ratio):
    """Use extracted scenario_map in order to create training InScript data as
    well as corresponding vectors data.
    Args:
        ratio (list) -- train/valid ratio e.g. [0.9, 0.1]
    """
    # compute which story belongs to valid set
    each_valid = math.floor(910 / (910 * ratio[1]))
    print "Each %d story is for validation" % each_valid

    # blacklisted labels go here
    black = []  #['UnrelEv', 'RelNScrEv']
    scenario_map = extract_token_labels_map(events=True, participants=False,
                                            blacklist=black)

    decay_vectors = []  # train data for network 2 (event predictor)
    decay_vectors_valid = []  # valid data for network 2
    tokens_and_vectors = []  # train token, vector tuple
    tokens_and_vectors_valid = []  # valid token, vector tuple
    sid, each = 1, 0
    token_freqs = defaultdict(int)  # dict of {token: {event: cnt}}
    token_event_freqs = defaultdict(int)  # dict of {token: cnt}

    # iterate over each scenario story
    for n, story in enumerate(scenario_map.items()):
        each += 1
        scenario = story[0]
        sents, labels_map = story[1][0], story[1][1]  # sents == story
        split_label = False
        nodecay_tokens = []

        # iterate sentences
        for sid, sent in enumerate(sents, 1):
            tokens = []  # tokens == story sentence
            diff = 0  # index diff
            dint = defaultdict(int)  # label id: value

            # iterate tokens
            for tokid, token in enumerate(sent, 1):
                # create current token idx
                idx = ''.join([str(sid), '-', str(tokid)])
                islabel = labels_map.get(idx)  # current label if exists
                to, label = None, None

                """
                curlabel = islabel[-1] if islabel is not None else 'NONE'
                token_freqs[token] += 1
                if not token_event_freqs.get(token):
                    token_event_freqs[token] = defaultdict(int)
                if curlabel != 'NONE':
                    token_event_freqs[token][curlabel] += 1
                """

                if islabel is not None:
                    to, _, label = islabel

                if to is not None and not split_label:
                    diff = map(int, to.split('-'))[1] - tokid
                    last = label
                    split_label = True
                    diff -= 1
                    tokens.append([token, last, dint, False])

                elif to is None and split_label and diff != 0:
                    diff -= 1
                    tokens.append([token, last, dint, False])

                elif split_label and diff == 0:
                    dint[LABEL2ID[last]] += 1
                    split_label = False
                    tokens.append([token, last, dint, True])
                else:
                    if label is not None:
                        dint[LABEL2ID[label]] += 1
                    else:
                        dint[-1] += 1  # -1 no event id
                    single = True if label is not None else False
                    tokens.append([token, label, dint, single])

            nodecay_tokens.append(tokens)

        # decay label information according to ideces
        for n, sent in enumerate(nodecay_tokens, 1):
            for token in sent:
                if token[1] is None:
                    token[2][-1] -= 1  # no event vector idx

                elif token[1]:
                    if token[3]:
                        token[2][LABEL2ID[token[1]]] -= 1

                # create vector representation
                vrepr = [[str(TOPIC2ID[scenario.split('/', 3)[2]])]] +\
                        [(k, v) for k, v in token[2].items()]

                # 18 for 0, LookupTable does not like zeros and negatives
                decvec = sum([i[1] for i in vrepr[1:] if i[0] != -1]) or 0  # 18
                if each == each_valid:
                    decay_vectors_valid.append(decvec)
                else:
                    decay_vectors.append(decvec)

                #vrepr = [(TOPIC2ID[scenario.split('/', 3)[2]], )] +\
                #        [(k, v) for k, v in token[2].items()]
                # uncomment below for debugging
                # sys.stdout.write('{0:<15} {1:<7} {2:<25} {3}\n'.format(token[0], scenario, token[1], vrepr))
                if each == each_valid:
                    tokens_and_vectors_valid.append((token[0].lower(), vrepr))
                else:
                    tokens_and_vectors.append((token[0].lower(), vrepr))

            # compensate for <eos> token, add additional last vector
            if each == each_valid:
                tokens_and_vectors_valid.append(('<eos>', vrepr))
            else:
                tokens_and_vectors.append(('<eos>', vrepr))

            # decay_vectors.append(decvec)  # <eos> will be automatically added
            if each == each_valid:
                decay_vectors_valid.append("END")
            else:
                decay_vectors.append("END")

        if each == each_valid:
            each = 0
    """
    freqrows = []
    for t, cnt in token_freqs.items():
        if token_event_freqs[t]:
            lst = token_event_freqs[t].items()
            lst.sort(key=lambda x: x[1], reverse=True)
            evs = ', '.join([str(b) for a in lst for b in a])
            freqrows.append([t, cnt, evs])

    for r in sorted(freqrows, key=lambda x: x[1], reverse=True):
        print '\t'.join([str(i) for i in r])

    exit()
    """
    return tokens_and_vectors, decay_vectors, tokens_and_vectors_valid, decay_vectors_valid


def write_data(data, fname, single):
    # replace all digits with 0 and punctuation with <punct>
    data2 = []
    found, found_punct = 0, 0
    for elem in data:
        if re.match(r'(?:[-\/,\.])?[0-9]+(?:[-\/,\.])?', elem[0]):
            elem = ('0', elem[1])
            found += 1
        if re.match(r'[_\W]+$', elem[0]):
            elem = ('<punct>', elem[1])
            found_punct += 1
        data2.append(elem)

    print 'Found %d digits, replacing with 0s' % found
    print 'Found %d punctuation, replacing with <punct>' % found_punct

    ftrain = open('traindata/%s.txt' % fname, 'a')
    fvecs = open('traindata/%s_vecs.csv' % fname, 'a')
    ftrain.write(' ')  # adding first line whitespace
    for elem in data2:
        token, vec = elem[0], elem[1]
        # replace singletons with <unk> label
        if single.get(token):
            token = '<unk>'
        # write training tokens
        if token != '<eos>':
            ftrain.write(''.join([token, ' ']))
        else:
            # do not write <eos>, it will be added by loader.data_load()
            ftrain.write('\n ')

        # write corresponding vectors
        vec = [str(i2) for i in vec for i2 in i]
        vec = ','.join(vec).rstrip(',') + '\n'
        fvecs.write(vec)

    ftrain.close()
    print('saved traindata/%s.txt' % fname)
    fvecs.close()
    print('saved traindata/%s_vecs.csv' % fname)


def write_network2_data(data, fname):
    # write network 2 training file
    fdecs = open('traindata/%s_decay.txt' % fname, 'a')
    data = ' '.join([str(s) for s in data]).split('END')

    # writing decay tokens for network 2
    for row in data:
        row_spl = row.split()
        noenvs = len([s for s in row_spl if s == '0'])
        for num in row_spl:
            if num == '0':
                noenvs -= 1
            fdecs.write(' ' + '{0}-{1}'.format(num, noenvs))
            # add <eos> idx 19 here because we don't do it in dataloader.lua
        fdecs.write(' {0}-{1} \n'.format(19, noenvs))

    fdecs.close()
    print('saved traindata/%s_decay.txt' % fname)


def create_train_files():
    """Create train.txt file and corresponding train_vecs.csv file"""
    if os.path.exists('traindata'):
        print('traindata dir exists, removing...')
        shutil.rmtree('traindata')
    os.mkdir('traindata')

    traindata, decay_tr, validata, decay_valid = create_train_data(ratio)

    # detect singletons
    tokcount = Counter([elem[0] for elem in traindata + validata])
    single = dict(filter(lambda x: x[1] == 1, tokcount.items()))  # singletons
    print 'Found %d singetons, replacing with <unk>' % len(single)

    write_data(traindata, 'train', single)
    write_data(validata, 'valid', single)
    write_network2_data(decay_tr, 'train')
    write_network2_data(decay_valid, 'valid')


def main():
    create_train_files()


if __name__ == '__main__':
    assert(len(sys.argv) == 3), 'Specify train/valid ratio, e.g. 0.8 0.2'
    if float(sys.argv[1]) + float(sys.argv[2]) != 1.0:
        print('ERROR: incorrect train/valid ratio')
        sys.exit(0)

    ratio = [float(sys.argv[1]), float(sys.argv[2])]
    main()
